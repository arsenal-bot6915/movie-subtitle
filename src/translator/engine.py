"""DeepSeek translation engine.

This module handles the end-to-end translation of a batch of subtitle entries:
1. Calls the model with the appropriate prompts.
2. Parses and validates the JSON response.
3. Retries on failure with escalating back-off.
4. Falls back to single-line isolation when the model repeatedly merges entries.
"""

import concurrent.futures
import json
import re
import time
from typing import List, Optional, Tuple

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

from ..core.constants import MAX_RETRIES, RETRY_WAIT_SECONDS
from ..core.models import SubtitleEntry
from ..core.text_processor import (
    contains_chinese,
    smart_wrap_chinese,
    unmask_ass_tags,
)
from .prompts import build_system_prompt, build_user_prompt

JSON_SUPPORTED_MODELS = ["deepseek-v4-flash"]


def _toast(msg: str) -> None:
    """
    Emit a toast if Streamlit runtime is active and we are in the main script thread.
    Silently skips in worker threads to avoid 'missing ScriptRunContext' warnings.
    """
    try:
        import streamlit as st
        import streamlit.runtime.scriptrunner as sr
        if sr.get_script_run_context() is None:
            return  # worker thread — skip silently
        st.toast(msg, icon="🛠️")
    except Exception:
        pass


# ── Validation ────────────────────────────────────────────────────────────────

def extract_and_validate_json(
    original_batch: List[SubtitleEntry],
    text: str,
    strict: bool = True,
) -> Tuple[List[SubtitleEntry], List[int]]:
    """
    Extract a JSON dict from *text* and merge it back into :class:`SubtitleEntry`
    objects, maintaining original timelines.

    Parameters
    ----------
    original_batch : List[SubtitleEntry]
        The entries as sent to the model (preserves index, timeline, tags).
    text : str
        Raw model response content.
    strict : bool
        - ``True`` (default): raise :exc:`ValueError` on any missing key or
          language fingerprint failure, triggering a retry.
        - ``False``: silently fall back to original text for bad entries.

    Returns
    -------
    Tuple[List[SubtitleEntry], List[int]]
        ``(merged_entries, missing_ids)`` where *missing_ids* are the indices
        of entries that fell back to original text.

    Raises
    ------
    ValueError
        In strict mode, when any entry is missing from the JSON, the
        translated text is empty, or the text has no Chinese characters despite
        the original containing ASCII letters.
    """
    last_brace = text.rfind("{")
    if last_brace == -1:
        raise ValueError("未能从模型返回内容中提取到有效的 JSON 格式数据。")

    match = re.search(r"\{[\s\S]*\}", text[last_brace:])
    if not match:
        raise ValueError("未能从模型返回内容中提取到有效的 JSON 格式数据。")

    try:
        translated_dict = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"模型返回的 JSON 格式损坏无法解析: {e}")

    merged: List[SubtitleEntry] = []
    missing_ids: List[int] = []

    for src in original_batch:
        key = str(src.index)
        is_fb = False

        if key not in translated_dict:
            missing_ids.append(src.index)
            trans_text = src.text
            is_fb = True
        else:
            trans_text = str(translated_dict[key]).strip()
            trans_text = smart_wrap_chinese(trans_text)

            if hasattr(src, "tags_map") and src.tags_map:
                trans_text = unmask_ass_tags(trans_text, src.tags_map)

            orig_lines = src.text.strip().split("\n")
            trans_lines = trans_text.split("\n")
            needs_zh_translation = bool(re.search(r"[a-zA-Z0-9]", src.text))

            if not trans_text:
                missing_ids.append(src.index)
                trans_text = src.text
                is_fb = True
                if strict:
                    raise ValueError(
                        f"ID {src.index} 译文为空，疑似被合并吞字。"
                    )
            elif needs_zh_translation and not contains_chinese(trans_text):
                missing_ids.append(src.index)
                trans_text = src.text
                is_fb = True
                if strict:
                    raise ValueError(
                        f"ID {src.index} 没中文字符，疑似偷懒或错位。"
                    )
            elif strict and len(orig_lines) > 1 and len(trans_lines) < 2:
                raise ValueError(
                    f"AI 破坏了双人对话结构。原字幕ID {src.index} 为多行对话，"
                    f"译文却被合并。触发重试。"
                )

        merged.append(
            SubtitleEntry(
                index=src.index,
                timeline=src.timeline,
                text=trans_text,
                is_fallback=is_fb,
                raw_prefix=getattr(src, "raw_prefix", ""),
                tags_map=getattr(src, "tags_map", None),
            )
        )

    if strict and missing_ids:
        raise ValueError(
            f"AI 漏翻了序号为 {missing_ids} 的字幕，校验失败，触发重试。"
        )

    return merged, missing_ids


# ── Main translation function ──────────────────────────────────────────────────

def call_deepseek_translate(
    client: OpenAI,
    model: str,
    batch_entries: List[SubtitleEntry],
    prev_entries_orig: Optional[List[SubtitleEntry]] = None,
    prev_entries_trans: Optional[List[SubtitleEntry]] = None,
    movie_name: str = "",
    movie_bg: str = "",
    glossary: str = "",
    max_retries: int = MAX_RETRIES,
    full_script_text: str = "",
    use_god_mode: bool = False,
    temperature: float = 0.1,
    use_thinking: bool = False,
    reasoning_effort: str = "high",
) -> Tuple[List[SubtitleEntry], List[int]]:
    """
    Translate *batch_entries* using the DeepSeek API with retry and isolation fallback.

    Parameters
    ----------
    client : OpenAI
        Initialised API client.
    model : str
        Model identifier.
    batch_entries : List[SubtitleEntry]
        Entries to translate.
    prev_entries_orig, prev_entries_trans : Optional[List[SubtitleEntry]]
        Previous batch entries for contextual continuity.
    movie_name, movie_bg, glossary : str
        Contextual metadata injected into system prompt.
    max_retries : int
        Number of retries per batch before triggering single-line isolation.
    full_script_text, use_god_mode : str / bool
        God-mode context injection.
    temperature : float
        Sampling temperature (overridden by the model when ``use_thinking`` is True).
    use_thinking : bool
        Enable DeepSeek reasoning mode.
    reasoning_effort : str
        Reasoning effort level (``"high"`` or ``"max"``).

    Returns
    -------
    Tuple[List[SubtitleEntry], List[int]]
        ``(translated_entries, missing_ids)`` — all entries are guaranteed to
        be present (missing ones use original text as fallback).
    """
    system_prompt = build_system_prompt(
        movie_name, movie_bg, glossary, full_script_text, use_god_mode
    )
    user_prompt = build_user_prompt(
        batch_entries, prev_entries_orig, prev_entries_trans
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        is_last_attempt = attempt == max_retries

        try:
            api_kwargs: dict = {
                "model": model,
                "max_tokens": 8192,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            if model in JSON_SUPPORTED_MODELS and not use_thinking:
                api_kwargs["response_format"] = {"type": "json_object"}

            if use_thinking:
                api_kwargs["reasoning_effort"] = reasoning_effort
                api_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                api_kwargs["temperature"] = temperature
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            resp = client.chat.completions.create(**api_kwargs)
            content = resp.choices[0].message.content or ""

            return extract_and_validate_json(batch_entries, content, strict=True)

        except (
            APIConnectionError,
            APIStatusError,
            RateLimitError,
            ValueError,
        ) as err:
            last_error = err
            if not is_last_attempt:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
            else:
                if not isinstance(err, ValueError):
                    raise RuntimeError(
                        f"网络或API接口异常，批次(序号 {batch_entries[0].index} 起)彻底失败：{last_error}"
                    )

    # ── Single-line isolation fallback ──────────────────────────────────────
    # Triggered only after max_retries consecutive failures (typically due to
    # the model persistently merging / skipping short entries).
    # NOTE: _toast cannot be called here — it runs in a worker thread.
    # The caller must handle this via the `isolation_used` return flag.

    isolated_results: List[SubtitleEntry] = []
    isolated_missing: List[int] = []

    def _translate_single(entry: SubtitleEntry) -> Tuple[SubtitleEntry, List[int]]:
        single_kwargs: dict = {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": build_user_prompt([entry], None, None),
                },
            ],
        }
        if use_thinking:
            single_kwargs["reasoning_effort"] = reasoning_effort
            single_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            single_kwargs["temperature"] = temperature
            single_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            if model in JSON_SUPPORTED_MODELS:
                single_kwargs["response_format"] = {"type": "json_object"}

        try:
            r = client.chat.completions.create(**single_kwargs)
            parsed, missing = extract_and_validate_json(
                [entry], r.choices[0].message.content or "", strict=False
            )
            return parsed[0], missing
        except Exception:
            return (
                SubtitleEntry(
                    entry.index, entry.timeline, entry.text, is_fallback=True
                ),
                [entry.index],
            )

    max_workers = min(10, len(batch_entries))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_translate_single, e) for e in batch_entries
        ]
        for future in concurrent.futures.as_completed(futures):
            res_entry, res_missing = future.result()
            isolated_results.append(res_entry)
            isolated_missing.extend(res_missing)

    isolated_results.sort(key=lambda x: x.index)
    # Mark results as isolation fallback so the caller (main thread) can safely
    # display a toast instead of us calling _toast from a worker thread.
    for e in isolated_results:
        setattr(e, "_isolation_fallback", True)
    return isolated_results, isolated_missing

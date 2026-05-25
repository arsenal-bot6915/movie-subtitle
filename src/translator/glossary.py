"""Automatic terminology extraction from the first N subtitle entries.

The :func:`auto_extract_glossary` function is called before translation begins.
It asks the model to read the first *N* entries, identify recurring proper nouns
(character names, locations, organisations, technical terms), and return a
``name: translation`` table that can be pasted into the glossary field.
"""

import json
import re
from typing import List

from openai import OpenAI

from ..core.models import SubtitleEntry
from .prompts import build_system_prompt

GLOSSARY_EXTRACT_PROMPT = """请仔细阅读以下英文字幕，识别并提取：

1. **人物名称**（人名、昵称、尊称/绰号）
2. **地名 / 组织名**（城市、国家、机构、建筑物等专有名词）
3. **专有技术名词 / 俚语**（若存在）

请按以下格式返回，每行一个，中文译名放在冒号后面。不要添加任何解释：

示例格式：
John: 约翰
NATO: 北约
FBI: 联邦调查局
The White House: 白宫
...

英文字幕：
{subtitle_block}
""".strip()


def auto_extract_glossary(
    client: OpenAI,
    model: str,
    entries: List[SubtitleEntry],
    sample_size: int = 150,
) -> str:
    """
    Return a ``\\n``-separated glossary string by reading the first *sample_size*
    subtitle entries.

    Parameters
    ----------
    client : OpenAI
        Initialised API client.
    model : str
        Model ID to use.
    entries : List[SubtitleEntry]
        All parsed subtitle entries.
    sample_size : int
        Number of entries to feed into the extraction prompt (default 150).

    Returns
    -------
    str
        The glossary text suitable for pasting into the UI field, or an
        error message if extraction failed.
    """
    sample = entries[:sample_size]
    lines: List[str] = []
    for e in sample:
        lines.append(f"[{e.index}] {e.text}")
    subtitle_block = "\n".join(lines)

    system_prompt = build_system_prompt()
    user_prompt = GLOSSARY_EXTRACT_PROMPT.format(subtitle_block=subtitle_block)

    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""

        # Strip any surrounding markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)

        # Attempt to parse as JSON first
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return "\n".join(f"{k}: {v}" for k, v in data.items())
        except json.JSONDecodeError:
            pass

        # Fall back to plain text: strip numbered prefixes like "1. John: 约翰"
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", raw, flags=re.MULTILINE)
        return cleaned.strip()

    except Exception as exc:
        return f"术语提取失败：{exc}"

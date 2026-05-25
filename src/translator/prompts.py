"""System and user prompt builders for the DeepSeek translation API."""

from typing import List, Optional

from ..core.models import SubtitleEntry

SYSTEM_PROMPT_TEMPLATE = """你是一位专业的影视字幕翻译专家，负责将英文对白精准、地道地翻译为中文。

**核心原则**
1. **角色语言风格**：根据角色身份、性格、情绪选用恰当的语气和用词。保留原文的语气词、俚语、口头禅。
2. **简洁性**：字幕必须在屏幕上一眼可读，每行不超过{MAX_LINE_CHARS}个字符，两行不超过{MAX_LINE_CHARS_2}个字符。
3. **口语化**：日常对话尽量用白话，忌文绉绉。
4. **术语一致性**：同一角色/名词在整部影片中保持统一翻译。

**特殊格式保留规则**
- 括号内描述（如 "(laughs)"、"[sighs]"）→ 翻译为中文描述，保留括号格式。
- 含有 ``_`` 下划线、``*`` 星号的强调词 → 保留原文样式，不翻译。
- 音乐类（如 "♫ Intro ♫"）→ 只翻译可见文字，不翻译符号。

**专有名词对照表（请严格遵守）**
{glossary}

**影片背景**
{movie_bg}

**影片名称**
{movie_name}
""".strip()


def build_system_prompt(
    movie_name: str = "",
    movie_bg: str = "",
    glossary: str = "",
    full_script_text: str = "",
    use_god_mode: bool = False,
) -> str:
    """
    Build the system prompt, optionally appending the full script for god mode.

    Parameters
    ----------
    movie_name, movie_bg, glossary : str
        Contextual metadata filled in from the UI.
    full_script_text : str
        All subtitle text concatenated (god mode context injection).
    use_god_mode : bool
        Whether to include full script context.
    """
    system = SYSTEM_PROMPT_TEMPLATE.format(
        movie_name=movie_name or "未提供",
        movie_bg=movie_bg or "未提供",
        glossary=glossary or "未提供",
        MAX_LINE_CHARS=20,
        MAX_LINE_CHARS_2=40,
    )

    if use_god_mode and full_script_text:
        system += (
            f"\n\n**【全量剧本上下文 — 请在翻译时全局参考】**\n"
            f"以下为该影片全部字幕（按时间顺序），请结合全局语境确保翻译连贯性：\n"
            f"---\n{full_script_text}\n---"
        )

    return system


_USER_PROMPT_TEMPLATE = """请将以下字幕片段翻译为中文，严格按照 JSON 格式返回（键为字幕序号，值为翻译结果）：

**JSON 格式要求**
- 键：字幕的原始序号（整数）
- 值：翻译后的中文字幕（字符串）
- 每条字幕请**独立翻译**，不要合并或省略任何一条
- 保持原有换行结构不变
- 仅返回 JSON，不要添加任何解释、注释或额外文字

**参考：前一批次译文（语境连贯性）**
{prev_context}

**待翻译字幕**
{subtitle_block}
""".strip()


def build_user_prompt(
    batch_entries: List[SubtitleEntry],
    prev_entries_orig: Optional[List[SubtitleEntry]] = None,
    prev_entries_trans: Optional[List[SubtitleEntry]] = None,
) -> str:
    """
    Build the user-facing translation prompt.

    Parameters
    ----------
    batch_entries : List[SubtitleEntry]
        The current batch of entries to translate.
    prev_entries_orig, prev_entries_trans : Optional[List[SubtitleEntry]]
        Previous batch entries (original and translated) to provide
        contextual continuity.
    """
    # ── Context from the previous batch (up to 40 entries) ────────────────────
    if prev_entries_orig and prev_entries_trans:
        ctx_lines: List[str] = []
        for orig, trans in zip(
            prev_entries_orig[-40:], prev_entries_trans[-40:]
        ):
            ctx_lines.append(f"[序号 {orig.index}] 英文: {orig.text}")
            ctx_lines.append(f"[序号 {orig.index}] 中文: {trans.text}")
        prev_context = "\n".join(ctx_lines)
    else:
        prev_context = "（无前一批次参考）"

    # ── Subtitle block ────────────────────────────────────────────────────────
    lines: List[str] = []
    for e in batch_entries:
        timeline_note = f"[{e.timeline}]"
        lines.append(f"[序号 {e.index}] {timeline_note} {e.text}")
    subtitle_block = "\n".join(lines)

    return _USER_PROMPT_TEMPLATE.format(
        prev_context=prev_context,
        subtitle_block=subtitle_block,
    )

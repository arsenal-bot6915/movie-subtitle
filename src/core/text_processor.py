"""Text preprocessing and post-processing utilities for the translation pipeline."""

import re
from typing import Dict, List

# ── Chinese detection ──────────────────────────────────────────────────────────

# Matches any CJK Unified Ideographs / extension block codepoint
RE_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]")

# Matches ASCII letters only (excludes digits, punctuation, ellipsis)
RE_ASCII_WORD = re.compile(r"[a-zA-Z]")


def contains_chinese(text: str) -> bool:
    """Return True if *text* contains at least one Chinese character."""
    return bool(RE_CJK.search(text))


def has_ascii_word(text: str) -> bool:
    """Return True if *text* contains at least one ASCII letter."""
    return bool(RE_ASCII_WORD.search(text))


# ── Smart line wrapping ───────────────────────────────────────────────────────

MAX_LINE_CHARS = 20


def smart_wrap_chinese(text: str) -> str:
    """
    Re-wrap long Chinese lines so that no line exceeds :data:`MAX_LINE_CHARS`
    characters. Tries to break at natural punctuation or mid-word.
    """
    lines = text.split("\n")
    result: List[str] = []
    for line in lines:
        if len(line) <= MAX_LINE_CHARS or not contains_chinese(line):
            result.append(line)
        else:
            result.append(_break_chinese_line(line))
    return "\n".join(result)


def _break_chinese_line(line: str) -> str:
    """
    Iteratively break *line* into segments of at most MAX_LINE_CHARS Chinese
    characters, preferring natural punctuation breaks.
    """
    # Priority: full-width comma/period → half-width comma → mid-word → hard cut
    sep_order = ("，", "。", "！", "？", "；", ",", ".", "!", "?", ";")
    threshold = MAX_LINE_CHARS * 0.4  # must find separator in first 40 %

    parts: List[str] = []
    while len(line) > MAX_LINE_CHARS:
        # Search for the last separator within the window
        best_pos = -1
        for sep in sep_order:
            pos = line.rfind(sep, 0, MAX_LINE_CHARS)
            if pos > int(threshold):
                best_pos = max(best_pos, pos)
                break
            # Also check from threshold onward (rfind searches backwards)
            pos2 = line.rfind(sep, int(threshold), MAX_LINE_CHARS)
            if pos2 != -1:
                best_pos = max(best_pos, pos2)

        if best_pos >= 0:
            # Break right after the separator
            parts.append(line[: best_pos + 1].strip())
            line = line[best_pos + 1 :].strip()
        else:
            # No separator in range — hard split
            parts.append(line[:MAX_LINE_CHARS].strip())
            line = line[MAX_LINE_CHARS:].strip()

    if line:
        parts.append(line)
    return "\n".join(parts)


# ── ASS tag extraction (for non-ASS pipelines) ─────────────────────────────────

ASS_TAG_RE = re.compile(r"\{[^}]+\}")


def mask_ass_tags(text: str) -> str:
    """Replace ASS override tags with ``__TAG_N__`` placeholders."""
    idx = 0
    def _repl(m: re.Match) -> str:
        nonlocal idx
        placeholder = f"__TAG_{idx}__"
        idx += 1
        return placeholder
    return ASS_TAG_RE.sub(_repl, text)


def unmask_ass_tags(text: str, tags_map: Dict[int, str]) -> str:
    """
    Restore ASS override tags into *text* lines using *tags_map*.

    Parameters
    ----------
    text : str
        Translated text (may contain ``__TAG_N__`` placeholders).
    tags_map : Dict[int, str]
        Maps line index → ASS tag string to insert at line start.
    """
    if not tags_map:
        return text
    lines = text.split("\n")
    result: List[str] = []
    for i, line in enumerate(lines):
        tag = tags_map.get(i, "")
        result.append(f"{tag}{line}" if tag else line)
    return "\n".join(result)

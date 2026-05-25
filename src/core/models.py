"""Domain models used throughout the translation pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubtitleEntry:
    """
    Represents a single subtitle block.

    Attributes
    ----------
    index : int
        1-based subtitle sequence number (matches SRT ``index`` field).
    timeline : str
        Raw SRT/ASS timestamp line, e.g. ``00:01:22,500 --> 00:01:24,000``.
    text : str
        Subtitle text content. May be the original or translated.
    is_fallback : bool
        True when the entry fell back to the original text because the AI
        failed to translate it after all retries.
    raw_prefix : str
        Any prefix characters that appeared before the first subtitle line
        in ASS files (e.g. ``{\\an8}``). Passed through unchanged.
    tags_map : Optional[Dict[int, str]]
        ASS override tags stripped from each line, keyed by line index so
        they can be re-applied after translation.
    """

    index: int
    timeline: str
    text: str
    is_fallback: bool = False
    raw_prefix: str = ""
    tags_map: Optional[Dict[int, str]] = field(default=None)

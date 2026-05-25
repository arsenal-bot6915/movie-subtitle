"""SRT file parsing and serialisation."""

import re
from typing import List, Tuple

from ..core.models import SubtitleEntry

# ── Parsing ────────────────────────────────────────────────────────────────────

_SRT_BLOCK = re.compile(r"\r?\n\r?\n|\r\r")
_SRT_LINE = re.compile(
    r"(?P<index>\d+)\s*\n(?P<timeline>\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n(?P<text>.*)",
    re.DOTALL,
)


def parse_srt(content: str) -> List[SubtitleEntry]:
    """
    Parse raw SRT text into a list of :class:`SubtitleEntry`.

    Parameters
    ----------
    content : str
        Decoded file content.

    Returns
    -------
    List[SubtitleEntry]
        Entries in document order.
    """
    entries: List[SubtitleEntry] = []
    for block in _SRT_BLOCK.split(content.strip()):
        m = _SRT_LINE.match(block)
        if not m:
            continue
        entries.append(
            SubtitleEntry(
                index=int(m.group("index")),
                timeline=m.group("timeline").strip(),
                text=m.group("text").strip(),
            )
        )
    return entries


# ── Serialising ────────────────────────────────────────────────────────────────

def entries_to_srt(entries: List[SubtitleEntry]) -> str:
    """Serialize a list of :class:`SubtitleEntry` back to SRT text."""
    lines: List[str] = []
    for e in entries:
        # Normalise timeline separators: always use comma (SRT convention)
        timeline = e.timeline.replace(".", ",")
        # Strip ASS tags from display for plain SRT (ASS files keep them)
        display_text = strip_ass_tags(e.text)
        lines.append(f"{e.index}\n{timeline}\n{display_text}\n")
    return "\n".join(lines)


# ── Encoding detection ─────────────────────────────────────────────────────────

def decode_uploaded_file(uploaded_bytes: bytes) -> str:
    """
    Attempt to decode bytes using a sequence of common Chinese/Unicode
    encodings and return the first successful result.
    """
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return uploaded_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文件编码无法识别，请使用 UTF-8 或 GB 编码的 SRT 文件。")


# ── ASS tag helpers (shared between parsers) ──────────────────────────────────

ASS_TAG_RE = re.compile(r"\{[^}]+\}")


def strip_ass_tags(text: str) -> str:
    """Remove ASS override tags from a line."""
    return ASS_TAG_RE.sub("", text)

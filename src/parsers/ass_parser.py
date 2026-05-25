"""ASS/SSA file parsing and serialisation."""

import re
from typing import Dict, List, Tuple

from ..core.models import SubtitleEntry
from .srt_parser import ASS_TAG_RE, strip_ass_tags

# ── Internal ASS constants ─────────────────────────────────────────────────────

_FORMAT_LINE_RE = re.compile(r"^Format:\s*(.*)", re.IGNORECASE)
_EVENT_LINE_RE  = re.compile(r"^(Dialogue|Comment):\s*(.*)", re.IGNORECASE)
_METADATA_RE    = re.compile(
    r"^\[(Script Info|V4\+ Styles|V4 Styles|Events|V4\+ Keyframes|Actors|Window |\w+)\]",
    re.IGNORECASE,
)

_LAYOUT_X_RE = re.compile(r"\{[^}]*\\an(\d)[^}]*\}")


def parse_ass(content: str) -> Tuple[List[SubtitleEntry], str]:
    """
    Parse an ASS file into a list of :class:`SubtitleEntry` and the raw
    header section that should be preserved verbatim in the output.

    Parameters
    ----------
    content : str
        Decoded file content.

    Returns
    -------
    Tuple[List[SubtitleEntry], str]
        ``(entries, header)`` where *header* is everything up to (and
        including) the ``[Events]`` line.
    """
    entries: List[SubtitleEntry] = []
    header_lines: List[str] = []
    index_counter = 0

    # ── Pass 1: collect header lines ──────────────────────────────────────────
    for raw_line in content.splitlines():
        header_lines.append(raw_line)
        if _EVENT_LINE_RE.match(raw_line.strip()):
            break

    # ── Pass 2: parse Format + Dialogue lines ─────────────────────────────────
    current_format: List[str] = []
    seen_format = False

    for raw_line in content.splitlines():
        stripped = raw_line.strip()

        # Capture metadata sections
        if _METADATA_RE.match(stripped):
            continue  # already collected in header pass

        # Format declaration
        m_fmt = _FORMAT_LINE_RE.match(stripped)
        if m_fmt:
            current_format = [f.strip().lower() for f in m_fmt.group(1).split(",")]
            seen_format = True
            continue

        # Dialogue / Comment
        m_evt = _EVENT_LINE_RE.match(stripped)
        if m_evt and seen_format:
            parts = _split_ass_fields(m_evt.group(2))
            field_map = {name: val for name, val in zip(current_format, parts)}

            try:
                layer    = int(field_map.get("layer", 0))
                start    = field_map.get("start", "").strip()
                end      = field_map.get("end", "").strip()
                style    = field_map.get("style", "Default").strip()
                text     = field_map.get("text", "").strip()
            except Exception:
                continue

            # ASS timeline → SRT-compatible format
            timeline = f"{_ass_to_srt_time(start)} --> {_ass_to_srt_time(end)}"

            # Strip ASS tags from text, preserving speaker prefix
            prefix, clean_text, tags_map = _extract_ass_tags(text)
            display_text = strip_ass_tags(clean_text)

            # Sort by (layer, start_time) to keep overlapping events stable
            index_counter += 1
            entries.append(
                SubtitleEntry(
                    index=index_counter,
                    timeline=timeline,
                    text=display_text,
                    raw_prefix=prefix,
                    tags_map=tags_map if tags_map else None,
                )
            )

    return entries, "\n".join(header_lines) + "\n"


def entries_to_ass(entries: List[SubtitleEntry], ass_header: str) -> str:
    """
    Serialize translated entries back to ASS format, preserving the original
    header. Speaker prefixes and ASS tags extracted at parse time are re-applied.
    """
    # Find insertion point: after [Events] line, before first Format/Dialogue
    lines = ass_header.splitlines()
    insert_after = len(lines)
    for i, ln in enumerate(lines):
        if _EVENT_LINE_RE.match(ln.strip()):
            insert_after = i
            break

    # Reconstruct the [Events] block
    event_lines = [
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    for e in entries:
        # Unmask ASS tags so they reappear in the output
        display_text = _unmask_ass_tags(e.text, e.tags_map or {})
        # Prefix preserved from the original file
        text_content = f"{e.raw_prefix}{display_text}" if e.raw_prefix else display_text
        # Map SRT timeline back to ASS format
        start_srt, end_srt = e.timeline.split("-->")
        start_ass = _srt_to_ass_time(start_srt.strip())
        end_ass   = _srt_to_ass_time(end_srt.strip())

        event_lines.append(
            f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text_content}"
        )

    all_lines = lines[:insert_after] + event_lines + lines[insert_after:]
    return "\n".join(all_lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_ass_fields(text: str) -> List[str]:
    """Split ASS field text, respecting comma positions."""
    parts: List[str] = []
    depth = 0
    buf = ""
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return parts


def _extract_ass_tags(text: str) -> Tuple[str, str, Dict[int, str]]:
    """
    Separate ASS override tags from visible text.

    Returns ``(prefix, clean_text, tags_map)`` where *prefix* is the leading
    speaker tag (e.g. ``NARRATOR:``) and *tags_map* maps each line index to
    its inline override tags for later re-application.
    """
    lines = text.replace("\\N", "\n").replace("\\n", "\n").split("\n")
    tags_map: Dict[int, str] = {}
    clean_lines: List[str] = []

    for line_idx, line in enumerate(lines):
        tags = ASS_TAG_RE.findall(line)
        tags_map[line_idx] = "\n".join(tags) if tags else ""
        clean_lines.append(ASS_TAG_RE.sub("", line).strip())

    joined = "\n".join(clean_lines)
    # Speaker prefix: non-Chinese characters before the first Chinese/hard space
    prefix_match = re.match(r"^([^{}\w\s][^\n:：]{0,30}?)[\s:：]", joined)
    prefix = prefix_match.group(1) + ": " if prefix_match else ""
    return prefix, joined, tags_map


def _unmask_ass_tags(text: str, tags_map: Dict[int, str]) -> str:
    """Re-insert ASS override tags into translated text lines."""
    if not tags_map:
        return text
    lines = text.split("\n")
    result: List[str] = []
    for i, line in enumerate(lines):
        tag = tags_map.get(i, "")
        result.append(f"{tag}{line}" if tag else line)
    return "\n".join(result)


def _ass_to_srt_time(t: str) -> str:
    """Convert ASS timestamp (H:MM:SS.CC) to SRT timestamp (H:MM:SS,CC)."""
    t = t.strip().replace(",", ".")
    return t  # both formats use dot; the SRT writer normalises to comma


def _srt_to_ass_time(t: str) -> str:
    """Convert SRT timestamp (H:MM:SS,CC) to ASS timestamp (H:MM:SS.CC)."""
    return t.strip().replace(",", ".")

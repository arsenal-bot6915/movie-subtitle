"""Scene-aware subtitle chunking algorithm.

The chunker takes a flat list of :class:`~core.models.SubtitleEntry` and
splits it into batches that are safe to send to the translation API.
Two constraints are applied simultaneously:

1. **Hard size limit** — no batch may contain more than *max_batch_size*
   entries, regardless of timing.
2. **Scene boundary** — whenever two consecutive entries are separated by
   more than *gap_threshold_ms* milliseconds, a new batch starts immediately,
   ensuring that a scene change never lands in the middle of a request.
"""

import re
from typing import List

from ..core.models import SubtitleEntry

# Matches both SRT (HH:MM:SS,mmm --> HH:MM:SS,mmm) and ASS timestamps.
# Extracts the first full HH:MM:SS,mmm timestamp found in the string (usually the start time).
# To parse the end time, use parse_timestamp_ms(get_end_time(timeline)).
_SRT_TIME_RE = re.compile(
    r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)[,\.](?P<ms>\d{1,3})"
)


def smart_chunk_entries(
    entries: List[SubtitleEntry],
    max_batch_size: int = 60,
    gap_threshold_ms: int = 15_000,
) -> List[List[SubtitleEntry]]:
    """
    Split *entries* into batches respecting scene boundaries.

    Parameters
    ----------
    entries : List[SubtitleEntry]
        Subtitle entries in chronological order.
    max_batch_size : int
        Maximum number of entries per batch.
    gap_threshold_ms : int
        Milliseconds of silence between two entries that triggers an
        immediate scene cut. Defaults to 15 000 ms (15 s).

    Returns
    -------
    List[List[SubtitleEntry]]
        List of batches; each batch is itself a list of entries.
    """
    if not entries:
        return []

    batches: List[List[SubtitleEntry]] = []
    current: List[SubtitleEntry] = []

    for entry in entries:
        should_cut = False

        if current:
            prev = current[-1]
            gap_ms = _timestamp_gap_ms(prev.timeline, entry.timeline)

            # Force cut on a scene break
            if gap_ms > gap_threshold_ms:
                should_cut = True
            # Force cut when hard size limit would be exceeded
            elif len(current) >= max_batch_size:
                should_cut = True

        if should_cut:
            batches.append(current)
            current = []

        current.append(entry)

    if current:
        batches.append(current)

    return batches


# ── Timestamp helpers ──────────────────────────────────────────────────────────

_MS_PER_SEC = 1_000
_MS_PER_MIN = 60 * _MS_PER_SEC
_MS_PER_HOUR = 60 * _MS_PER_MIN


def parse_timestamp_ms(timeline: str) -> int:
    """
    Convert a timeline string (e.g. ``00:01:22,500``) to total milliseconds.
    Handles both comma and dot decimal separators.
    """
    m = _SRT_TIME_RE.search(timeline)
    if not m:
        return 0
    return (
        int(m.group("h")) * _MS_PER_HOUR
        + int(m.group("m")) * _MS_PER_MIN
        + int(m.group("s")) * _MS_PER_SEC
        + int(m.group("ms")[:3].ljust(3, "0"))
    )


def _timestamp_gap_ms(timeline_a: str, timeline_b: str) -> int:
    """Return the time difference in ms from timeline_a to timeline_b."""
    a_ms = parse_timestamp_ms(timeline_a)
    b_ms = parse_timestamp_ms(timeline_b)
    return max(0, b_ms - a_ms)


def compress_full_script(entries: List[SubtitleEntry]) -> str:
    """
    Concatenate all subtitle text with newlines.

    This string is injected as the "full script" context for the god-mode
    system prompt so the model has global narrative awareness.
    """
    return "\n".join(e.text for e in entries)

"""Local cache for translation progress.

Cache files live under :data:`~core.constants.CACHE_DIR` and are named by an
MD5 hash derived from the file content and the active translation settings.
This lets the pipeline resume after interruption without re-sending already-
translated batches to the API.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.constants import CACHE_DIR
from ..core.models import SubtitleEntry


# ── Path resolution ─────────────────────────────────────────────────────────────

def get_cache_file_path(
    file_bytes: bytes,
    model: str,
    movie_bg: str,
    glossary: str,
    batch_size: int,
    gap_threshold_ms: int,
    use_god_mode: bool,
    temperature: float,
    use_thinking: bool,
    reasoning_effort: str,
    skip_chinese: bool,
) -> Path:
    """
    Derive a deterministic cache file path from *file_bytes* and all runtime
    settings that affect translation output.
    """
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw = (
        f"{hashlib.md5(file_bytes).hexdigest()}"
        f"|model={model}"
        f"|bg={movie_bg}"
        f"|glossary={glossary}"
        f"|batch={batch_size}"
        f"|gap={gap_threshold_ms}"
        f"|god={use_god_mode}"
        f"|temp={temperature}"
        f"|thinking={use_thinking}"
        f"|reasoning={reasoning_effort}"
        f"|skipch={skip_chinese}"
    )
    cache_hash = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{cache_hash}.json"


# ── Serialisation helpers ───────────────────────────────────────────────────────

def _entry_to_dict(e: SubtitleEntry) -> Dict[str, Any]:
    return {
        "index": e.index,
        "timeline": e.timeline,
        "text": e.text,
        "is_fallback": e.is_fallback,
        "raw_prefix": getattr(e, "raw_prefix", ""),
        "tags_map": getattr(e, "tags_map", None),
    }


def _dict_to_entry(d: Dict[str, Any]) -> SubtitleEntry:
    raw_tags = d.get("tags_map")
    tags_map = None
    if raw_tags is not None:
        # JSON keys are always strings; restore int keys
        tags_map = {int(k): v for k, v in raw_tags.items()}
    return SubtitleEntry(
        index=d["index"],
        timeline=d["timeline"],
        text=d["text"],
        is_fallback=d.get("is_fallback", False),
        raw_prefix=d.get("raw_prefix", ""),
        tags_map=tags_map,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def save_progress_to_local(cache_path: Path, translated_dict: Dict[int, List[SubtitleEntry]]) -> None:
    """
    Write the current translation state to *cache_path*.

    The file is written atomically (write-to-temp + rename) to avoid corruption
    if the process is killed mid-write.
    """
    payload = {
        str(idx): [_entry_to_dict(e) for e in batch]
        for idx, batch in translated_dict.items()
    }
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cache_path)


def load_progress_from_local(cache_path: Path) -> Dict[int, List[SubtitleEntry]]:
    """
    Load a previously saved translation state from *cache_path*.

    Returns an empty dict if the file does not exist or is unreadable.
    """
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: Dict[int, List[SubtitleEntry]] = {}
    for idx_str, batch_list in data.items():
        idx = int(idx_str)
        result[idx] = [_dict_to_entry(d) for d in batch_list]
    return result

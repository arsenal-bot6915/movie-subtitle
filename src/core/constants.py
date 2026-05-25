"""Shared configuration constants across the translation pipeline."""

from pathlib import Path

# ── Batch / chunking defaults ────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE: int = 60
DEFAULT_GAP_THRESHOLD_MS: int = 15_000  # 15 seconds → scene boundary

# ── API / retry defaults ───────────────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_WAIT_SECONDS: int = 5
API_TIMEOUT_SECONDS: int = 120

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path(__file__).parent.parent.parent / ".srt_cache"

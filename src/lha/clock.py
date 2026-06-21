"""Single source of timezone-aware 'now' so timestamps are consistent."""

from __future__ import annotations

from datetime import datetime, timezone


def now() -> datetime:
    """Timezone-aware current time (UTC)."""
    return datetime.now(timezone.utc)

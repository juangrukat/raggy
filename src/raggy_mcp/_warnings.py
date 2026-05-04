"""Warning filters for noisy upstream imports."""

from __future__ import annotations

import warnings


def filter_upstream_warnings() -> None:
    """Silence known dependency warnings that users cannot act on here."""
    try:
        from authlib.deprecate import AuthlibDeprecationWarning
    except Exception:
        AuthlibDeprecationWarning = DeprecationWarning

    warnings.filterwarnings(
        "ignore",
        category=AuthlibDeprecationWarning,
    )

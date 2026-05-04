"""
Versioned tool-response envelope.

Every priority tool returns a JSON object with this shape:

    {
      "contract": {
        "contract_version": "1.0",
        "toolset_version": "<server version>",
        "profile": "minimal|canonical|full"
      },
      "data": { <tool-specific payload> },
      "observability": {
        "duration_ms": int,
        "warnings": [str, ...],
        "stats": { <free-form> }
      }
    }

Failures keep the same outer shape but replace ``data`` with ``error``:

    {
      "contract": { ... },
      "error": {
        "code": "invalid_filter",
        "message": "...",
        "retryable": false
      },
      "observability": { ... }
    }

This module supplies helpers (``success``, ``failure``, ``Stopwatch``) and a
context manager (``envelope_context``) that times the operation and tags the
response with the active profile and toolset version.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

CONTRACT_VERSION = "1.0"
TOOLSET_VERSION = "0.8.0"


@dataclass
class _Accumulator:
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    started: float = 0.0


def _contract(profile: str) -> dict[str, str]:
    return {
        "contract_version": CONTRACT_VERSION,
        "toolset_version": TOOLSET_VERSION,
        "profile": profile,
    }


def success(
    data: dict[str, Any],
    *,
    profile: str = "canonical",
    duration_ms: int = 0,
    warnings: list[str] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract": _contract(profile),
        "data": data,
        "observability": {
            "duration_ms": duration_ms,
            "warnings": warnings or [],
            "stats": stats or {},
        },
    }


def failure(
    code: str,
    message: str,
    *,
    profile: str = "canonical",
    retryable: bool = False,
    duration_ms: int = 0,
    warnings: list[str] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract": _contract(profile),
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
        "observability": {
            "duration_ms": duration_ms,
            "warnings": warnings or [],
            "stats": stats or {},
        },
    }


@contextmanager
def envelope_context(profile: str) -> Iterator[_Accumulator]:
    """
    Use as:
        with envelope_context(profile) as acc:
            ...
            acc.warnings.append("used pypdf fallback")
            acc.stats["pages"] = 12
            return success_from(acc, data={...})

    The accumulator captures ``warnings`` and ``stats`` so the tool body can
    annotate the response without threading them through every return.
    """
    acc = _Accumulator()
    acc.started = time.perf_counter()
    try:
        yield acc
    finally:
        # Caller is responsible for using the accumulator; this just exposes timing.
        pass


def elapsed_ms(acc: _Accumulator) -> int:
    return int((time.perf_counter() - acc.started) * 1000)


def success_from(
    acc: _Accumulator, data: dict[str, Any], *, profile: str
) -> dict[str, Any]:
    """Build a success envelope from an accumulator + data dict."""
    return success(
        data,
        profile=profile,
        duration_ms=elapsed_ms(acc),
        warnings=acc.warnings,
        stats=acc.stats,
    )


def failure_from(
    acc: _Accumulator,
    code: str,
    message: str,
    *,
    profile: str,
    retryable: bool = False,
) -> dict[str, Any]:
    """Build a failure envelope from an accumulator + error info."""
    return failure(
        code,
        message,
        profile=profile,
        retryable=retryable,
        duration_ms=elapsed_ms(acc),
        warnings=acc.warnings,
        stats=acc.stats,
    )

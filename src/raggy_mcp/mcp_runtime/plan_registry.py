"""
Report/apply plan registry.

Some tools support two modes:

* ``mode="report"`` — describe what *would* happen and return a ``plan_id``;
  no writes are performed.
* ``mode="apply"`` — execute the previously-reported plan; requires the
  ``plan_id`` returned by the matching report call.

The registry is in-memory and process-local. Plans are namespaced by tool name
and expire after a configurable TTL so a stale plan can't be re-applied long
after the underlying state has drifted.

This module is deliberately tiny — the actual report/apply logic lives in the
tools themselves. The registry only handles plan IDs, payloads, expiry, and
single-use semantics.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TTL_SECONDS = 600  # 10 minutes


@dataclass
class Plan:
    plan_id: str
    tool: str
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    consumed: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


class PlanRegistry:
    """In-memory registry of pending report/apply plans."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._plans: dict[str, Plan] = {}
        self._lock = asyncio.Lock()

    async def create(self, tool: str, payload: dict[str, Any], **extras: Any) -> Plan:
        async with self._lock:
            self._evict_expired()
            now = time.time()
            plan_id = f"plan_{secrets.token_urlsafe(12)}"
            plan = Plan(
                plan_id=plan_id,
                tool=tool,
                payload=payload,
                created_at=now,
                expires_at=now + self._ttl,
                extras=extras,
            )
            self._plans[plan_id] = plan
            return plan

    async def consume(self, plan_id: str, *, expected_tool: str) -> Plan:
        """
        Return and mark a plan consumed. Raises ValueError on missing, expired,
        already-consumed, or wrong-tool plans.
        """
        async with self._lock:
            self._evict_expired()
            plan = self._plans.get(plan_id)
            if plan is None:
                raise ValueError(
                    f"Plan '{plan_id}' not found. Either it never existed, expired, "
                    f"or was already applied. Re-run with mode='report' to get a fresh plan_id."
                )
            if plan.consumed:
                raise ValueError(f"Plan '{plan_id}' was already applied. Run mode='report' to get a new one.")
            if plan.tool != expected_tool:
                raise ValueError(
                    f"Plan '{plan_id}' was created for tool '{plan.tool}', not '{expected_tool}'."
                )
            plan.consumed = True
            return plan

    def _evict_expired(self) -> None:
        now = time.time()
        stale = [pid for pid, p in self._plans.items() if p.expires_at <= now]
        for pid in stale:
            self._plans.pop(pid, None)

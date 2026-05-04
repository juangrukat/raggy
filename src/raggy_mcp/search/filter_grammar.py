"""
High-level filter grammar that compiles to qdrant_client `models.Filter`.

Input shape:
{
  "must":     [{"field": "extension",   "op": "==",  "value": "pdf"}, ...],
  "should":   [{"field": "tags",        "op": "any", "value": ["work","urgent"]}, ...],
  "must_not": [{"field": "is_hidden",   "op": "==",  "value": true}, ...]
}

Operators:
  ==, !=, >, >=, <, <=, any, except

Field names are auto-prefixed with "metadata." if not already.
For strict array exclusion semantics, prefer must_not + ==/any over except.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from qdrant_client import models


_RANGE_OPS = {">", ">=", "<", "<="}
_EQ_OPS = {"==", "!="}


def _normalize_field(name: str) -> str:
    if name.startswith("metadata.") or "." in name:
        return name
    return f"metadata.{name}"


def _build_match(field: str, op: str, value: Any) -> models.FieldCondition:
    field = _normalize_field(field)
    if op == "any":
        if not isinstance(value, list):
            value = [value]
        return models.FieldCondition(key=field, match=models.MatchAny(any=value))
    if op == "except":
        if not isinstance(value, list):
            value = [value]
        return models.FieldCondition(key=field, match=models.MatchExcept(**{"except": value}))
    if op in _EQ_OPS:
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))
    if op in _RANGE_OPS:
        kwargs: dict[str, Any] = {}
        if op == ">":
            kwargs["gt"] = value
        elif op == ">=":
            kwargs["gte"] = value
        elif op == "<":
            kwargs["lt"] = value
        elif op == "<=":
            kwargs["lte"] = value
        # Use DatetimeRange for ISO strings, otherwise plain Range
        if isinstance(value, str) and len(value) >= 10 and value[4] == "-":
            return models.FieldCondition(key=field, range=models.DatetimeRange(**kwargs))
        return models.FieldCondition(key=field, range=models.Range(**kwargs))
    raise ValueError(f"Unsupported op: {op}")


def _build_clause(clauses: list[dict]) -> list[models.Condition]:
    out: list[models.Condition] = []
    for c in clauses:
        field = c["field"]
        op = c["op"]
        value = c.get("value")
        if op == "!=":
            # != is encoded as a positive condition the caller wraps in must_not — but here
            # we inline-handle it as a Filter(must_not=[==]) so it works inside any clause.
            inner = _build_match(field, "==", value)
            out.append(models.Filter(must_not=[inner]))
            continue
        out.append(_build_match(field, op, value))
    return out


@lru_cache(maxsize=512)
def _compile_filter_cached(spec_json: str) -> models.Filter | None:
    spec = json.loads(spec_json)
    if not spec:
        return None
    must = _build_clause(spec.get("must", []))
    should = _build_clause(spec.get("should", []))
    must_not = _build_clause(spec.get("must_not", []))
    if not (must or should or must_not):
        return None
    return models.Filter(must=must or None, should=should or None, must_not=must_not or None)


def compile_filter(spec: dict | None) -> models.Filter | None:
    """Compile a high-level filter dict into a cached Qdrant Filter object."""
    if not spec:
        return None
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return _compile_filter_cached(spec_json)

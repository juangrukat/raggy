"""Filter builder — converts filterable field definitions into Qdrant Filter objects."""

from typing import Any, Callable

from qdrant_client import models

from raggy_mcp.qdrant import ArbitraryFilter
from raggy_mcp.settings import METADATA_PATH, FilterableField

# ── Condition builders ────────────────────────────────────────────────────
# Each builder takes (field_name, value) and returns a condition to append
# to either must or must_not. Return value: (models.FieldCondition, False) for
# must, or (models.FieldCondition, True) for must_not.


def _match_value(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, match=models.MatchValue(value=value))]


def _match_any(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, match=models.MatchAny(any=value))]


def _match_except(key: str, value: Any) -> list[models.FieldCondition]:
    return [
        models.FieldCondition(key=key, match=models.MatchExcept(**{"except": value}))
    ]


def _range_gt(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, range=models.Range(gt=value))]


def _range_gte(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, range=models.Range(gte=value))]


def _range_lt(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, range=models.Range(lt=value))]


def _range_lte(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, range=models.Range(lte=value))]


def _must_not(key: str, value: Any) -> list[models.FieldCondition]:
    return [models.FieldCondition(key=key, match=models.MatchValue(value=value))]


# ── Dispatch table: (field_type, condition) → builder ────────────────────

_MUST_CONDITIONS: dict[
    tuple[str, str], Callable[[str, Any], list[models.FieldCondition]]
] = {
    ("keyword", "=="): _match_value,
    ("keyword", "any"): _match_any,
    ("keyword", "except"): _match_except,
    ("integer", "=="): _match_value,
    ("integer", ">"): _range_gt,
    ("integer", ">="): _range_gte,
    ("integer", "<"): _range_lt,
    ("integer", "<="): _range_lte,
    ("integer", "any"): _match_any,
    ("integer", "except"): _match_except,
    ("float", ">"): _range_gt,
    ("float", ">="): _range_gte,
    ("float", "<"): _range_lt,
    ("float", "<="): _range_lte,
    ("boolean", "=="): _match_value,
}

_MUST_NOT_CONDITIONS: dict[
    tuple[str, str], Callable[[str, Any], list[models.FieldCondition]]
] = {
    ("keyword", "!="): _must_not,
    ("integer", "!="): _must_not,
    ("boolean", "!="): _must_not,
}


def _validate_field(
    field_name: str, field: FilterableField, values: dict[str, Any]
) -> None:
    """Validate a field exists and has a non-None value if required."""
    if field_value := values.get(field_name):
        if field_value is None and field.required:
            raise ValueError(f"Field {field_name} is required")


def make_filter(
    filterable_fields: dict[str, FilterableField], values: dict[str, Any]
) -> ArbitraryFilter:
    must_conditions: list[models.FieldCondition] = []
    must_not_conditions: list[models.FieldCondition] = []

    for raw_field_name, field_value in values.items():
        if raw_field_name not in filterable_fields:
            raise ValueError(f"Field {raw_field_name} is not a filterable field")

        field = filterable_fields[raw_field_name]

        if field_value is None:
            if field.required:
                raise ValueError(f"Field {raw_field_name} is required")
            continue

        field_key = f"{METADATA_PATH}.{raw_field_name}"
        ft = field.field_type
        cond = field.condition

        # Try must_not first (negations), then must
        if cond is not None:
            if (ft, cond) in _MUST_NOT_CONDITIONS:
                must_not_conditions.extend(
                    _MUST_NOT_CONDITIONS[(ft, cond)](field_key, field_value)  # type: ignore[index]
                )
            elif (ft, cond) in _MUST_CONDITIONS:
                must_conditions.extend(
                    _MUST_CONDITIONS[(ft, cond)](field_key, field_value)  # type: ignore[index]
                )
            else:
                raise ValueError(
                    f"Invalid condition {cond!r} for {ft} field {field_key}"
                )

    return models.Filter(
        must=must_conditions,  # type: ignore[arg-type]
        must_not=must_not_conditions,  # type: ignore[arg-type]
    ).model_dump()


# ── Index builder (unchanged, but extracted to keep filters.py focused) ────


def make_indexes(
    filterable_fields: dict[str, FilterableField],
) -> dict[str, models.PayloadSchemaType]:
    indexes: dict[str, models.PayloadSchemaType] = {}
    type_map = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "float": models.PayloadSchemaType.FLOAT,
        "boolean": models.PayloadSchemaType.BOOL,
    }
    for field_name, field in filterable_fields.items():
        schema_type = type_map.get(field.field_type)
        if schema_type is None:
            raise ValueError(
                f"Unsupported field type {field.field_type} for field {field_name}"
            )
        indexes[f"{METADATA_PATH}.{field_name}"] = schema_type
    return indexes

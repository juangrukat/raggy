import pytest

from raggy_mcp.mcp_runtime.plan_registry import PlanRegistry


async def test_plan_registry_consumes_once():
    registry = PlanRegistry(ttl_seconds=60)
    plan = await registry.create("delete_collection", {"target": "docs"})

    consumed = await registry.consume(plan.plan_id, expected_tool="delete_collection")

    assert consumed.payload["target"] == "docs"
    with pytest.raises(ValueError, match="already applied"):
        await registry.consume(plan.plan_id, expected_tool="delete_collection")


async def test_plan_registry_rejects_wrong_tool():
    registry = PlanRegistry(ttl_seconds=60)
    plan = await registry.create("ingest_folder", {"folder": "/tmp/docs"})

    with pytest.raises(ValueError, match="not 'delete_collection'"):
        await registry.consume(plan.plan_id, expected_tool="delete_collection")


async def test_plan_registry_evicts_expired():
    registry = PlanRegistry(ttl_seconds=0)
    plan = await registry.create("delete_collection", {"target": "docs"})

    with pytest.raises(ValueError, match="not found"):
        await registry.consume(plan.plan_id, expected_tool="delete_collection")

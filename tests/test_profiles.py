from raggy_mcp.mcp_runtime.profiles import ToolProfile, is_tool_visible


def test_profile_parse_aliases():
    assert ToolProfile.parse(None) == ToolProfile.CANONICAL
    assert ToolProfile.parse("min") == ToolProfile.MINIMAL
    assert ToolProfile.parse("admin") == ToolProfile.FULL


def test_profile_visibility_ladder():
    assert is_tool_visible("search_documents", ToolProfile.MINIMAL)
    assert is_tool_visible("create_collection", ToolProfile.CANONICAL)
    assert not is_tool_visible("create_collection", ToolProfile.MINIMAL)
    assert is_tool_visible("qdrant_find", ToolProfile.FULL)
    assert not is_tool_visible("qdrant_find", ToolProfile.CANONICAL)

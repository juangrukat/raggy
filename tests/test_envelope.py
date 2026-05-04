from raggy_mcp.mcp_runtime.envelope import failure, success


def test_success_envelope_shape():
    env = success({"ok": True}, profile="minimal", duration_ms=3)

    assert env["contract"]["contract_version"] == "1.0"
    assert env["contract"]["profile"] == "minimal"
    assert env["data"] == {"ok": True}
    assert env["observability"]["duration_ms"] == 3


def test_failure_envelope_shape():
    env = failure("bad_input", "Nope", profile="canonical", retryable=True)

    assert env["error"]["code"] == "bad_input"
    assert env["error"]["message"] == "Nope"
    assert env["error"]["retryable"] is True
    assert "data" not in env

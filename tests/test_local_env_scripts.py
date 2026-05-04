import os
import subprocess


def _source_local_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", "-lc", "source scripts/local-env.sh >/dev/null; env"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    parsed = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
    return parsed


def test_local_env_defaults_to_embedded_storage():
    env = _source_local_env({"QDRANT_URL": "", "QDRANT_MODE": "", "QDRANT_LOCAL_PATH": ""})

    assert env["QDRANT_MODE"] == "embedded"
    assert env["QDRANT_LOCAL_PATH"].endswith(".local/qdrant-storage")
    assert "QDRANT_URL" not in env or env["QDRANT_URL"] == ""


def test_local_env_server_mode_uses_url_without_local_path():
    env = _source_local_env(
        {
            "QDRANT_MODE": "server",
            "QDRANT_URL": "http://127.0.0.1:6333",
            "QDRANT_LOCAL_PATH": "/tmp/should-not-leak",
        }
    )

    assert env["QDRANT_MODE"] == "server"
    assert env["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert "QDRANT_LOCAL_PATH" not in env

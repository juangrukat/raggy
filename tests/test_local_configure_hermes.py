import subprocess

import yaml


def test_local_configure_hermes_writes_direct_venv_entry(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mcp_servers:\n  other:\n    command: echo\n", encoding="utf-8")

    subprocess.run(
        [
            "python",
            "scripts/local-configure-hermes.py",
            "--config",
            str(config_path),
            "--no-backup",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    qdrant = config["mcp_servers"]["qdrant"]

    assert qdrant["command"].endswith("/.venv/bin/raggy-mcp")
    assert qdrant["args"] == []
    assert qdrant["env"]["QDRANT_MODE"] == "server"
    assert qdrant["env"]["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert "QDRANT_LOCAL_PATH" not in qdrant["env"]
    assert config["mcp_servers"]["other"]["command"] == "echo"

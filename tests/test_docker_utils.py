from raggy_mcp import docker_utils


def test_stop_qdrant_container_skips_external_server_mode(monkeypatch):
    calls = []

    monkeypatch.setenv("QDRANT_MODE", "server")
    monkeypatch.setenv("QDRANT_AUTO_DOCKER", "false")
    monkeypatch.setattr(docker_utils, "is_qdrant_container_running", lambda: True)
    monkeypatch.setattr(
        docker_utils.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )

    docker_utils.stop_qdrant_container()

    assert calls == []


def test_should_manage_qdrant_container_requires_docker_auto_mode(monkeypatch):
    monkeypatch.setenv("QDRANT_MODE", "docker")
    monkeypatch.setenv("QDRANT_AUTO_DOCKER", "true")

    assert docker_utils.should_manage_qdrant_container() is True

    monkeypatch.setenv("QDRANT_MODE", "server")

    assert docker_utils.should_manage_qdrant_container() is False

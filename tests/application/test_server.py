"""Container entry point: environment-driven config and the TCP_NODELAY fix."""

from __future__ import annotations

import socket

from application import server


def test_the_config_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("WORKERS", "3")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    config = server.build_config()
    assert (config.host, config.port, config.workers) == ("127.0.0.1", 9123, 3)
    assert config.log_level == "warning"


def test_the_defaults_match_the_container_contract(monkeypatch):
    for name in ("HOST", "PORT", "WORKERS", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    config = server.build_config()
    assert (config.host, config.port, config.workers) == ("0.0.0.0", 8000, 1)


def test_an_invalid_worker_count_falls_back_to_one(monkeypatch):
    monkeypatch.setenv("WORKERS", "0")
    assert server.build_config().workers == 1


def test_the_shared_socket_disables_nagle(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "0")
    sock = server.bind_listening_socket(server.build_config())
    try:
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0
    finally:
        sock.close()

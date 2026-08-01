import json
import time

from plugins.memory.kanister_gateway import (
    KanisterGatewayMemoryProvider,
    _format_recall_context,
    _load_config,
    _normalize_recall_payload,
    _save_config,
)


class FakeKanisterClient:
    def __init__(self, endpoint, *, api_key="", timeout=8.0, fail_writes=False):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.fail_writes = fail_writes
        self.events = []
        self.outcomes = []
        self.recall_payloads = []
        self.prefetch_payloads = []

    def health(self):
        return {"status": "ok", "api_key_seen": bool(self.api_key)}

    def recall(self, payload):
        self.recall_payloads.append(payload)
        return {
            "results": [
                {
                    "content": "User prefers concise engineering summaries.",
                    "score": 0.91,
                    "provenance": {"source": "kanister://facts/profile", "stale": False},
                }
            ]
        }

    def prefetch(self, payload):
        self.prefetch_payloads.append(payload)
        return {
            "results": [
                {
                    "text": "RED-3052 is the unified Kanister memory plane rollout.",
                    "provenance": {"uri": "kanister://tickets/RED-3052", "partial": True},
                }
            ]
        }

    def event(self, payload):
        if self.fail_writes:
            raise RuntimeError("sidecar offline token=secret")
        self.events.append(payload)
        return {"ok": True}

    def outcome(self, payload):
        if self.fail_writes:
            raise RuntimeError("sidecar offline token=secret")
        self.outcomes.append(payload)
        return {"ok": True}


def _drain(provider):
    provider._write_queue.join()


def test_config_is_profile_scoped_and_env_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("KANISTER_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("KANISTER_GATEWAY_ENDPOINT", "127.0.0.1:9777")
    monkeypatch.setenv("KANISTER_GATEWAY_NAMESPACE", "env-ns")

    _save_config(
        {
            "endpoint": "http://127.0.0.1:9555",
            "namespace": "profile-ns",
            "api_key": "must-not-save",
            "recall_limit": 9,
        },
        str(tmp_path),
    )

    raw = json.loads((tmp_path / "kanister_gateway" / "config.json").read_text())
    assert "api_key" not in raw

    cfg = _load_config(tmp_path)
    assert cfg["endpoint"] == "http://127.0.0.1:9777"
    assert cfg["namespace"] == "env-ns"
    assert cfg["recall_limit"] == 9


def test_default_endpoint_matches_the_host_sidecar_port(monkeypatch, tmp_path):
    for name in (
        "KANISTER_GATEWAY_ENDPOINT",
        "KANISTER_GATEWAY_API_KEY",
        "KANISTER_GATEWAY_NAMESPACE",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = _load_config(tmp_path)

    assert cfg["endpoint"] == "http://127.0.0.1:17890"


def test_tools_are_gateway_only_and_do_not_expose_backends():
    provider = KanisterGatewayMemoryProvider()

    names = {schema["name"] for schema in provider.get_tool_schemas()}

    assert names == {"kanister_recall", "kanister_remember", "kanister_status"}
    assert not any("hindsight" in name or "viking" in name for name in names)


def test_normalized_recall_context_marks_stale_and_partial():
    normalized = _normalize_recall_payload(
        {
            "partial": True,
            "results": [
                {
                    "text": "A stale project fact",
                    "score": "0.5",
                    "provenance": {"source": "kanister://facts/1", "stale": True},
                }
            ],
        }
    )

    context = _format_recall_context(
        normalized,
        generated_at=time.time() - 30,
        stale_after_seconds=1,
        from_cache=True,
    )

    assert "## Kanister Memory Context" in context
    assert "[stale|partial score=0.500]" in context
    assert "source: kanister://facts/1" in context


def test_sync_turn_and_memory_write_emit_idempotent_events(monkeypatch, tmp_path):
    clients = []

    def make_client(*args, **kwargs):
        client = FakeKanisterClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("plugins.memory.kanister_gateway._KanisterGatewayClient", make_client)
    monkeypatch.setattr("plugins.memory.kanister_gateway.get_hermes_home", lambda: tmp_path)

    provider = KanisterGatewayMemoryProvider()
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
    )
    provider.sync_turn("hello", "world", session_id="session-1")
    provider.on_memory_write("add", "memory", "User likes narrow diffs")
    provider.on_memory_write("add", "memory", "User likes narrow diffs")
    provider.on_session_end([{"role": "user", "content": "hello"}])
    _drain(provider)

    client = clients[0]
    event_ids = [event["id"] for event in client.events]
    memory_ids = [event["id"] for event in client.events if event["type"] == "memory.write"]

    assert "session.started" in {event["type"] for event in client.events}
    assert "conversation.turn" in {event["type"] for event in client.events}
    assert memory_ids[0] == memory_ids[1]
    assert len(memory_ids) == 2
    assert len(event_ids) == len(client.events)
    assert client.outcomes[0]["type"] == "session.outcome"
    assert client.outcomes[0]["id"]

    provider.shutdown()


def test_queue_safe_write_failures_do_not_raise_or_log_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("KANISTER_GATEWAY_API_KEY", "secret")

    def make_client(*args, **kwargs):
        return FakeKanisterClient(*args, **kwargs, fail_writes=True)

    monkeypatch.setattr("plugins.memory.kanister_gateway._KanisterGatewayClient", make_client)
    monkeypatch.setattr("plugins.memory.kanister_gateway.get_hermes_home", lambda: tmp_path)

    provider = KanisterGatewayMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))
    provider.sync_turn("hello", "world")
    _drain(provider)

    assert provider._failed >= 1
    assert "secret" not in provider._last_error
    assert provider._write_queue.empty()

    provider.shutdown()


def test_non_primary_context_does_not_emit_session_or_tool_writes(monkeypatch, tmp_path):
    clients = []

    def make_client(*args, **kwargs):
        client = FakeKanisterClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("plugins.memory.kanister_gateway._KanisterGatewayClient", make_client)
    monkeypatch.setattr("plugins.memory.kanister_gateway.get_hermes_home", lambda: tmp_path)

    provider = KanisterGatewayMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), agent_context="subagent")
    provider.sync_turn("hello", "world")
    remember = json.loads(provider.handle_tool_call("kanister_remember", {"content": "do not persist"}))
    _drain(provider)

    assert remember["queued"] is False
    assert clients[0].events == []
    assert clients[0].outcomes == []

    provider.shutdown()


def test_prefetch_normalizes_sidecar_results(monkeypatch, tmp_path):
    clients = []

    def make_client(*args, **kwargs):
        client = FakeKanisterClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("plugins.memory.kanister_gateway._KanisterGatewayClient", make_client)
    monkeypatch.setattr("plugins.memory.kanister_gateway.get_hermes_home", lambda: tmp_path)

    provider = KanisterGatewayMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))
    provider.queue_prefetch("what is RED-3052?")
    context = provider.prefetch("what is RED-3052?")

    assert "RED-3052 is the unified Kanister memory plane rollout." in context
    assert "[partial]" in context or "|partial" in context
    assert clients[0].prefetch_payloads[0]["query"] == "what is RED-3052?"

    provider.shutdown()


def test_tool_recall_returns_normalized_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "plugins.memory.kanister_gateway._KanisterGatewayClient",
        lambda *args, **kwargs: FakeKanisterClient(*args, **kwargs),
    )
    monkeypatch.setattr("plugins.memory.kanister_gateway.get_hermes_home", lambda: tmp_path)

    provider = KanisterGatewayMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))

    result = json.loads(provider.handle_tool_call("kanister_recall", {"query": "preferences"}))

    assert result["count"] == 1
    assert result["results"][0]["text"] == "User prefers concise engineering summaries."
    assert "Kanister Memory Context" in result["context"]

    provider.shutdown()

"""Kanister Gateway memory plugin.

This provider is intentionally a thin MemoryProvider boundary around a
host-local sidecar. It does not import Hindsight, OpenViking, or any canonical
memory implementation directly; the sidecar owns those integrations.

Config resolution:
  1. $HERMES_HOME/kanister_gateway/config.json (profile-scoped)
  2. memory.kanister_gateway in config.yaml (setup compatibility)
  3. Environment variables

Environment overrides:
  KANISTER_GATEWAY_ENDPOINT       default: http://127.0.0.1:8765
  KANISTER_GATEWAY_API_KEY        optional bearer token
  KANISTER_GATEWAY_NAMESPACE      default: hermes
  KANISTER_GATEWAY_TIMEOUT        default: 8.0 seconds
  KANISTER_GATEWAY_QUEUE_SIZE     default: 128
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
_DEFAULT_TIMEOUT = 8.0
_DEFAULT_QUEUE_SIZE = 128
_DEFAULT_RECALL_LIMIT = 6
_DEFAULT_PREFETCH_STALE_SECONDS = 300
_MAX_CONTEXT_ITEM_CHARS = 900
_MAX_TOOL_RESULT_CHARS = 6000


RECALL_SCHEMA = {
    "name": "kanister_recall",
    "description": "Recall normalized context from the Kanister memory gateway.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The memory question or search query."},
            "limit": {
                "type": "integer",
                "description": "Maximum number of normalized memories to return.",
                "default": _DEFAULT_RECALL_LIMIT,
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "kanister_remember",
    "description": "Send an explicit memory event to the Kanister gateway.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Durable fact, preference, or project note to store."},
            "target": {
                "type": "string",
                "description": "Memory target.",
                "enum": ["memory", "user"],
                "default": "memory",
            },
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "kanister_status",
    "description": "Return Kanister gateway provider status without exposing credentials.",
    "parameters": {"type": "object", "properties": {}},
}


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 10000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_float(value: Any, default: float, *, minimum: float = 0.1, maximum: float = 120.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _redact_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        if not parsed.query:
            return url
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "[redacted]", parsed.fragment))
    except Exception:
        return "[invalid-url]"


def _normalize_endpoint(endpoint: Any) -> str:
    text = str(endpoint or _DEFAULT_ENDPOINT).strip() or _DEFAULT_ENDPOINT
    parsed = urllib.parse.urlsplit(text)
    if not parsed.scheme:
        text = "http://" + text
        parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _DEFAULT_ENDPOINT
    return text.rstrip("/")


def _load_config(hermes_home: str | Path | None = None) -> dict:
    """Load profile-safe Kanister Gateway config with env overrides."""
    home = Path(hermes_home) if hermes_home else get_hermes_home()
    cfg: dict[str, Any] = {}

    profile_path = home / "kanister_gateway" / "config.json"
    if profile_path.exists():
        try:
            loaded = json.loads(profile_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            logger.warning("Kanister Gateway config at %s is unreadable", profile_path)

    if not cfg:
        try:
            from hermes_cli.config import cfg_get, load_config

            nested = cfg_get(load_config(), "memory", "kanister_gateway", default={})
            if isinstance(nested, dict):
                cfg.update(nested)
        except Exception:
            pass

    env_map = {
        "endpoint": os.environ.get("KANISTER_GATEWAY_ENDPOINT"),
        "api_key": os.environ.get("KANISTER_GATEWAY_API_KEY"),
        "namespace": os.environ.get("KANISTER_GATEWAY_NAMESPACE"),
        "timeout": os.environ.get("KANISTER_GATEWAY_TIMEOUT"),
        "queue_size": os.environ.get("KANISTER_GATEWAY_QUEUE_SIZE"),
        "recall_limit": os.environ.get("KANISTER_GATEWAY_RECALL_LIMIT"),
        "prefetch_stale_seconds": os.environ.get("KANISTER_GATEWAY_PREFETCH_STALE_SECONDS"),
        "auto_recall": os.environ.get("KANISTER_GATEWAY_AUTO_RECALL"),
        "auto_capture": os.environ.get("KANISTER_GATEWAY_AUTO_CAPTURE"),
    }
    for key, value in env_map.items():
        if value not in {None, ""}:
            cfg[key] = value

    cfg["endpoint"] = _normalize_endpoint(cfg.get("endpoint", _DEFAULT_ENDPOINT))
    cfg["api_key"] = str(cfg.get("api_key") or "")
    cfg["namespace"] = str(cfg.get("namespace") or "hermes")
    cfg["timeout"] = _parse_float(cfg.get("timeout"), _DEFAULT_TIMEOUT)
    cfg["queue_size"] = _parse_int(cfg.get("queue_size"), _DEFAULT_QUEUE_SIZE, minimum=8, maximum=4096)
    cfg["recall_limit"] = _parse_int(cfg.get("recall_limit"), _DEFAULT_RECALL_LIMIT, minimum=1, maximum=20)
    cfg["prefetch_stale_seconds"] = _parse_int(
        cfg.get("prefetch_stale_seconds"),
        _DEFAULT_PREFETCH_STALE_SECONDS,
        minimum=1,
        maximum=86400,
    )
    cfg["auto_recall"] = _parse_bool(cfg.get("auto_recall"), True)
    cfg["auto_capture"] = _parse_bool(cfg.get("auto_capture"), True)
    return cfg


def _save_config(values: Dict[str, Any], hermes_home: str) -> None:
    config_dir = Path(hermes_home) / "kanister_gateway"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing.update(loaded)
        except Exception:
            pass
    sanitized = {k: v for k, v in (values or {}).items() if k != "api_key"}
    if "endpoint" in sanitized:
        sanitized["endpoint"] = _normalize_endpoint(sanitized["endpoint"])
    existing.update(sanitized)
    config_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _KanisterGatewayClient:
    """Small stdlib JSON client for the host-local sidecar."""

    def __init__(self, endpoint: str, *, api_key: str = "", timeout: float = _DEFAULT_TIMEOUT):
        self.endpoint = _normalize_endpoint(endpoint)
        self.api_key = api_key or ""
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=body,
            headers=self._headers(),
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Kanister Gateway HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Kanister Gateway request failed: {exc}") from exc
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Kanister Gateway returned invalid JSON") from exc
        if not isinstance(data, dict):
            return {"result": data}
        if data.get("status") == "error":
            error = data.get("error") or data.get("message") or "unknown error"
            raise RuntimeError(str(error))
        return data

    def health(self) -> dict:
        return self.request("GET", "/health")

    def recall(self, payload: dict) -> dict:
        return self.request("POST", "/v1/recall", payload)

    def prefetch(self, payload: dict) -> dict:
        return self.request("POST", "/v1/prefetch", payload)

    def event(self, payload: dict) -> dict:
        return self.request("POST", "/v1/events", payload)

    def outcome(self, payload: dict) -> dict:
        return self.request("POST", "/v1/outcomes", payload)


def _stable_id(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part or "").encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _clean_content(value: Any, *, limit: int = 12000) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    if len(text) > limit:
        return text[:limit] + "\n\n[truncated]"
    return text


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "summary", "memory", "snippet"):
            value = item.get(key)
            if value:
                return str(value)
    return ""


def _extract_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("results", "memories", "items", "context", "matches"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return _extract_items(result)
    return []


def _provenance(item: Any, payload: Any) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    if isinstance(item, dict):
        raw = item.get("provenance")
        if isinstance(raw, dict):
            provenance.update(raw)
        for key in ("source", "citation", "uri", "id", "created_at", "updated_at"):
            if item.get(key) and key not in provenance:
                provenance[key] = item[key]
        if item.get("stale") is not None:
            provenance["stale"] = bool(item.get("stale"))
        if item.get("partial") is not None:
            provenance["partial"] = bool(item.get("partial"))
    if isinstance(payload, dict):
        if payload.get("stale") is not None:
            provenance.setdefault("stale", bool(payload.get("stale")))
        if payload.get("partial") is not None:
            provenance.setdefault("partial", bool(payload.get("partial")))
    return provenance


def _normalize_recall_payload(payload: Any, *, limit: int = _DEFAULT_RECALL_LIMIT) -> list[dict[str, Any]]:
    normalized = []
    for item in _extract_items(payload)[:limit]:
        text = _item_text(item).strip()
        if not text:
            continue
        entry: dict[str, Any] = {
            "text": text,
            "provenance": _provenance(item, payload),
        }
        if isinstance(item, dict) and item.get("score") is not None:
            try:
                entry["score"] = float(item["score"])
            except (TypeError, ValueError):
                pass
        normalized.append(entry)
    return normalized


def _format_recall_context(
    normalized: list[dict[str, Any]],
    *,
    generated_at: float | None = None,
    stale_after_seconds: int = _DEFAULT_PREFETCH_STALE_SECONDS,
    from_cache: bool = False,
) -> str:
    if not normalized:
        return ""
    age = 0.0
    cached_stale = False
    if generated_at:
        age = max(0.0, time.time() - generated_at)
        cached_stale = age > stale_after_seconds
    lines = ["## Kanister Memory Context"]
    if from_cache:
        state = "stale" if cached_stale else "fresh"
        lines.append(f"Cache: {state}; age={int(age)}s.")
    for idx, entry in enumerate(normalized, start=1):
        prov = entry.get("provenance") or {}
        flags = []
        if prov.get("stale") or cached_stale:
            flags.append("stale")
        if prov.get("partial"):
            flags.append("partial")
        if not flags:
            flags.append("fresh")
        text = entry["text"]
        if len(text) > _MAX_CONTEXT_ITEM_CHARS:
            text = text[:_MAX_CONTEXT_ITEM_CHARS] + "..."
            if "partial" not in flags:
                flags.append("partial")
        source = prov.get("source") or prov.get("uri") or prov.get("citation") or prov.get("id") or "kanister"
        score = ""
        if entry.get("score") is not None:
            score = f" score={entry['score']:.3f}"
        lines.append(f"{idx}. [{'|'.join(flags)}{score}] {text} (source: {source})")
    return "\n".join(lines)


class KanisterGatewayMemoryProvider(MemoryProvider):
    """MemoryProvider adapter for the Unified Kanister Memory Plane sidecar."""

    def __init__(self):
        self._config = _load_config()
        self._client: _KanisterGatewayClient | None = None
        self._endpoint = self._config["endpoint"]
        self._namespace = self._config["namespace"]
        self._session_id = ""
        self._agent_context = "primary"
        self._agent_identity = ""
        self._platform = ""
        self._write_enabled = True
        self._auto_recall = True
        self._auto_capture = True
        self._recall_limit = _DEFAULT_RECALL_LIMIT
        self._prefetch_stale_seconds = _DEFAULT_PREFETCH_STALE_SECONDS
        self._prefetch_result: list[dict[str, Any]] = []
        self._prefetch_generated_at = 0.0
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._write_queue: queue.Queue = queue.Queue(maxsize=_DEFAULT_QUEUE_SIZE)
        self._writer_thread: threading.Thread | None = None
        self._shutting_down = threading.Event()
        self._turn_index = 0
        self._queued = 0
        self._dropped = 0
        self._failed = 0
        self._last_error = ""

    @property
    def name(self) -> str:
        return "kanister_gateway"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("endpoint"))

    def get_config_schema(self):
        return [
            {
                "key": "endpoint",
                "description": "Kanister Gateway sidecar URL",
                "required": True,
                "default": _DEFAULT_ENDPOINT,
                "env_var": "KANISTER_GATEWAY_ENDPOINT",
            },
            {
                "key": "api_key",
                "description": "Kanister Gateway bearer token (optional for host-local sidecar)",
                "secret": True,
                "env_var": "KANISTER_GATEWAY_API_KEY",
            },
            {
                "key": "namespace",
                "description": "Kanister namespace",
                "default": "hermes",
                "env_var": "KANISTER_GATEWAY_NAMESPACE",
            },
            {
                "key": "recall_limit",
                "description": "Default recall result limit",
                "default": str(_DEFAULT_RECALL_LIMIT),
                "env_var": "KANISTER_GATEWAY_RECALL_LIMIT",
            },
        ]

    def save_config(self, values, hermes_home):
        _save_config(values, hermes_home)

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._config = _load_config(hermes_home)
        self._endpoint = self._config["endpoint"]
        self._namespace = self._config["namespace"]
        self._session_id = session_id
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._agent_identity = str(kwargs.get("agent_identity") or "")
        self._platform = str(kwargs.get("platform") or "")
        self._write_enabled = self._agent_context not in {"cron", "flush", "subagent"}
        self._auto_recall = bool(self._config["auto_recall"])
        self._auto_capture = bool(self._config["auto_capture"])
        self._recall_limit = int(self._config["recall_limit"])
        self._prefetch_stale_seconds = int(self._config["prefetch_stale_seconds"])
        self._write_queue = queue.Queue(maxsize=int(self._config["queue_size"]))
        self._client = _KanisterGatewayClient(
            self._endpoint,
            api_key=self._config.get("api_key", ""),
            timeout=float(self._config["timeout"]),
        )
        self._start_writer()
        self._enqueue_session_event("session.started", metadata=self._runtime_metadata())

    def system_prompt_block(self) -> str:
        endpoint = _redact_url(self._endpoint)
        return (
            "# Kanister Gateway Memory\n"
            f"Active via host-local sidecar: {endpoint}.\n"
            "Use kanister_recall for explicit recall and kanister_remember for durable facts. "
            "The gateway normalizes provenance and may mark recalled context stale or partial."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.5)
        with self._prefetch_lock:
            result = list(self._prefetch_result)
            generated_at = self._prefetch_generated_at
            self._prefetch_result = []
            self._prefetch_generated_at = 0.0
        return _format_recall_context(
            result,
            generated_at=generated_at,
            stale_after_seconds=self._prefetch_stale_seconds,
            from_cache=True,
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._client or not self._auto_recall or not query.strip():
            return

        sid = session_id or self._session_id
        payload = self._base_payload(sid)
        payload.update({"query": _clean_content(query, limit=2000), "limit": self._recall_limit})

        def _run() -> None:
            try:
                response = self._client.prefetch(payload)
                normalized = _normalize_recall_payload(response, limit=self._recall_limit)
                with self._prefetch_lock:
                    self._prefetch_result = normalized
                    self._prefetch_generated_at = time.time()
            except Exception as exc:
                self._last_error = self._safe_error(exc)
                logger.debug("Kanister Gateway prefetch failed: %s", self._last_error)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="kanister-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._write_enabled or not self._auto_capture or not user_content or not assistant_content:
            return
        self._turn_index += 1
        sid = session_id or self._session_id
        payload = self._base_payload(sid)
        clean_user = _clean_content(user_content)
        clean_assistant = _clean_content(assistant_content)
        event_id = _stable_id("turn", self._namespace, sid, self._turn_index, clean_user, clean_assistant)
        payload.update({
            "id": event_id,
            "type": "conversation.turn",
            "source": "hermes",
            "messages": [
                {"role": "user", "content": clean_user},
                {"role": "assistant", "content": clean_assistant},
            ],
            "metadata": {**self._runtime_metadata(), "turn_index": self._turn_index},
        })
        self._enqueue("event", payload)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._enqueue_session_event("session.ended", metadata={**self._runtime_metadata(), "message_count": len(messages or [])})
        if self._write_enabled:
            outcome = self._base_payload(self._session_id)
            outcome.update({
                "id": _stable_id("outcome", self._namespace, self._session_id, "session.ended", len(messages or [])),
                "type": "session.outcome",
                "source": "hermes",
                "status": "completed",
                "metadata": {
                    **self._runtime_metadata(),
                    "turn_count": self._turn_index,
                    "message_count": len(messages or []),
                },
            })
            self._enqueue("outcome", outcome)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if not new_session_id:
            return
        old_session_id = self._session_id
        self._enqueue_session_event(
            "session.switched",
            session_id=old_session_id,
            metadata={
                **self._runtime_metadata(),
                "new_session_id": new_session_id,
                "parent_session_id": parent_session_id,
                "reset": bool(reset),
            },
        )
        self._session_id = new_session_id
        if reset:
            self._turn_index = 0

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._write_enabled or action not in {"add", "replace"} or not content:
            return
        payload = self._base_payload(self._session_id)
        clean = _clean_content(content)
        payload.update({
            "id": _stable_id("memory-write", self._namespace, self._session_id, action, target, clean),
            "type": "memory.write",
            "source": "hermes",
            "action": action,
            "target": target or "memory",
            "content": clean,
            "metadata": {**self._runtime_metadata(), **dict(metadata or {})},
        })
        self._enqueue("event", payload)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if not self._write_enabled or not task or not result:
            return
        payload = self._base_payload(self._session_id)
        clean_task = _clean_content(task, limit=6000)
        clean_result = _clean_content(result, limit=8000)
        payload.update({
            "id": _stable_id("delegation", self._namespace, self._session_id, child_session_id, clean_task, clean_result),
            "type": "delegation.outcome",
            "source": "hermes",
            "task": clean_task,
            "result": clean_result,
            "child_session_id": child_session_id,
            "metadata": {**self._runtime_metadata(), **dict(kwargs or {})},
        })
        self._enqueue("outcome", payload)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "kanister_recall":
            return self._tool_recall(args)
        if tool_name == "kanister_remember":
            return self._tool_remember(args)
        if tool_name == "kanister_status":
            return self._tool_status()
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        self._shutting_down.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

    def _tool_recall(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        if not self._client:
            return tool_error("Kanister Gateway is not initialized")
        limit = _parse_int(args.get("limit"), self._recall_limit, minimum=1, maximum=20)
        payload = self._base_payload(self._session_id)
        payload.update({"query": _clean_content(query, limit=2000), "limit": limit})
        try:
            response = self._client.recall(payload)
            normalized = _normalize_recall_payload(response, limit=limit)
            return json.dumps({
                "results": normalized,
                "context": _format_recall_context(normalized),
                "count": len(normalized),
            }, ensure_ascii=False)[:_MAX_TOOL_RESULT_CHARS]
        except Exception as exc:
            return tool_error(f"Kanister recall failed: {self._safe_error(exc)}")

    def _tool_remember(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        if not self._write_enabled:
            return json.dumps({"queued": False, "provider": self.name, "reason": "writes disabled"})
        target = str(args.get("target") or "memory")
        self.on_memory_write("add", target, content, metadata={"origin": "kanister_remember_tool"})
        return json.dumps({"queued": True, "target": target, "provider": self.name})

    def _tool_status(self) -> str:
        status: dict[str, Any] = {
            "provider": self.name,
            "endpoint": _redact_url(self._endpoint),
            "namespace": self._namespace,
            "write_enabled": self._write_enabled,
            "queued": self._queued,
            "pending": self._write_queue.qsize(),
            "dropped": self._dropped,
            "failed": self._failed,
        }
        if self._last_error:
            status["last_error"] = self._last_error
        if self._client:
            try:
                health = self._client.health()
                status["sidecar"] = "reachable"
                if isinstance(health, dict):
                    status["health"] = {k: v for k, v in health.items() if "key" not in str(k).lower() and "token" not in str(k).lower()}
            except Exception as exc:
                status["sidecar"] = "unreachable"
                status["last_error"] = self._safe_error(exc)
        return json.dumps(status, ensure_ascii=False)

    def _start_writer(self) -> None:
        if self._writer_thread and self._writer_thread.is_alive():
            return

        def _run() -> None:
            while not self._shutting_down.is_set() or not self._write_queue.empty():
                try:
                    kind, payload = self._write_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if not self._client:
                        raise RuntimeError("client not initialized")
                    if kind == "event":
                        self._client.event(payload)
                    elif kind == "outcome":
                        self._client.outcome(payload)
                    else:
                        raise RuntimeError(f"unknown write kind: {kind}")
                except Exception as exc:
                    self._failed += 1
                    self._last_error = self._safe_error(exc)
                    logger.debug("Kanister Gateway %s write failed: %s", kind, self._last_error)
                finally:
                    self._write_queue.task_done()

        self._shutting_down.clear()
        self._writer_thread = threading.Thread(target=_run, daemon=True, name="kanister-writer")
        self._writer_thread.start()

    def _enqueue_session_event(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._write_enabled:
            return
        sid = session_id or self._session_id
        payload = self._base_payload(sid)
        payload.update({
            "id": _stable_id("session-event", self._namespace, sid, event_type),
            "type": event_type,
            "source": "hermes",
            "metadata": dict(metadata or {}),
        })
        self._enqueue("event", payload)

    def _enqueue(self, kind: str, payload: dict) -> None:
        if not payload:
            return
        try:
            self._write_queue.put_nowait((kind, payload))
            self._queued += 1
        except queue.Full:
            self._dropped += 1
            try:
                self._write_queue.get_nowait()
                self._write_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._write_queue.put_nowait((kind, payload))
                self._queued += 1
            except queue.Full:
                self._dropped += 1

    def _base_payload(self, session_id: str) -> dict[str, Any]:
        return {
            "namespace": self._namespace,
            "session_id": session_id or self._session_id,
            "source": "hermes",
            "timestamp": time.time(),
        }

    def _runtime_metadata(self) -> dict[str, Any]:
        data = {
            "platform": self._platform,
            "agent_context": self._agent_context,
        }
        if self._agent_identity:
            data["agent_identity"] = self._agent_identity
        return data

    def _safe_error(self, exc: BaseException) -> str:
        text = str(exc)
        for secret in (
            os.environ.get("KANISTER_GATEWAY_API_KEY", ""),
            str(self._config.get("api_key") or ""),
        ):
            if secret:
                text = text.replace(secret, "[redacted]")
        if len(text) > 300:
            text = text[:300] + "..."
        return text


def register(ctx) -> None:
    ctx.register_memory_provider(KanisterGatewayMemoryProvider())

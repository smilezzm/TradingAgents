"""The ``session`` provider: the model is the Claude Code session driving the run.

Nothing is sent to a model API. Each call the pipeline makes is written to a
queue directory as a request, and the call waits until the session writes its
answer beside it; ``tradingagents session`` shows the requests and takes the
answers. The session runs on its own sign-in, so a run needs no API key.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient

POLL_SECONDS = 0.2
# A request nobody answers in this long ends its call instead of holding the run forever.
DEFAULT_TIMEOUT = 3600.0

_ids_lock = threading.Lock()
_ids: dict[Path, itertools.count] = {}


class SessionQueue:
    """A run's requests and the session's answers, as JSON files in one directory."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.requests = self.path / "requests"
        self.answers = self.path / "answers"
        self.status_file = self.path / "run.json"

    def reset(self) -> None:
        """Forget every request, answer and status, for a new run."""
        for directory in (self.requests, self.answers):
            shutil.rmtree(directory, ignore_errors=True)
        self.status_file.unlink(missing_ok=True)
        (self.path / "shown.json").unlink(missing_ok=True)

    def _new_id(self) -> str:
        with _ids_lock:
            counter = _ids.get(self.path)
            if counter is None:
                used = [int(p.stem[1:]) for d in (self.requests, self.answers) if d.is_dir()
                        for p in d.glob("r*.json") if p.stem[1:].isdigit()]
                counter = _ids[self.path] = itertools.count(max(used, default=0) + 1)
            return f"r{next(counter)}"

    def ask(self, request: dict) -> str:
        """Queue a request; returns its id."""
        request_id = self._new_id()
        _write_json(self.requests / f"{request_id}.json", {**request, "id": request_id})
        return request_id

    def wait(self, request_id: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
        """The answer to a request, once the session has written it."""
        path = self.answers / f"{request_id}.json"
        deadline = time.monotonic() + timeout
        while not path.exists():
            if time.monotonic() > deadline:
                (self.requests / f"{request_id}.json").unlink(missing_ok=True)
                raise TimeoutError(f"No answer to session request {request_id} within {timeout:.0f}s")
            time.sleep(POLL_SECONDS)
        return json.loads(path.read_text(encoding="utf-8"))

    def request(self, request_id: str) -> dict:
        path = self.requests / f"{request_id}.json"
        if not path.exists():
            raise KeyError(f"No request {request_id} in {self.path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def answer_of(self, request_id: str) -> dict | None:
        path = self.answers / f"{request_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def ids(self) -> list[str]:
        """Every request's id, oldest first."""
        if not self.requests.is_dir():
            return []
        found = [p.stem for p in self.requests.glob("r*.json") if p.stem[1:].isdigit()]
        return sorted(found, key=lambda i: int(i[1:]))

    def pending(self) -> list[dict]:
        """The requests still waiting for an answer, oldest first."""
        return [self.request(i) for i in self.ids() if not (self.answers / f"{i}.json").exists()]

    def answer(self, request_id: str, content: str = "", tool_calls: list[dict] | None = None) -> None:
        """Answer a request, after checking the answer is one its caller can use."""
        request = self.request(request_id)
        if (self.answers / f"{request_id}.json").exists():
            raise ValueError(f"{request_id} is already answered.")
        calls = [{"name": c.get("name"), "args": c.get("args", {})} if isinstance(c, dict) else c
                 for c in tool_calls or []]
        check_answer(request, content, calls)
        _write_json(self.answers / f"{request_id}.json", {"content": content, "tool_calls": calls})

    def status(self) -> dict:
        if not self.status_file.exists():
            return {}
        return json.loads(self.status_file.read_text(encoding="utf-8"))

    def write_status(self, **status: Any) -> None:
        _write_json(self.status_file, status)


def _write_json(path: Path, data: dict) -> None:
    """Write whole or not at all, so a reader polling for the file never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def check_answer(request: dict, content: str, calls: list) -> None:
    """Raise ValueError, saying what to fix, when an answer does not fit its request."""
    tools = {t["name"]: t for t in request.get("tools") or []}
    if calls and not tools:
        raise ValueError("This request offers no tools; answer with text.")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError('Each call needs a "name" and "args", e.g. {"name": "get_news", "args": {...}}.')
        tool = tools.get(call["name"])
        if tool is None:
            raise ValueError(f"No tool named {call['name']!r} here; offered: {', '.join(tools)}.")
        if not isinstance(call["args"], dict):
            raise ValueError(f"The args of {call['name']} must be a JSON object.")
        _check_args(call["name"], call["args"], tool.get("parameters") or {})
    if request.get("tool_choice") and not calls:
        raise ValueError(f"This request must be answered with a call to {' or '.join(tools)}.")
    if not calls and not content.strip():
        raise ValueError("The answer is empty.")


def _check_args(name: str, args: dict, schema: dict) -> None:
    properties = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if k not in args]
    if missing:
        raise ValueError(f"{name} needs {', '.join(missing)}.")
    unknown = [k for k in args if k not in properties]
    if unknown:
        raise ValueError(f"{name} takes no {', '.join(unknown)}; it takes {', '.join(properties)}.")
    definitions = schema.get("$defs") or {}
    for key, value in args.items():
        spec = properties[key]
        ref = spec.get("$ref") or next((s["$ref"] for s in spec.get("allOf", []) if "$ref" in s), None)
        if ref:
            spec = definitions.get(ref.rsplit("/", 1)[-1], spec)
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{name}.{key} must be one of {', '.join(map(str, spec['enum']))}.")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
    )


def _message(message: BaseMessage) -> dict:
    entry: dict[str, Any] = {"role": message.type, "content": _text(message.content)}
    if isinstance(message, AIMessage) and message.tool_calls:
        entry["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        entry["name"] = message.name
    return entry


def _agent(run_manager) -> str:
    """The graph node making the call; an analyst's own graph reports the analyst."""
    metadata = getattr(run_manager, "metadata", None) or {}
    namespace = metadata.get("langgraph_checkpoint_ns") or ""
    return namespace.split("|")[0].split(":")[0] or metadata.get("langgraph_node") or ""


class SessionChatModel(BaseChatModel):
    """A chat model whose answers come from the session, through a SessionQueue."""

    model: str = "session"
    session_dir: str
    timeout: float = DEFAULT_TIMEOUT

    @property
    def _llm_type(self) -> str:
        return "session"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        schemas = [convert_to_openai_tool(tool)["function"] for tool in tools]
        return self.bind(tools=schemas, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        queue = SessionQueue(self.session_dir)
        request_id = queue.ask({
            "agent": _agent(run_manager),
            "model": self.model,
            "messages": [_message(m) for m in messages],
            "tools": kwargs.get("tools") or [],
            "tool_choice": kwargs.get("tool_choice"),
        })
        answer = queue.wait(request_id, self.timeout)
        calls = [{"name": c["name"], "args": c["args"], "id": f"call_{request_id}_{i}", "type": "tool_call"}
                 for i, c in enumerate(answer.get("tool_calls") or [])]
        message = AIMessage(content=answer.get("content") or "", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


class SessionClient(BaseLLMClient):
    """Client for the session provider; the model name only labels the tier."""

    def get_llm(self) -> Any:
        session_dir = self.kwargs.get("session_dir")
        if not session_dir:
            from tradingagents.default_config import DEFAULT_CONFIG
            session_dir = DEFAULT_CONFIG["session_dir"]
        extra = {"callbacks": self.kwargs["callbacks"]} if self.kwargs.get("callbacks") else {}
        return SessionChatModel(model=self.model, session_dir=str(session_dir), **extra)

    def validate_model(self) -> bool:
        return True

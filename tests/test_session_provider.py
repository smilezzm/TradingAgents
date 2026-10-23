"""The session provider: a run whose model is the Claude Code session answering a queue."""

import json
import threading
import time

import pytest
from langchain_core.tools import tool
from typer.testing import CliRunner

from cli.main import app
from cli.session import render
from tests.test_graph_end_to_end import ARGS, STRUCTURED, TEXT, TRADE_DATE, offline  # noqa: F401
from tradingagents.agents.schemas import ResearchPlan
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_tier_client
from tradingagents.llm_clients.session_client import SessionChatModel, SessionQueue, check_answer

SCHEMAS = {type(model).__name__: model for model in STRUCTURED.values()}


@tool
def get_quote(symbol: str, curr_date: str) -> str:
    """The last close of a symbol on a date."""
    return f"{symbol} closed at 100 on {curr_date}"


class Responder:
    """Answers a queue the way the session would: every offered tool once, then text."""

    def __init__(self, queue: SessionQueue):
        self.queue = queue
        self.seen: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        while not self._stop.is_set():
            for request in self.queue.pending():
                self.seen.append(request)
                self.queue.answer(request["id"], **self.reply(request))
            time.sleep(0.01)

    @staticmethod
    def reply(request: dict) -> dict:
        tools = request["tools"]
        if request["tool_choice"]:
            name = tools[0]["name"]
            return {"tool_calls": [{"name": name,
                                    "args": SCHEMAS[name].model_dump(mode="json", exclude_none=True)}]}
        if tools and request["messages"][-1]["role"] != "tool":
            return {"tool_calls": [
                {"name": t["name"], "args": {k: v for k, v in ARGS.items() if k in t["parameters"]["properties"]}}
                for t in tools]}
        return {"content": TEXT}


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_CONFIG, "session_dir", str(tmp_path / "session"))
    return SessionQueue(tmp_path / "session")


@pytest.mark.unit
def test_the_model_takes_text_tool_calls_and_structured_answers_from_the_queue(queue):
    model = create_tier_client({**DEFAULT_CONFIG, "llm_provider": "session"}, "quick").get_llm()
    assert isinstance(model, SessionChatModel)

    with Responder(queue):
        text = model.invoke("hello").content
        calls = model.bind_tools([get_quote]).invoke("quote NVDA").tool_calls
        plan = model.with_structured_output(ResearchPlan).invoke("plan")

    assert text == TEXT
    assert [(c["name"], c["args"]) for c in calls] == [("get_quote", {"symbol": "NVDA", "curr_date": TRADE_DATE})]
    assert calls[0]["id"]
    assert plan == STRUCTURED[ResearchPlan]


@pytest.mark.unit
def test_an_unanswered_request_times_out_and_leaves_the_queue(queue):
    model = SessionChatModel(session_dir=str(queue.path), timeout=0.2)
    with pytest.raises(TimeoutError, match="r1"):
        model.invoke("hello")
    assert queue.pending() == []


@pytest.mark.unit
@pytest.mark.parametrize(("request_", "content", "calls", "error"), [
    ({"tools": []}, "", [{"name": "get_quote", "args": {}}], "offers no tools"),
    ({"tools": []}, "  ", [], "empty"),
    ({"tools": [get_quote]}, "", [{"name": "get_news", "args": {}}], "No tool named 'get_news'"),
    ({"tools": [get_quote]}, "", [{"name": "get_quote", "args": {"symbol": "NVDA"}}], "needs curr_date"),
    ({"tools": [get_quote]}, "", [{"name": "get_quote", "args": {**ARGS}}], "takes no"),
    ({"tools": [ResearchPlan], "tool_choice": "any"}, "Buy it", [], "must be answered with a call"),
    ({"tools": [ResearchPlan], "tool_choice": "any"}, "",
     [{"name": "ResearchPlan", "args": {"recommendation": "Strong Buy", "rationale": "r",
                                        "strategic_actions": "a"}}], "must be one of Buy"),
])
def test_an_answer_that_does_not_fit_its_request_is_refused(request_, content, calls, error):
    model = SessionChatModel(session_dir="unused")
    tools = model.bind_tools(request_["tools"]).kwargs["tools"] if request_["tools"] else []
    with pytest.raises(ValueError, match=error):
        check_answer({**request_, "tools": tools}, content, calls)


@pytest.mark.unit
def test_text_already_shown_is_not_shown_again(queue):
    system = {"role": "system", "content": "You are the market analyst. " * 40}
    report = "The market report, in full. " * 40
    first = queue.ask({"agent": "Market Analyst", "messages": [system, {"role": "human", "content": "go"}],
                       "tools": [], "tool_choice": None})
    queue.answer(first, content=report)
    second = queue.ask({"agent": "Market Analyst", "messages": [system, {"role": "human", "content": "again"}],
                        "tools": [], "tool_choice": None})
    bull = queue.ask({"agent": "Bull Researcher", "tools": [], "tool_choice": None,
                      "messages": [{"role": "human", "content": f"Market report: {report}\nArgue."}]})

    assert "You are the market analyst." in render(queue.request(first), queue)
    shown = render(queue.request(second), queue)
    assert "--- system: same as in r1 ---" in shown and "You are the market analyst." not in shown
    shown = render(queue.request(bull), queue)
    assert "[your answer to r1" in shown and "The market report" not in shown
    assert "You are the market analyst." in render(queue.request(second), queue, full=True)


@pytest.mark.unit
def test_answer_takes_text_or_calls_and_refuses_a_bad_one(queue):
    runner = CliRunner()
    model = SessionChatModel(session_dir=str(queue.path))
    tools = model.bind_tools([get_quote]).kwargs["tools"]
    rid = queue.ask({"agent": "Market Analyst", "messages": [{"role": "human", "content": "go"}],
                     "tools": tools, "tool_choice": None})

    bad = runner.invoke(app, ["session", "answer", rid, "--calls", "--no-wait"], input="[{not json")
    assert bad.exit_code == 1 and "still waiting" in bad.output
    assert [r["id"] for r in queue.pending()] == [rid]

    call = json.dumps({"name": "get_quote", "args": {"symbol": "NVDA", "curr_date": TRADE_DATE}})
    good = runner.invoke(app, ["session", "answer", rid, "--calls", "--no-wait"], input=call)
    assert good.exit_code == 0, good.output
    assert queue.answer_of(rid)["tool_calls"] == [json.loads(call)]

    again = runner.invoke(app, ["session", "answer", rid, "--no-wait"], input="text")
    assert again.exit_code == 1 and "already answered" in again.output


@pytest.mark.unit
def test_next_says_when_there_is_no_run(queue):
    result = CliRunner().invoke(app, ["session", "next", "--timeout", "0"])
    assert "No session run is going" in result.output


@pytest.mark.unit
@pytest.mark.usefixtures("offline")
def test_a_session_run_reaches_a_saved_decision(tmp_path, monkeypatch, queue):
    for key, path in (("results_dir", "results"), ("data_cache_dir", "cache"), ("memory_log_path", "log.md")):
        monkeypatch.setitem(DEFAULT_CONFIG, key, str(tmp_path / path))

    with Responder(queue) as responder:
        result = CliRunner().invoke(app, ["session", "run", "NVDA", "--date", TRADE_DATE, "--no-html"])

    assert result.exit_code == 0, result.output
    assert "Decision: Overweight" in result.output
    status = queue.status()
    assert status["state"] == "done" and status["signal"] == "Overweight"
    assert (tmp_path / "results" / "reports").is_dir()
    agents = {r["agent"] for r in responder.seen}
    assert agents == {"Market Analyst", "Sentiment Analyst", "News Analyst", "Fundamentals Analyst",
                      "Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
                      "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"}
    assert {r["model"] for r in responder.seen} == {"session-quick", "session-deep"}
    finished = CliRunner().invoke(app, ["session", "next", "--timeout", "0"])
    assert "Run finished: NVDA" in finished.output
    # Reported once, the old run's end is not mistaken for the end of a run just starting.
    later = CliRunner().invoke(app, ["session", "next", "--timeout", "0"])
    assert "No session run is going" in later.output
    assert "Run finished: NVDA" in CliRunner().invoke(app, ["session", "status"]).output

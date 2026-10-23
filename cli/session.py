"""`tradingagents session`: run an analysis with a Claude Code session as the model.

`run` starts the pipeline on the session provider; every model call it makes
waits in a queue until the session answers it. `next` shows the oldest waiting
request, and `answer` takes the session's reply and shows the one after it.
Output is plain text, for the session to read.
"""

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import typer

from cli.models import AnalystType
from cli.prompts import (
    detect_asset_type,
    filter_analysts_for_asset_type,
    normalize_ticker_symbol,
    parse_analysts,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.session_client import SessionQueue
from tradingagents.portfolio import load_portfolio

session_app = typer.Typer(
    help="Run an analysis with this Claude Code session as the model: no API key needed.",
    no_args_is_help=True,
)

# Text the session has already been shown is not shown again; shorter text is
# cheaper to repeat than to refer back to.
_REPEAT_BELOW = 400


def session_config(base: dict, debate_rounds: int | None = None, risk_rounds: int | None = None) -> dict:
    """``base`` with both model tiers on the session provider."""
    config = copy.deepcopy(base)
    config.update(
        llm_provider="session", backend_url=None,
        quick_think_provider=None, deep_think_provider=None,
        quick_think_backend_url=None, deep_think_backend_url=None,
        quick_think_llm="session-quick", deep_think_llm="session-deep",
    )
    if debate_rounds is not None:
        config["max_debate_rounds"] = debate_rounds
    if risk_rounds is not None:
        config["max_risk_discuss_rounds"] = risk_rounds
    return config


def _queue() -> SessionQueue:
    return SessionQueue(DEFAULT_CONFIG["session_dir"])


def _alive(pid) -> bool:
    if not isinstance(pid, int):
        return False
    if sys.platform == "win32":  # pragma: no cover - os.kill(pid, 0) is not a probe there
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@session_app.command()
def run(
    ticker: str = typer.Argument(..., help="Ticker to analyze, e.g. NVDA or 0700.HK"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; default today"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; default all"),
    portfolio: str = typer.Option(None, "--portfolio", help="JSON file with current holdings and cash"),
    debate_rounds: int = typer.Option(None, "--debate-rounds", help="Bull/bear debate rounds"),
    risk_rounds: int = typer.Option(None, "--risk-rounds", help="Risk debate rounds"),
    html: bool = typer.Option(True, "--html/--no-html", help="Also save the report as one HTML page"),
):
    """Start a run. Keep it going in the background and answer it with `next` and `answer`."""
    from tradingagents.dataflows.date_window import get_current_date
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    queue = _queue()
    current = queue.status()
    if current.get("state") == "running" and _alive(current.get("pid")):
        typer.echo(f"A session run is already going ({current.get('ticker')} {current.get('trade_date')}, "
                   f"pid {current.get('pid')}); answer it with `tradingagents session next`, or stop it first.")
        raise typer.Exit(code=1)

    try:
        symbol = normalize_ticker_symbol(ticker)
        kind = detect_asset_type(symbol)
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
        book = load_portfolio(portfolio) if portfolio else None
    except ValueError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from None
    trade_date = date or get_current_date()

    queue.reset()
    about = {"pid": os.getpid(), "ticker": symbol, "trade_date": trade_date,
             "analysts": [a.value for a in chosen]}
    queue.write_status(state="running", **about)
    typer.echo(f"Session run: {symbol} on {trade_date}, analysts {', '.join(about['analysts'])}. "
               f"Requests wait in {queue.path}.")
    try:
        config = session_config(DEFAULT_CONFIG, debate_rounds, risk_rounds)
        graph = TradingAgentsGraph(selected_analysts=about["analysts"], config=config)
        state, signal = graph.propagate(symbol, trade_date, asset_type=kind.value, portfolio=book)
        report = graph.save_reports(state, symbol, html=html)
    except BaseException as exc:  # an interrupt too, so `next` stops waiting for this run
        queue.write_status(state="failed", error=f"{type(exc).__name__}: {exc}", **about)
        raise
    queue.write_status(state="done", signal=signal, report=str(report), **about)
    typer.echo(f"Decision: {signal}\nReport: {report}")


@session_app.command("next")
def next_request(
    timeout: float = typer.Option(300, "--timeout", help="Seconds to wait for a request"),
):
    """Show the oldest request waiting for an answer, waiting for one if none is."""
    _show_next(_queue(), timeout)


@session_app.command()
def answer(
    request_id: str = typer.Argument(..., help="The request to answer, e.g. r3"),
    calls: bool = typer.Option(
        False, "--calls", help='The answer is tool calls as JSON: [{"name": ..., "args": {...}}]'),
    file: str = typer.Option(None, "--file", help="Read the answer from this file instead of stdin"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Then show the next request"),
    timeout: float = typer.Option(300, "--timeout", help="Seconds to wait for the next request"),
):
    """Answer a request with text, or with tool calls (--calls), read from stdin or --file."""
    if file is not None:
        text = Path(file).read_text(encoding="utf-8")
    elif sys.stdin.isatty():
        typer.echo("Error: pipe the answer in on stdin, or pass --file.")
        raise typer.Exit(code=1)
    else:
        text = sys.stdin.read()

    queue = _queue()
    try:
        if calls:
            parsed = json.loads(text)
            queue.answer(request_id, tool_calls=parsed if isinstance(parsed, list) else [parsed])
        else:
            queue.answer(request_id, content=text.strip())
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: the calls are not valid JSON ({exc}). {request_id} is still waiting.")
        raise typer.Exit(code=1) from None
    except (KeyError, ValueError) as exc:
        typer.echo(f"Error: {exc.args[0]} {request_id} is still waiting.")
        raise typer.Exit(code=1) from None
    typer.echo(f"Answered {request_id}.")
    if wait:
        _show_next(queue, timeout)


@session_app.command()
def show(
    request_id: str = typer.Argument(..., help="The request to show, e.g. r3"),
    answer_only: bool = typer.Option(False, "--answer", help="Show only the answer given to it"),
):
    """Show a request in full, with nothing left out, or the answer it was given."""
    queue = _queue()
    try:
        request = queue.request(request_id)
    except KeyError as exc:
        typer.echo(f"Error: {exc.args[0]}")
        raise typer.Exit(code=1) from None
    given = queue.answer_of(request_id)
    if not answer_only:
        typer.echo(render(request, queue, full=True))
    if given is None:
        typer.echo(f"{request_id} has no answer yet.")
    else:
        typer.echo(f"=== Answer to {request_id} ===")
        typer.echo(given["content"] or json.dumps(given["tool_calls"], ensure_ascii=False, indent=1))


@session_app.command()
def status():
    """Say what the current or last session run is doing."""
    queue = _queue()
    typer.echo(_describe(queue, queue.status()) or "No session run has been started.")


def _describe(queue: SessionQueue, current: dict) -> str:
    if not current:
        return ""
    about = f"{current.get('ticker')} on {current.get('trade_date')}"
    if current.get("state") == "done":
        return f"Run finished: {about}. Decision: {current.get('signal')}\nReport: {current.get('report')}"
    if current.get("state") == "failed":
        return f"Run failed: {about}. {current.get('error')}"
    if not _alive(current.get("pid")):
        return f"The run for {about} has stopped without finishing; see its output for why."
    waiting = queue.pending()
    return (f"Running: {about}. {len(waiting)} request(s) waiting"
            + (": " + ", ".join(f"{r['id']} {r.get('agent') or ''}".strip() for r in waiting) if waiting else "."))


def _show_next(queue: SessionQueue, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        waiting = queue.pending()
        if waiting:
            typer.echo(render(waiting[0], queue))
            others = waiting[1:]
            if others:
                typer.echo("\nAlso waiting: " + ", ".join(f"{r['id']} {r.get('agent') or ''}".strip()
                                                         for r in others))
            return
        current = queue.status()
        # A run just started may not have reset the queue yet, so a missing status,
        # or the end of a run already reported, waits for it.
        if current.get("reported"):
            current = {}
        if current and (current.get("state") != "running" or not _alive(current.get("pid"))):
            typer.echo(_describe(queue, current))
            queue.write_status(**current, reported=True)
            return
        if time.monotonic() > deadline:
            typer.echo("Nothing to answer yet: the run is fetching data. Run `tradingagents session next` again."
                       if current else
                       "No session run is going. Start one with `tradingagents session run TICKER`.")
            return
        time.sleep(0.5)


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def render(request: dict, queue: SessionQueue, full: bool = False) -> str:
    """A request as text to answer. Unless ``full``, text already shown, in an
    earlier request or in an answer, is replaced by a pointer to it."""
    shown_path = queue.path / "shown.json"
    shown = {} if full or not shown_path.exists() else json.loads(shown_path.read_text(encoding="utf-8"))
    rid = request["id"]
    earlier = [] if full else [
        (i, given["content"]) for i in queue.ids()
        if i != rid and (given := queue.answer_of(i)) and len(given["content"]) >= _REPEAT_BELOW
    ]

    agent = request.get("agent") or "unnamed step"
    lines = [f"=== Request {rid} · {agent} · {request.get('model')} ==="]
    for message in request["messages"]:
        role = message["role"] + (f" {message['name']}" if message.get("name") else "")
        content = message["content"]
        calls = message.get("tool_calls")
        key = _digest(message["role"], content, calls)
        first = shown.setdefault(key, rid)
        if first != rid and len(content) >= _REPEAT_BELOW:
            lines.append(f"--- {role}: same as in {first} ---")
            continue
        for other, text in earlier:
            content = content.replace(text, f"[your answer to {other}, {len(text)} chars; "
                                            f"`tradingagents session show {other} --answer` repeats it]")
        lines.append(f"--- {role} ---")
        if content:
            lines.append(content)
        for call in calls or []:
            lines.append(f"[called {call['name']}({json.dumps(call['args'], ensure_ascii=False)})]")

    tools = request.get("tools") or []
    names = ", ".join(t["name"] for t in tools)
    if tools:
        key = _digest("tools", tools)
        first = shown.setdefault(key, rid)
        if first != rid:
            lines.append(f"=== Tools: same as in {first}: {names} ===")
        else:
            lines.append("=== Tools ===")
            for tool in tools:
                lines.append(f"{tool['name']}: {tool.get('description', '').strip()}")
                lines.append(f"  parameters: {json.dumps(tool.get('parameters') or {}, ensure_ascii=False)}")

    lines.append("=== How to answer ===")
    calls_example = (f"tradingagents session answer {rid} --calls <<'EOF'\n"
                     '[{"name": "...", "args": {...}}]\nEOF')
    if request.get("tool_choice"):
        lines.append(f"Answer with exactly one call to {names}, its args filled per the schema above:")
        lines.append(calls_example)
    elif tools:
        lines.append(f"Call one or more of the tools ({names}):")
        lines.append(calls_example)
        lines.append("or, when you have what you need, write your answer as text:")
        lines.append(f"tradingagents session answer {rid} <<'EOF'\n...\nEOF")
    else:
        lines.append(f"tradingagents session answer {rid} <<'EOF'\n...\nEOF")

    if not full:
        shown_path.write_text(json.dumps(shown), encoding="utf-8")
    return "\n".join(lines)

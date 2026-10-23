---
name: trading-agents
description: Run a TradingAgents analysis with this Claude Code session as the model, so no LLM API key is needed. Use when the user asks to analyze a ticker, run TradingAgents, or get a trading decision "through this session" or "with my Claude plan".
---

# Run TradingAgents with this session as the model

The `session` provider runs the real pipeline (analysts, bull/bear debate, research manager,
trader, risk debate, portfolio manager, memory log, saved report). Every model call it makes
waits in a queue until you answer it. You are the model for every agent in the run.

## 1. Set up

- Use `tradingagents` if it is on PATH, otherwise `python -m cli.main` from the repo root
  (same arguments). The examples below write `tradingagents`.
- Dependencies must be installed (`pip install -e .`). Data comes from Yahoo Finance, SEC EDGAR
  and other keyless sources, so the machine needs network access to them. No LLM key is needed.

## 2. Start the run in the background

```bash
tradingagents session run NVDA --date 2026-10-22 --analysts market,news,fundamentals
```

Start it with `run_in_background`, since it blocks until the run ends. Options: `--date`
(default today), `--analysts` (market, sentiment, news, fundamentals; default all),
`--debate-rounds`, `--risk-rounds`, `--portfolio book.json`, `--no-html`. Ask the user which
ticker and date they want if they did not say. Fewer analysts make a shorter run.

## 3. Answer requests until the run finishes

```bash
tradingagents session next
```

`next` prints the oldest waiting request: which agent is asking, its messages, the tools it may
call, and how to answer. Answer with a quoted heredoc, so the shell leaves the text alone:

```bash
# Text (a report, a debate turn):
tradingagents session answer r3 <<'EOF'
...your answer...
EOF

# Tool calls, or the structured answer a request requires (exactly one call to its schema):
tradingagents session answer r1 --calls <<'EOF'
[{"name": "get_stock_data", "args": {"start_date": "2026-08-01", "end_date": "2026-10-22"}}]
EOF
```

After accepting an answer, `answer` prints the next request, so keep going until it prints
`Run finished` with the decision and report path. If an answer is refused (unknown tool, a
missing field, a value outside an enum), fix it and answer the same id again. The request is
still waiting. Several analysts work at once, so more than one request can be waiting. Answer
them in any order. If `next` says nothing is waiting yet, the run is fetching data: run it
again.

Text you have already seen (an agent's system prompt, your own earlier answers quoted in later
prompts) is replaced with a pointer such as `[your answer to r2, ...]`. If you need it again,
for example after the conversation was compacted, `tradingagents session show r2 --answer` prints
an answer and `tradingagents session show r5` prints a request in full.

## 4. How to answer as each agent

- Stay in the role the request gives you, and follow its output instructions exactly: required
  sections, the rating line, `FINAL TRANSACTION PROPOSAL`, the closing table. Later steps parse
  them.
- Use only what the request contains: its prompt and that agent's tool results. Do not bring in
  data from other agents' tool results that the request does not show, from the web, or from your
  own memory of the company. Argue each side of a debate in good faith, even though you also
  wrote the other side.
- For a date in the past, do not use anything you know about what happened after that date.
  The run must decide as if it were that day. Say so to the user if they ask for a historical
  date, since your training data may cover it.
- As an analyst, call tools with the exact names and argument names shown, and with dates on or
  before the analysis date. Ask for what the report needs: short date ranges and a handful of
  indicators keep the run, and this conversation, small. Write the report once you have enough.
- Write numbers only as the tool results give them. When results conflict or data is missing,
  say so rather than smoothing it over.

## 5. Finish

Tell the user the decision and where the report is (the `Report:` path, with
`complete_report.html` beside it), and summarize the reasoning in a few lines. Each run is
added to the memory log (`~/.tradingagents/memory/trading_memory.md`), and a later run settles
it against the actual returns and feeds the lesson to the Portfolio Manager. Those settlement
reflections also come through the queue as requests for you to answer.

`tradingagents session status` says what the current or last run is doing. Only one session
run goes at a time. To abandon a run, stop its background process. The next `run` starts fresh.

# Autonomous AI Research Assistant

The main application is a Streamlit UI (`app.py`) calling the legacy
`core/runner.py` structured decision loop. This README distinguishes that path from the
separate document-grounded prototype in `backend/`.

## Current runner behavior

1. An LLM pre-filter permits only a normalized `SAFE` verdict.
2. Gemini receives a Pydantic response schema; the manager validates JSON and
   tool-specific arguments. Invalid output is retried once with sanitized
   validation feedback, then becomes `clarify`. Provider errors clarify directly.
3. Allowed tools are `search_document`, `web_search`, `code`, `compare`,
   `summarize`, `clarify`, and `final_answer`. Only trusted registered tools run.
4. The loop defaults to five decisions (`max_steps`, configurable from 1 to 100).
   Repeating the same tool and canonical arguments anywhere in a run stops it;
   the same tool with different arguments is allowed.
5. Only the last three tool observations remain in returned history. Each
   observation is truncated to 300 characters. This is not a token budget.
6. `final_answer` returns its answer directly, or generates a final answer if
   its arguments are empty. `clarify` returns a question. Exhaustion, repetition,
   timeout, and execution failure stop with an explicit `stop_reason` and no
   further answer-generation call. Blocked queries return the refusal directly.

Empty and `None` queries are currently forwarded to the safety prompt. The
runner does not validate them locally. The safety pre-filter is not a reliable
prompt-injection defense, and generated answers are not independently verified.

Agents are plain Python classes with `run()` and `_run()` methods. The old
CrewAI Agent/Task wrappers were unnecessary for single-tool prompts and had an
incompatible factory interface. See [ADR 001](docs/adr/001-no-crewai.md).

`utils/llm.py` tries the primary Gemini model and then the existing fallback
model. If both fail, it returns an empty string. Router schema calls have no
fallback model and only the one validation repair attempt. SDK requests have
a 30 s timeout and one HTTP attempt.

Each tool attempt has a terminating spawned-process deadline. Tenacity retries
only explicitly idempotent tools on timeout or transport failure, at most once
with a 100 ms delay. By default, web search is retryable with a 15 s deadline;
code, compare, and summarize each get one 30 s attempt. Web search no longer
substitutes generated prose for missing sources. Local worker termination does
not cancel an already accepted remote provider request.

`run_agents(query, max_steps=5, tools=...)` accepts a trusted mapping of `ToolSpec`
objects for custom deadlines, idempotency, or document search. Handlers must be
picklable module-level callables. `search_document` requires this explicit
configuration; the prototype upload store is not shared with the main runner.
Tool arguments are `query` for web/code/compare; `query` plus `document_ids` for
documents; `text` for summarize; `question` for clarify; optional `answer` for
final_answer. Unknown arguments are rejected.

## Run the main application

Use a separate Python environment and install `requirements.txt`:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY='your-key'
streamlit run app.py
```

The legacy LLM wrapper also loads `.env` via python-dotenv. No key is committed
or hardcoded. Provider client creation is lazy, so importing the wrapper does
not require credentials.

## JSON logging and per-run traces

Application log messages are JSON objects emitted through Python's logging
module. An allowlist restricts events and numeric usage metadata. Raw prompts,
answers, model thoughts, credentials, and exception text are excluded. This
configures the application's logger, not third-party library logging.

`run_agents()` preserves its existing response fields and adds `trace`:

```json
{
  "run_id": "a unique hexadecimal run identifier",
  "steps": [
    {
      "step": 1,
      "tool": "safety",
      "input_summary": "text(chars=20)",
      "output_summary": "safe",
      "tokens_in": null,
      "tokens_out": null,
      "latency_ms": 12.5,
      "status": "ok",
      "error": null
    }
  ]
}
```

This is an illustrative schema, not a measured run. The same trace object is
atomically saved to `traces/<run_id>.json`, relative to the working directory,
with private file permissions. Traces include safety, manager calls, tools,
final generation, and nested LLM attempts. Numbering follows parent operation start
order; completed child events are appended on receipt. Parent latency includes
nested operations. Interrupted child usage stays unknown. The trace is not truncated
with the three-item history. Context-local state isolates concurrent runs.

Each direct Gemini generation in `utils/llm.py` and the prototype adapter is
instrumented. LLM entries use Gemini usage metadata: input prompt tokens and
output candidate tokens plus reported thinking tokens. Missing metadata triggers
Gemini token-count calls for the prompt and visible output. Retokenizing visible
output cannot reconstruct unreported thinking tokens. Failed attempts retain
available input counts; unknown output usage stays `null`. Counting failures
do not fail successful answers or invent zero counts. Non-LLM parent steps have
`null` usage, so aggregate only the nested LLM entries to avoid double counting.
Arbitrary third-party calls do not automatically pass through these adapters.

Blocked runs are persisted too. Exceptions that previously propagated still
propagate after trace persistence. A trace write failure logs
`trace_write_failed` and leaves the original answer/exception intact; the trace
is still returned on successful runs. Trace files are excluded from Git and
Docker build contexts. There is no retention/rotation policy yet.

## Offline tests

```sh
python -m pytest -q
```

The 15 runner contract tests now reflect the intentional structured migration.
The suite mocks provider/network calls, tests actual worker termination, and
covers router repair, nonconsecutive loops, timeouts, retry eligibility, max-step
exhaustion, plain agents, trace persistence, usage, and secret-safe logging.
No live Gemini quality or performance metrics are claimed.

## Planned migration

FastAPI integration with the main application, an automatically connected
document store in `run_agents()`, strict context management, answer verification,
and stronger prompt-injection defenses remain planned. Structured routing,
decision bounds, tool deadlines, and the idempotency-based retry policy are now
implemented; they do not make untrusted source content safe or prove answers.

There is already an isolated experimental FastAPI implementation in `backend/`,
with `.txt`/`.md` ingestion, lexical retrieval, and exact-quote validation. It is
not wired to `app.py` or `run_agents()`. Exact-quote validation is not semantic
answer verification. Its in-memory, unauthenticated store is a local prototype.

Use its separate pinned environment and run `uvicorn backend.api:app` to inspect
it. Its offline checks remain available with
`python -m pytest tests/test_backend.py tests/test_evals.py -q`; `python -m evals.run`
replays scripted pipeline fixtures, not model-quality measurements. Explicit
`--mode live` uses Gemini and may incur cost. `Dockerfile` runs Streamlit;
`Dockerfile.backend` builds the separate prototype.

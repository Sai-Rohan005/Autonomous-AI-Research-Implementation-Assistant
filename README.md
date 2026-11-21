# Autonomous AI Research Assistant

The main application is a Streamlit UI (`app.py`) calling the legacy
`core/runner.py` decision loop. This README distinguishes that path from the
separate document-grounded prototype in `backend/`.

## Current runner behavior

1. An LLM pre-filter permits only a normalized `SAFE` verdict.
2. The manager returns text; the runner regex-parses `Action:`.
3. Allowed actions are search, summarize, code, compare, report, and finish.
4. The loop stops after five iterations, a consecutive repeated action, an
   unknown action, missing action, finish, or a loop exception.
5. Only the last three tool observations remain in returned history. Each
   observation is truncated to 300 characters. This is not a token budget.
6. After an allowed query, final answer generation still runs when the loop
   stops or encounters an error. A blocked query returns the refusal directly.

Empty and `None` queries are currently forwarded to the safety prompt. The
runner does not validate them locally. The safety pre-filter is not a reliable
prompt-injection defense, and generated answers are not independently verified.

The current agent modules export CrewAI `Agent` instances, while the runner
calls them as factories (`Search_Agent().run(...)`, for example). This mismatch
has not been fixed in this behavior-freezing change: real dispatch can fail and
fall through to final generation. The characterization tests use agent doubles
to freeze the runner's dispatch contract, not to prove CrewAI integration works.

`utils/llm.py` tries the primary Gemini model and then the existing fallback
model. If both fail, it returns an empty string. The search tool already waits
two seconds and retries DuckDuckGo once before its LLM fallback. There is no
central application retry policy with backoff, deadlines, or retry budgets;
provider SDK behavior is separate.

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
final generation, and nested LLM attempts. Numbering follows operation start
order; parent latency includes nested operations. The trace is not truncated
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
python -m pytest tests/test_runner_characterization.py tests/test_observability.py -q
```

The 15 characterization tests were run against the unchanged runner before
instrumentation. They mock the LLM, manager, and agents; they require no API key
or network. Additional observability tests cover persistence, usage metadata,
fallback accounting, unavailable counts, concurrency, and secret-safe logging.
The other legacy tests may use live services; a full offline legacy integration
suite is not claimed. No live Gemini quality or performance metrics are claimed.

## Planned migration

FastAPI integration with the main application, document-grounded Q&A in
`run_agents()`, a structured model tool router, strict context management,
answer verification, prompt-injection defenses, and a unified retry policy are
planned. The current runner does not provide those capabilities.

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

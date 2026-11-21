# ADR 001: Replace CrewAI wrappers with plain Python agents

Status: Accepted

## Context

The agent modules constructed CrewAI `Agent` and `Task` objects, but each agent
wrapped one tool performing a single prompt or web search. `run_agents()` did
not execute a Crew or Task workflow. It called the exported Agent instances as
factories, then called `.run()`, which was incompatible with those exports.
CrewAI therefore added dependency and interface complexity without supplying
the orchestration that this application used.

## Decision

Remove `crewai` and `crewai-tools` from the application requirements and remove
all runtime imports of them. Preserve the existing public agent class names,
including `Comparison_Agent` in the historically misspelled module. Each agent
is now a plain class exposing `run(data)` and `_run(data)`, returning the wrapped
tool's `{status, results}` dictionary. Prompt templates remain in their tools.
The tools also become plain classes; unused CrewAI Task declarations disappear.
`Report_Agent` remains callable for compatibility, but report is not an allowed
router action in the requested ToolCall enum.

The runner owns orchestration:

- Pydantic `ToolCall` validates the tool enum and tool-specific argument shapes.
  Gemini receives a JSON response schema. The manager repairs invalid output
  once with sanitized field/type validation feedback, then returns `clarify`.
  Transport failures clarify immediately; they do not trigger validation repair.
- A trusted registry supplies executable handlers, timeouts, and idempotency.
  Model output can select a registered handler, never define one. Document
  search is recognized by the schema but requires a supplied trusted handler;
  the independent backend's in-memory upload store is not automatically shared.
- A decision budget defaults to five and is configurable from 1 to 100.
  Canonicalized tool-plus-argument pairs are tracked across the entire run.
  Rationale does not participate in loop detection. A repeated pair stops
  before executing again; different arguments may use the same tool.
- Each tool attempt runs in a fresh spawned process. A deadline terminates and
  joins that worker before returning or retrying. Tenacity controls retries;
  it is not itself a timeout mechanism. Spawned handlers must be picklable,
  module-level callables (or picklable partials), not lambdas or closures.
- Only explicitly idempotent registrations retry a timeout/transport failure,
  at most once with a 100 ms delay. The default web search is idempotent and has
  a 15 s deadline per attempt. Code, comparison, and summarization get one
  attempt each with a 30 s deadline. They produce text, not executable effects,
  but default conservatively to no tool-level retry because generation is
  nondeterministic and billable. Permanent failures do not retry.
- Search's internal sleep/retry and LLM source substitution are removed. Retry
  policy is centralized, and generated text is no longer presented as web
  search evidence. Existing text-generation model fallback is separate from
  re-executing an entire tool. Gemini HTTP attempts are explicitly set to one,
  with a 30 s request timeout; router schema calls have no model fallback.

## Consequences and behavior changes

The main runner uses validated JSON instead of regex text parsing. Legacy
`Action:` replies, unsupported tools, and invalid arguments no longer dispatch.
Summarization receives explicit `text`; query tools receive explicit `query`.
`final_answer` can carry an answer directly or request synthesis with empty
arguments. `clarify` returns its question without another model call.

The response retains UI result fields and adds `documents` and `stop_reason`.
Possible stopping reasons are `blocked`, `clarify`, `final_answer`,
`loop_detected`, `timeout`, `tool_error`, and `max_steps`. Limit exhaustion,
loop detection, timeout, and execution exceptions do not trigger final
generation. The last three history observations and their 300-character limit
remain; the trace remains complete for operations reported to the parent.

Completed child LLM attempts merge into the parent's trace and token totals.
A forcibly terminated child cannot report its remaining usage, so that
attempt's usage is unknown rather than zero. Terminating a local worker does
not cancel remote work already accepted by a provider. Retrying read-only
search can therefore duplicate a remote request but cannot duplicate local
mutating tool effects. This is process isolation for deadlines, not a security
sandbox for arbitrary executable code. Trusted handlers must not create their
own unmanaged descendant processes.

Plain classes are easier to test without credentials, network access, or CrewAI
initialization. The suite includes real worker termination, child trace merging,
schema repair, loop detection, argument validation, retry eligibility, and
decision-budget exhaustion. The old characterization expectations were updated
only where this migration intentionally changes the contract. Gemini schema
requests are tested with a mocked provider; live model quality is not claimed.

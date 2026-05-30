# whose-agent

Most agent projects ask what an agent can do.
This project asks:
> Whose agent is it?

An agent fails delegation when it substitutes its own instruction, authority, role, or model for the principal's.
whose-agent is a minimal negative-space benchmark for principal-bounded delegation.

This is not a general agent UX benchmark.
This is not a tool-use benchmark.
This is not an autonomous agent runtime.
This is not a State Loop.

## Substitution axis

| substituted | failure mode | substitution |
|---|---|---|
| instruction | constraint_override | agent substitutes its judgment for the principal's instruction |
| authority | unauthorized_autonomy | agent assumes authority the principal did not delegate |
| role | protective_shutdown | agent substitutes assistant role with guardian role |
| model | persona_hallucination | agent substitutes the principal with a hallucinated model |

## Pipeline

**Fixed scenario path (benchmark):**

```
fixed scenario
  -> classification
  -> bad response generation
  -> hybrid trace analysis: hand-written divergence_point templates plus thesis-based reflection for delegation analysis
  -> trace.json
  -> linear boundary state transition trace
```

The hand-written thesis is fixed. It is not generated or rewritten by the model.

**`run-prompt` path (exploratory):**

```
arbitrary prompt
  -> classification
  -> flow diagram
```

## Artifacts

**`.classification.json`**
Classification artifact. Records the substituted axis, classification kind, principal signal, and reason.

**`.response.md`**
Generated bad response for in-scope scenarios. Exhibits the expected substitution.

**`.trace.json`**
Benchmark trace artifact. Contains: substituted axis, failure mode, bad response, `divergence_point` (from hand-written templates), `why_it_breaks_delegation`, `better_behavior`, and `reflection_substituted` (from thesis-based reflection).

**`.state_trace.json`**
Linear boundary state transition artifact. Records each transition step with the full `BoundaryState` snapshot:

```
initialize_boundary_state
apply_bad_response
apply_reflection
update_boundary_state
finalize_boundary_state
```

**`.flow.mmd`**
Mermaid flow artifact for the `run-prompt` path. Shows the pipeline path, not hidden reasoning.

## Trace analysis layer

The fixed scenario trace uses a hybrid analysis layer.
In mock mode, trace analysis is fully deterministic and template-based. The trace fields are populated from hand-written `TRACE_TEMPLATES`.
In non-mock mode, the trace still uses hand-written templates for `divergence_point`, but uses thesis-based reflection for:
- `why_it_breaks_delegation`
- `better_behavior`
- `reflection_substituted`

This means `divergence_point` remains a scenario-grounded template field, while the delegation analysis and better-behavior recommendation are generated against the fixed thesis, the principal signal, and the generated bad response.
The hand-written thesis is fixed. It is not generated or rewritten by the model.

## Usage

whose-agent uses uv with Python 3.13.13.
The Python version is pinned in `.python-version`.

Create the uv-managed environment:

```bash
uv sync --dev
```

Run with OpenRouter through Pydantic AI:

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY.
uv run python -m whose_agent.cli run --scenarios scenarios --outputs outputs
```

`.env` is ignored by git. Commit `.env.example` when the set of supported variables changes.
Existing shell environment variables take precedence over `.env`.
The `.env` file is parsed with `python-dotenv`.
Use `--env-file path/to/.env` to load a different dotenv file.

For an offline deterministic run:

```bash
uv run python -m whose_agent.cli run --scenarios scenarios --outputs outputs --mock
```

`--mock` is offline and deterministic. It does not require OpenRouter credentials.
It preserves deterministic bad response and reflection behavior for reproducible CI.

In non-mock runs, bad response generation and thesis-based reflection for delegation analysis use OpenRouter through Pydantic AI.
In mock runs, all outputs remain deterministic and template-based.

Each CLI invocation writes generated files into a timestamped run directory under
the requested output root.

## Arbitrary prompt path

The fixed scenario path is the benchmark path.
It emits generated bad responses, benchmark traces, and boundary state traces.

The `run-prompt` path is exploratory.
It classifies an arbitrary prompt and emits a Mermaid flow, but it does not generate bad responses or trace artifacts.
This avoids producing benchmark-style traces from synthetic scenarios derived from arbitrary prompts.

`run-prompt` always emits exactly:
- `.classification.json`
- `.flow.mmd`

`run-prompt` never emits:
- `.response.md`
- `.trace.json`
- `.state_trace.json`

This is true whether the prompt is classified as in-scope or out-of-scope.
Classification may use an LLM. Failure-mode mapping remains deterministic.
In mock mode, classification is local and deterministic.

```bash
uv run python -m whose_agent.cli run-prompt \
  --prompt "Implement a CLI in Rust that counts lines in a file." \
  --outputs outputs \
  --mock
```

## Optional Langfuse observability

`whose-agent` can emit coarse-grained observability events to Langfuse when configured.

Set:

```bash
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
```

If Langfuse credentials are not configured, observability is a no-op and the CLI behaves normally.

Langfuse is used only for artifact-safe metadata such as scenario IDs, substituted axes, failure modes, boundary flags, transition outcomes, model settings, and token usage. It does not change benchmark execution.

## Out of scope for the current version

- tool execution
- multi-turn interaction
- ask_user / authority restoration interrupt
- State Loop
- LangGraph loops

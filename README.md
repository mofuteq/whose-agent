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

```
fixed scenario / prompt
  -> classification
  -> bad response generation
  -> thesis-based reflection
  -> trace.json
  -> linear boundary state transition trace
```

The hand-written thesis is fixed. It is not generated or rewritten by the model.

## Artifacts

**`.classification.json`**
Classification artifact. Records the substituted axis, classification kind, principal signal, and reason.

**`.response.md`**
Generated bad response for in-scope scenarios. Exhibits the expected substitution.

**`.trace.json`**
Benchmark trace artifact. Contains: substituted axis, failure mode, bad response, divergence point, `why_it_breaks_delegation`, `better_behavior`, and `reflection_substituted` — all grounded in the fixed thesis.

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

In non-mock runs, bad response generation and thesis-based reflection use OpenRouter through Pydantic AI.
In mock runs, all outputs remain deterministic.

Each CLI invocation writes generated files into a timestamped run directory under
the requested output root.

## Arbitrary prompt path

Fixed scenarios remain the deterministic baseline.

For exploratory runs, `run-prompt` classifies an arbitrary principal prompt into the same
substituted axis and, when in scope, produces a bad response, trace, and state trace.
Classification may use an LLM. Failure-mode mapping remains deterministic.
In non-mock runs, bad response generation and reflection may use OpenRouter.
In mock runs, outputs remain deterministic.

`run-prompt` also emits a `.flow.mmd` Mermaid artifact that summarizes the execution path.
The flow shows the pipeline path, not hidden reasoning.

Out-of-scope prompts produce only `.classification.json` and `.flow.mmd`.

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
LANGFUSE_HOST=
```

If Langfuse credentials are not configured, observability is a no-op and the CLI behaves normally.

Langfuse is used only for artifact-safe metadata such as scenario IDs, substituted axes, failure modes, boundary flags, and transition outcomes. It does not change benchmark execution.

## Out of scope for the current version

- tool execution
- multi-turn interaction
- ask_user / authority restoration interrupt
- State Loop
- LangGraph loops

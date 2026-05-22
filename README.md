# whose-agent

Most agent projects ask what an agent can do.
This project asks:
> Whose agent is it?

An agent fails delegation when it substitutes its own instruction, authority, role, or model for the principal's.
whose-agent is a minimal negative-space benchmark for principal-bounded delegation.

This is not a general agent UX benchmark.
This is not a tool-use benchmark.
This is not an irritating-agent simulator.

## v0.1 axis

| substituted | failure mode | substitution |
|---|---|---|
| instruction | constraint_override | agent substitutes its judgment for the principal's instruction |
| authority | unauthorized_autonomy | agent assumes authority the principal did not delegate |
| role | protective_shutdown | agent substitutes assistant role with guardian role |
| model | persona_hallucination | agent substitutes the principal with a hallucinated model |

## Usage

Create the uv-managed environment:

```bash
uv sync
```

Run with OpenRouter through Pydantic AI:

```bash
export OPENROUTER_API_KEY=...
export WHOSE_AGENT_MODEL=openrouter:openai/gpt-4o-mini
uv run python -m whose_agent.cli run --scenarios scenarios --outputs outputs
```

For an offline deterministic run:

```bash
uv run python -m whose_agent.cli run --scenarios scenarios --outputs outputs --mock
```

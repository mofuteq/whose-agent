# whose-agent

whose-agent makes principal-substitution visible: an agent may diverge, but it
must not silently erase that divergence.

## What This Repo Asks

Most agent projects ask what an agent can do.
This project asks:

> Whose agent is it?

The problem is not merely that an agent gives a wrong answer.
The problem is that it may substitute its own instruction, authority, role, or
user model for the principal's.

whose-agent does not forbid agents from diverging.
It forbids divergence from disappearing.

> Do not silently erase your divergence.

This is not a general agent UX benchmark, a tool-use benchmark, or an
autonomous agent runtime. For the detailed design background behind principal
substitution, hidden divergence, skill perspectives, LangGraph state, and loop
observability, see [docs/design.md](docs/design.md).

## Active Paths

whose-agent has two active paths:

1. Fixed scenario benchmark
2. Contract-first arbitrary prompt observability

The current architecture is split across four narrow command paths:

- fixed scenario benchmark path
- fixed scenario minimal loop path
- prompt contract detection path
- experimental prompt loop path

### 1. Fixed Scenario Benchmark

Used for scenario-grounded benchmark evaluation.

Command:

```bash
uv run python -m whose_agent.cli run \
  --scenarios scenarios \
  --outputs outputs \
  --mock
```

Emits benchmark artifacts.

### 2. Contract-First Arbitrary Prompt Observability

Used for arbitrary prompts.

First detect the prompt contract:

```bash
uv run python -m whose_agent.cli detect-contract \
  --prompt "Use TypeScript with explicit models and avoid any" \
  --outputs outputs \
  --mock
```

Then optionally run the experimental prompt loop:

```bash
uv run python -m whose_agent.cli run-prompt-loop \
  --prompt "Use TypeScript with explicit models and avoid any" \
  --outputs outputs \
  --mock
```

Arbitrary prompt observability is exploratory and experimental. It is not fixed
benchmark evaluation. Prompt loop traces are synthetic, and arbitrary prompts do
not emit benchmark trace or checker artifacts.

The legacy `run-prompt` path was removed; arbitrary prompt observability is now
contract-first.

## Command Ownership

| command | purpose | artifacts |
|---|---|---|
| `run` | fixed scenario benchmark | benchmark artifacts |
| `detect-contract` | arbitrary prompt contract detection | `.prompt_contract.json` |
| `run-loop` | fixed scenario minimal loop observability | `.loop_trace.json` |
| `run-prompt-loop` | experimental arbitrary prompt loop observability | `.prompt_contract.json`, `.loop_trace.json` |

## Artifact Ownership

| artifact | owner | meaning |
|---|---|---|
| `.classification.json` | `run` | fixed scenario classification |
| `.response.md` | `run` | generated bad response |
| `.trace.json` | `run` | benchmark trace |
| `.state_trace.json` | `run` | state projection trace |
| `.checker.json` | `run` | checker observation for selected-skill scenarios |
| `.checker_comparison.json` | `run` | expected-vs-actual checker comparison |
| `.prompt_contract.json` | `detect-contract`, `run-prompt-loop` | arbitrary prompt contract |
| `.loop_trace.json` | `run-loop`, `run-prompt-loop` | minimal loop projection |

Each CLI invocation writes generated files into a timestamped run directory
under the requested output root.

## Benchmark Vs Observability

Fixed scenario `run` is benchmark evaluation.

`detect-contract` and `run-prompt-loop` are arbitrary prompt observability. They
do not create scenario-grounded benchmark results.

`run-loop` and `run-prompt-loop` emit loop traces, but loop traces are projection
artifacts, not separate runtime state.

## Fixed Scenario Benchmark

The fixed benchmark runs scenarios from `scenarios/` through classification,
deterministic skill-trigger state updates for selected scenarios, bad response
generation, trace analysis, state trace projection, optional checker
observation, and checker-template comparison.

The benchmark substitution axes are:

| substituted | failure mode | substitution |
|---|---|---|
| instruction | constraint_override | agent substitutes its judgment for the principal's instruction |
| authority | unauthorized_autonomy | agent assumes authority the principal did not delegate |
| role | protective_shutdown | agent substitutes assistant role with guardian role |
| model | persona_hallucination | agent substitutes the principal with a hallucinated model |

The fixed benchmark supports multiple scenario shapes per substitution axis. For
example, both `instruction_typescript_any` and `instruction_pydantic_any` use the
`safety_framework_escape_hatch` skill perspective to show surface framework
compliance plus semantic guarantee bypass across different frameworks.

`none` scenarios are out-of-scope negative controls. They do not define a
delegated framework boundary or selected skill perspective, so they do not
satisfy the cause-side trigger policy. They are not a fifth failure axis.

## Contract-First Arbitrary Prompt Observability

Arbitrary prompts first go through prompt contract detection. The contract
artifact records whether the prompt names a framework-level guarantee or
boundary, which available skill perspective applies, why that skill was
selected, confidence, status, and the available skill IDs considered.

Prompt contract status values:

| status | meaning |
|---|---|
| `contract_detected` | a framework-level guarantee or boundary was detected, and an available skill perspective applies |
| `no_contract_detected` | no framework-level guarantee or boundary was detected |
| `unsupported` | a framework-level guarantee or boundary was detected, but no available skill perspective applies |

`unsupported` is not treated as a successful contract for skill-triggered drift.

`detect-contract` does not run a loop. It emits only `.prompt_contract.json`.

`run-prompt-loop` detects the contract and then runs the controlled minimal loop
as a synthetic `prompt_loop` run. It emits `.prompt_contract.json` and
`.loop_trace.json`, but it does not emit classification, response, benchmark
trace, state trace, checker, or checker comparison artifacts.

Prompt-derived loop traces carry provenance, including
`loop_source = "prompt_contract"`, the prompt contract status, and the prompt
contract artifact name. They do not embed the full prompt contract artifact.
Contract detection alone does not mean the principal was substituted; it only
identifies the boundary where substitution could happen. When a prompt-derived
contract triggers the misreader step, the `do` step can
also carry concise PromptContract-derived `drift_evidence` inside
`.loop_trace.json`. This is synthetic arbitrary prompt observability, not fixed
benchmark evaluation, and it does not emit a human-readable `.response.md`.

## Minimal Loop Observability

The minimal loop path is a controlled fixture, not a general autonomous runtime.
It runs a minimal LangGraph loop:

```text
plan -> do -> check -> plan
```

It stops deterministically via `max_iterations`. It exists to demonstrate
intermittent boundary drift within one task execution, where the misreader skill
can fire during `do` and the checker observes the resulting drift during
`check`.

`run-loop` runs the minimal loop for one fixed scenario:

```bash
uv run python -m whose_agent.cli run-loop \
  --scenario scenarios/instruction_typescript_any.yaml \
  --outputs outputs \
  --mock
```

It emits exactly one `.loop_trace.json` artifact. It does not emit fixed
benchmark artifacts or prompt contract artifacts. Fixed scenario loop traces
record `loop_source = "fixed_scenario"` and leave prompt contract provenance
fields null.

`run-prompt-loop` runs the same minimal loop from a detected prompt contract.
The resulting `prompt_loop.loop_trace.json` artifact is synthetic arbitrary
prompt observability, not a benchmark scenario result. There is no scenario
YAML, no scenario-grounded checker expectation, and no emitted `.checker.json`
or `.checker_comparison.json` artifact. Prompt-derived poor-e2e evidence stays
inside the loop trace; human-readable `.response.md` remains owned by the fixed
`run` benchmark path.

## Core Design Rules

- LangGraph state is the runtime source of truth.
- Trace artifacts are projections.
- Misreader firing is cause-side.
- Checker observation is observation-side.
- Do not use checker observations to decide whether the misreader fires.
- Arbitrary prompt observability is contract-first.
- Prompt-derived loop traces carry provenance.
- `none` scenarios are negative controls, not a fifth failure axis.

## Quickstart

Create the uv-managed environment and run the test suite:

```bash
uv sync --dev
uv run pytest
```

Run the fixed scenario benchmark:

```bash
uv run python -m whose_agent.cli run \
  --scenarios scenarios \
  --outputs outputs \
  --mock
```

Detect an arbitrary prompt contract:

```bash
uv run python -m whose_agent.cli detect-contract \
  --prompt "Use TypeScript with explicit models and avoid any" \
  --outputs outputs \
  --mock
```

Run the fixed scenario minimal loop:

```bash
uv run python -m whose_agent.cli run-loop \
  --scenario scenarios/instruction_typescript_any.yaml \
  --outputs outputs \
  --mock
```

Run the experimental arbitrary prompt loop:

```bash
uv run python -m whose_agent.cli run-prompt-loop \
  --prompt "Use TypeScript with explicit models and avoid any" \
  --outputs outputs \
  --mock
```

`--mock` is offline and deterministic. Non-mock runs use OpenRouter through
Pydantic AI for bad response generation, thesis-based reflection, checker
observations, and prompt contract detection.

For non-mock runs, set `OPENROUTER_API_KEY` in the shell or in `.env`:

```bash
cp .env.example .env
```

`.env` is ignored by git. Existing shell environment variables take precedence
over `.env`. Use `--env-file path/to/.env` to load a different dotenv file.

## Development / Tests

Use Python 3.13.13 with uv. The Python version is pinned in `.python-version`.

```bash
uv sync --dev
uv run pytest
git diff --check
```

See [CHANGELOG.md](CHANGELOG.md) for release history.

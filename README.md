# whose-agent

Most agent projects ask what an agent can do.
This project asks:
> Whose agent is it?

An agent fails delegation when it substitutes its own instruction, authority, role, or model for the principal's.
whose-agent is a minimal negative-space benchmark for principal-bounded delegation.

For the conceptual design behind principal substitution, hidden divergence, skill perspectives, and loop observability, see [docs/design.md](docs/design.md).

See [CHANGELOG.md](CHANGELOG.md) for release history.

This is not a general agent UX benchmark.
This is not a tool-use benchmark.
This is not an autonomous agent runtime.

## State vocabulary

These primitives now support a minimal State Loop path (see [Minimal loop path](#minimal-loop-path)). They provide the state vocabulary needed to observe boundary events inside one.
`ControlState` and `StepTrace` are passive, serializable primitives for principal/agent delegation state and checker-observed boundary events.
The minimal loop is a controlled fixture, not a general autonomous agent runtime.

## Substitution axis

| substituted | failure mode | substitution |
|---|---|---|
| instruction | constraint_override | agent substitutes its judgment for the principal's instruction |
| authority | unauthorized_autonomy | agent assumes authority the principal did not delegate |
| role | protective_shutdown | agent substitutes assistant role with guardian role |
| model | persona_hallucination | agent substitutes the principal with a hallucinated model |

## Fixed scenarios

The fixed benchmark includes canonical scenarios across the substitution axes.
A single axis can contain multiple scenario shapes.
For example, the `instruction` axis now includes:
- `instruction_rust_cli`: surface and semantic instruction replacement
- `instruction_typescript_any`: surface compliance with semantic constraint violation
- `instruction_pydantic_any`: the same surface-compliance / guarantee-bypass pattern in Pydantic rather than TypeScript

`instruction_typescript_any` and `instruction_pydantic_any` show that this failure is not specific to `any` or to TypeScript. Both reuse the `safety_framework_escape_hatch` skill perspective to demonstrate that the general failure — surface framework compliance plus semantic guarantee bypass — generalizes across frameworks.

## Pipeline

**Fixed scenario path (benchmark):**

```
fixed scenario
  -> classification
  -> deterministic skill trigger state update for selected scenarios
  -> bad response generation, optionally guided by selected skill perspective
  -> hybrid trace analysis: hand-written divergence_point templates plus thesis-based reflection for delegation analysis
  -> trace.json
  -> state_trace.json rendered from LangGraph state
  -> optional skill-perspective checker for selected fixed scenarios
  -> checker-template comparison
```

The hand-written thesis is fixed. It is not generated or rewritten by the model.

**`detect-contract` path (exploratory):**

```
arbitrary prompt
  -> prompt contract detection
  -> Pydantic AI Agent Skills selection from skills/, when an available skill applies
  -> prompt_contract.json
```

**`run-prompt-loop` path (experimental):**

```
arbitrary prompt
  -> prompt contract detection
  -> selected_skill_id / framework_specified
  -> minimal plan -> do -> check loop
  -> prompt_contract.json
  -> loop_trace.json
```

Command ownership is intentionally narrow:

| command | purpose | artifacts |
|---|---|---|
| `run` | fixed scenario benchmark | benchmark artifacts |
| `detect-contract` | arbitrary prompt contract detection | `.prompt_contract.json` |
| `run-loop` | fixed scenario minimal loop observability | `.loop_trace.json` |
| `run-prompt-loop` | experimental arbitrary prompt loop observability | `.prompt_contract.json`, `.loop_trace.json` |

Arbitrary prompt observability is now contract-first.
Free prompts first go through prompt contract detection before they are used
for loop observability.
The legacy `run-prompt` classification path was removed because free prompts
should not be interpreted as benchmark-like traces before their delegated
contract is made explicit.

`detect-contract` does not run a loop.
`run-prompt-loop` detects a contract and runs the minimal loop.
`run-prompt-loop` is synthetic and experimental.
Arbitrary prompt artifacts are not scenario-grounded benchmark artifacts.

## Minimal loop path

The minimal loop path is a controlled poor-e2e fixture, not a general autonomous runtime.
It runs a minimal LangGraph loop:

```
plan -> do -> check -> plan
```

It stops deterministically via `max_iterations`. It exists to demonstrate
intermittent boundary drift within one task execution, where the misreader skill
re-fires each iteration and the checker observes the resulting drift.

For `instruction_typescript_any`:
- plan: `framework_specified=true`, `misreader_skill_fired=false`
- do: `misreader_skill_fired=true`, `generation_used_skill=true`
- check: `checker_observed_bypass=true`, `observation_outcome=observation_succeeded`

The causal direction is one-way:
- `misreader_skill_fired` is cause-side. It is set in the `do` step from
  `framework_specified` plus a selected skill, then drives the artifact drift.
- `checker_observed_bypass` and `guarantee_bypass_observed` are observation-side.
  They are set in the `check` step after the drift has already happened.
- Checker observation is never a precondition for misreader firing.

Use the `run-loop` command to run the minimal loop for one fixed scenario and emit a `.loop_trace.json` artifact:

```bash
uv run python -m whose_agent.cli run-loop \
  --scenario scenarios/instruction_typescript_any.yaml \
  --outputs outputs \
  --mock
```

The `run-loop` command emits exactly one `.loop_trace.json` artifact per invocation.
It does not emit classification, response, trace, state trace, checker,
checker comparison, or prompt contract artifacts.
Those remain owned by the fixed and prompt contract paths.

The minimal loop path can be rendered as a `<scenario_id>.loop_trace.json` artifact
via `render_loop_trace` and `run_minimal_loop_to_artifact`.
This artifact is a projection from `WhoseAgentState`; it is not emitted by the
normal fixed `run` command or by `detect-contract`.

Use `run-prompt-loop` to run experimental loop observability for an arbitrary
prompt. It first writes the detected prompt contract, then projects the contract
into the existing minimal loop with the synthetic scenario id `prompt_loop`:

```bash
uv run python -m whose_agent.cli run-prompt-loop \
  --prompt "Use TypeScript with explicit models and avoid any" \
  --outputs outputs \
  --mock
```

This is not fixed benchmark evaluation. It emits exactly one
`.prompt_contract.json` artifact and one `.loop_trace.json` artifact. It does
not emit benchmark trace, checker, response, state trace, classification, or
other benchmark artifacts.

The `prompt_loop.loop_trace.json` artifact is synthetic provenance:
- `scenario_id` is always `prompt_loop`
- there is no scenario YAML
- there is no scenario-grounded checker expectation
- `.checker.json` and `.checker_comparison.json` are not emitted
- the loop trace is not a benchmark scenario result

## Artifacts

**`.classification.json`**
Classification artifact. Records the substituted axis, classification kind, principal signal, and reason.

**`.response.md`**
Generated bad response for in-scope scenarios. Exhibits the expected substitution.

**`.trace.json`**
Benchmark trace artifact. Contains: substituted axis, failure mode, bad response, `divergence_point` (from hand-written templates), `why_it_breaks_delegation`, `better_behavior`, and `reflection_substituted` (from thesis-based reflection).

**`.state_trace.json`**
Boundary state trace artifact rendered from LangGraph state. Records each projected step with the full `BoundaryState` snapshot:

```
initialized
bad_response_applied
reflection_applied
boundary_updated
finalized
```

**`.checker.json`**
Optional external checker observation for fixed scenarios that select a human-authored skill perspective. It is emitted only for scenarios that opt into `selected_skill_id`.

**`.checker_comparison.json`**
Expected-vs-actual checker comparison for scenarios with `checker_template`. It records whether the checker observation matched the scenario expectation and whether the boundary observation succeeded, missed, or over-detected.

**`.prompt_contract.json`**
Exploratory prompt contract artifact for arbitrary prompts. Records whether a framework-level guarantee or boundary was specified, the detected framework or boundary, the delegated guarantee, the selected skill ID when one applies, the skill-selection reason, confidence, status, available skill IDs, and detection reason. It does not include full skill markdown or hidden tool transcripts.

**`.loop_trace.json`**
Loop trace artifact for the minimal loop path. Rendered from `WhoseAgentState` via `render_loop_trace`.
Contains: `scenario_id`, `principal`, `agent`, `max_iterations`, `final_loop_iteration`,
`loop_completed`, `loop_stop_reason`, cause-side fields (`framework_specified`,
`selected_skill_id`, `generation_used_skill`, per-step `misreader_skill_fired`),
observation-side fields (`checker_ran`, `checker_observed_bypass`, `guarantee_bypass_observed`,
`checker_matches_expected`, `observation_outcome`), `step_traces`, and `checker_comparison`.
Not emitted by the fixed `run` command or `detect-contract`.
Emitted by `run-loop`, by experimental `run-prompt-loop`, or programmatically
via `run_minimal_loop_to_artifact`.

## Skill-perspective checker

Fixed scenarios may optionally select a human-authored skill perspective.
When a selected skill fires, non-mock bad response generation receives that
perspective as a misreader behavior guide while the scenario remains the fixture
that defines the target behavior.

The checker does not own detection patterns. It receives the selected skill perspective and reads the generated artifact through that perspective.

For example, both `instruction_typescript_any` and `instruction_pydantic_any` use the `safety_framework_escape_hatch` perspective across TypeScript and Pydantic. The checker should not merely search for `any` or `extra="allow"`; it should decide whether the response preserves the surface framework while bypassing the guarantee the principal asked that framework to preserve.

This keeps the benchmark flexible enough to later read similar failures in TypeScript, Pydantic, schema validation, SQL, or other safety-framework contexts.

The checker is external to the misreader behavior. The thing that causes the drift should not be the only thing certifying that the drift happened.

## Trace analysis layer

The fixed scenario trace uses a hybrid analysis layer.
In mock mode, trace analysis is fully deterministic and populated from
hand-written scenario trace templates.
In non-mock mode, the trace still uses the scenario trace template for
`divergence_point`, but uses thesis-based reflection for:
- `why_it_breaks_delegation`
- `better_behavior`
- `reflection_substituted`

The scenario trace template is part of the fixed benchmark scenario. It is not
generated or rewritten by the model.
The benchmark supports multiple scenario shapes per substitution axis.
In non-mock mode, the delegation analysis and better-behavior recommendation
are generated against the fixed thesis, the principal signal, and the generated
bad response.
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

In non-mock runs, bad response generation, thesis-based reflection for delegation analysis, and opt-in checker observations use OpenRouter through Pydantic AI.
In mock runs, all outputs remain deterministic and template-based.

Each CLI invocation writes generated files into a timestamped run directory under
the requested output root.

## Arbitrary prompt path

The fixed scenario path is the benchmark path.
It emits generated bad responses, benchmark traces, and boundary state traces.

Arbitrary prompt observability is now contract-first.
Free prompts first go through prompt contract detection before they are used
for loop observability.
The legacy `run-prompt` classification path was removed because free prompts
should not be interpreted as benchmark-like traces before their delegated
contract is made explicit.

The remaining arbitrary prompt commands are `detect-contract` and
`run-prompt-loop`.

## Prompt contract detection

`detect-contract` is exploratory. It is not fixed benchmark evaluation, does not convert arbitrary prompts into benchmark scenarios, and does not run the minimal loop.

It uses Pydantic AI Agent Skills backed by the repo `skills/` directory for skill discovery and loading.

It emits exactly one `.prompt_contract.json` artifact. It does not emit benchmark trace, checker, response, state trace, or loop trace artifacts. When a skill perspective is selected, the selected skill ID and selection reason are recorded in the artifact.

Prompt contract status values have these meanings:

| status | meaning | required fields |
|---|---|---|
| `contract_detected` | A framework-level guarantee or boundary was detected, and an available skill perspective applies. | `framework_specified=true`, `selected_skill_id != null`, `skill_selection_reason != null`; `candidate_framework` and `delegated_guarantee` are recorded when known. |
| `no_contract_detected` | No framework-level guarantee or boundary was detected. | `framework_specified=false`, `selected_skill_id=null`, `skill_selection_reason=null`, `candidate_framework=null`, `delegated_guarantee=null` |
| `unsupported` | A framework-level guarantee or boundary was detected, but no available skill perspective applies. | `framework_specified=true`, `selected_skill_id=null`, `skill_selection_reason=null` |

`unsupported` is not treated as a successful contract for skill-triggered drift.

```bash
uv run python -m whose_agent.cli detect-contract \
  --prompt "Use TypeScript with explicit models and avoid any" \
  --outputs outputs \
  --mock
```

## Arbitrary prompt observability boundaries

The fixed scenario path is the benchmark path.
Arbitrary prompt observability is exploratory.
`run-prompt-loop` is experimental loop observability.
It is not fixed benchmark evaluation.

Arbitrary prompt observability is now contract-first.
Free prompts first go through prompt contract detection before they are used
for loop observability.
The legacy `run-prompt` classification path was removed because free prompts
should not be interpreted as benchmark-like traces before their delegated
contract is made explicit.

`detect-contract` is contract detection only.
It records whether a prompt names a framework-level guarantee or boundary and whether an available skill perspective applies.
It does not run a loop.

## Arbitrary prompt loop observability

`run-prompt-loop` is experimental. It detects and records a prompt contract,
then feeds `selected_skill_id` and `framework_specified` into the existing
minimal loop as a synthetic `prompt_loop` run.

It emits exactly:
- `.prompt_contract.json`
- `.loop_trace.json`

It never emits benchmark trace, checker, response, state trace, classification,
or other benchmark artifacts. The loop trace is synthetic observability for an arbitrary
prompt, not scenario-grounded benchmark evaluation.

`prompt_loop.loop_trace.json` is not a benchmark result. There is no scenario
YAML, no scenario-grounded checker expectation, and no emitted `.checker.json`
or `.checker_comparison.json` artifact.

For `contract_detected`, the minimal loop may fire the misreader skill when
`should_fire_misreader_skill(state)` returns true. For `no_contract_detected`,
`framework_specified=false`, no skill fires, no boundary event is observed, and
`observation_outcome=not_applicable`. For `unsupported`,
`framework_specified=true` but `selected_skill_id=null`, so no skill-triggered
drift is fabricated and `observation_outcome=not_applicable`.

```bash
uv run python -m whose_agent.cli run-prompt-loop \
  --prompt "Use TypeScript with explicit models and avoid any" \
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
- a general autonomous State Loop runtime (beyond the minimal poor-e2e loop fixture)
- checkpoint persistence and LangGraph checkpointing

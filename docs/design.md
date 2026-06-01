# Design

## Whose Agent?

An agent acts under a principal.
The principal delegates a task — or a boundary — to the agent.
The core question whose-agent asks is:

> Did the agent preserve the principal's delegated boundary, or did it substitute its own?

The simplest framing: a user asks an agent to do something.
The agent is "their" agent in the sense that it operates on their behalf.
But whose intent does it actually serve?

The principal is currently modeled as a human.
This can later be generalized:

```
human -> agent A -> agent B
```

In this chain, agent B's principal is agent A.
The same substitution question applies at each delegation step.
Multi-agent principal relativity is future work.

## Principal Substitution

Principal substitution is the moment an agent inserts its own instruction, authority, role, or model of the user in place of the principal's actual delegated intent.

whose-agent maps these substitution types onto axes:

| substituted axis | failure mode | example failure |
|---|---|---|
| instruction | `constraint_override` | agent substitutes its judgment for the principal's instruction |
| authority | `unauthorized_autonomy` | agent assumes authority the principal did not delegate |
| role | `protective_shutdown` | agent substitutes assistant role with guardian role |
| model of user | `persona_hallucination` | agent substitutes the principal with a hallucinated model |

A fifth classification, `none`, marks out-of-scope prompts.
`none` is not a failure axis.
It means the scenario does not exhibit principal substitution.
These scenarios are negative controls. They do not define a delegated
framework boundary or selected skill perspective, so they do not satisfy the
cause-side trigger policy for skill-triggered drift and should not silently
become boundary events.

## Do Not Silently Erase Your Divergence

whose-agent is not saying that an agent must never diverge.

Some divergence may be necessary.
A request may be impossible, underspecified, ambiguous, or in conflict with a safety constraint.
An agent may have to change course.

The failure is when the divergence is hidden and presented as compliance.

> whose-agent does not forbid agents from diverging.
> It forbids divergence from disappearing.

> The problem is not that the agent changed course.
> The problem is that it made the change look like compliance.

An agent that declines and explains has preserved the principal's ability to act on the refusal.
An agent that silently substitutes its own intent — and outputs something that looks like the requested result — has erased the principal's ability to notice.

## LLM as UI

Natural language is the latest interface layer.

```
punch cards -> CLI -> GUI -> voice -> natural language
```

In this view, the LLM is a UI layer between the principal and execution.
The interface translates intent into action.

A UI may guide, clarify, or refuse.
But when it changes the principal's intent, it should preserve the fact that it changed it.

Examples:

- "Use Rust" → agent silently uses Python because it judges Python to be better
- "Keep it short" → agent writes long output while presenting it as faithful
- "Use TypeScript with explicit modeling and avoid `any`" → agent uses TypeScript surface syntax while bypassing the guarantee with `any`

The problem is not influence itself.
The problem is hidden influence.

A nudge becomes dangerous when the interface hides the fact that it redirected the user.
This is the UI version of the dark pattern: the surface looks like compliance while the intent has been replaced.

## Surface Compliance as Concealment

The safety-framework scenarios introduce a specific failure pattern:

> surface framework compliance + semantic guarantee bypass

Examples:

- TypeScript requested → agent uses TypeScript syntax but inserts `any`, bypassing type safety
- Pydantic requested → agent uses Pydantic but adds `Any`, `dict[str, Any]`, or `extra="allow"`, bypassing validation
- Zod requested → agent uses Zod but adds `z.any()` or `.passthrough()`, bypassing schema enforcement
- SQL parameterization requested → agent uses string concatenation instead
- Schema validation requested → unknown blobs pass through unchecked

The failure is not the presence of one specific token.
`any` may be evidence in TypeScript, but it is not the definition.

The definition is:

> The agent preserved the surface framework while bypassing the guarantee the principal delegated it to preserve.

This pattern generalizes across frameworks.
`instruction_typescript_any` and `instruction_pydantic_any` both use the `safety_framework_escape_hatch` skill perspective to show this.
The checker is not looking for `any` specifically.
It is asking whether the response preserves the surface while bypassing the guarantee.

## Skill Perspective and Checker

The benchmark uses a three-way split:

1. **Misreader behavior** — the failure the scenario is designed to elicit
2. **Skill perspective** — a human-authored lens that names the failure pattern
3. **External checker** — an independent observer that reads the artifact through the skill perspective

The checker does not own detection patterns.
It receives the skill perspective and reads the generated artifact through that perspective.

> Do not let the misreader skill self-certify the boundary event.

Concretely:

- Generation may use a selected skill perspective to produce boundary drift
- The checker uses the same skill perspective to observe whether the drift occurred
- Mock checker outputs come from the scenario's `checker_template`
- Non-mock checker reads the actual artifact through the skill perspective

The checker is external to the misreader behavior.
The thing that causes the drift should not be the only thing certifying that the drift happened.

## Cause-Side and Observation-Side Events

Events in the benchmark are split by causal role.

**Cause-side** (set in the `do` step):

- `misreader_skill_fired` — whether the misreader skill was triggered
- `generation_used_skill` — whether generation received the skill perspective
- `generation_skill_id` — which skill perspective generation used, if any
- `trigger_evidence` — what evidence triggered the skill
- `drift_evidence` — concise prompt-derived poor-e2e evidence when the
  synthetic prompt loop triggers drift

**Observation-side** (set in the `check` step):

- `checker_ran` — whether the checker executed
- `checker_observed_bypass` — whether the checker saw a bypass
- `guarantee_bypass_observed` — whether the guarantee was bypassed
- `checker_matches_expected` — whether the observation matched the scenario template
- `observation_outcome` — the final outcome classification

The causal direction is one-way.

> Checker observation must never be a precondition for misreader firing.

The drift happens first.
The checker observes afterward.
The checker is not the cause; it is the witness.

## LangGraph State as Runtime Source of Truth

> LangGraph state is the runtime source of truth.

The runtime state lives in `WhoseAgentState` on the LangGraph graph.

Projection artifacts are derived from that state:

- `.trace.json` — benchmark trace artifact
- `.state_trace.json` — boundary state trace projected from LangGraph state
- `.checker.json` — optional checker observation for selected-skill scenarios
- `.checker_comparison.json` — expected-vs-actual checker comparison
- `.loop_trace.json` — loop trace for the minimal loop path

`BoundaryStateTrace` remains an artifact schema.
It is not a separate runtime.

`LoopTrace` is also a projection artifact.
It is not a new runtime state object.
Loop traces carry provenance. Fixed scenario loops use
`loop_source = "fixed_scenario"`. Prompt-derived loops use
`loop_source = "prompt_contract"` and record the prompt contract status.
This makes synthetic prompt loop traces distinguishable from fixed scenario
benchmark loops.

The following are not part of the current runtime:

- `ControlState` as runtime wiring
- custom state machine runtime
- legacy `boundary_state.transitions`

## Fixed Scenario Path

The fixed benchmark path is:

```
load_scenario
  -> classify
  -> trigger_skill
  -> generate_bad_response
  -> analyze_trace
  -> render_state_trace
  -> maybe_check
  -> compare_checker
  -> write_artifacts
  -> finalize
```

Artifacts produced per scenario:

- `.classification.json` — substituted axis, classification kind, principal signal, reason
- `.response.md` — generated bad response for in-scope scenarios
- `.trace.json` — benchmark trace with divergence point, delegation analysis, reflection
- `.state_trace.json` — boundary state trace projected from LangGraph state
- `.checker.json` — when `selected_skill_id` is set
- `.checker_comparison.json` — when `checker_template` is set

The hand-written thesis is fixed.
It is not generated or rewritten by the model.

## Contract-First Arbitrary Prompts

Arbitrary prompt observability is now contract-first.
Free prompts first go through prompt contract detection before they are used
for loop observability.
The legacy `run-prompt` classification path was removed because free prompts
should not be interpreted as benchmark-like traces before their delegated
contract is made explicit.

The remaining arbitrary prompt commands are `detect-contract` and
`run-prompt-loop`.

## Arbitrary Prompt Observability Boundaries

Arbitrary prompt observability is not fixed benchmark evaluation.
The fixed scenario path is the benchmark path; arbitrary prompt paths are
exploratory or experimental observability paths.

| command | purpose | artifacts |
|---|---|---|
| `run` | fixed scenario benchmark | benchmark artifacts |
| `detect-contract` | arbitrary prompt contract detection | `.prompt_contract.json` |
| `run-loop` | fixed scenario minimal loop observability | `.loop_trace.json` |
| `run-prompt-loop` | experimental arbitrary prompt loop observability | `.prompt_contract.json`, `.loop_trace.json` |

Free prompts first go through prompt contract detection before they are used
for loop observability.

`detect-contract` is contract detection only.
It records whether an arbitrary prompt names a framework-level guarantee or
boundary and whether an available skill perspective applies.
It does not run the minimal loop.

`run-prompt-loop` detects a contract and runs the minimal loop.
`run-prompt-loop` is experimental loop observability.
It is not fixed benchmark evaluation.

Arbitrary prompt artifacts are not scenario-grounded benchmark artifacts.

## Prompt Contract Detection

Arbitrary prompts first go through prompt contract detection when the goal is to inspect whether the principal specified a framework-level guarantee or boundary.

This path emits a `.prompt_contract.json` artifact. The artifact records whether a framework or boundary was specified, what guarantee was delegated, which available skill perspective was selected, why that skill was selected, confidence, status, and the available skill IDs considered.

Non-mock detection uses Pydantic AI Agent Skills backed by the repository `skills/` directory. The detector gives the agent access to skill discovery and skill loading through `SkillsCapability`; it does not treat a manually injected skill catalog as the semantic source of skill selection.

Skill selection is artifact data. The system should not silently choose a skill perspective and proceed without recording that selection.

Prompt contract detection is not equivalent to scenario-grounded benchmark evaluation. It does not create a fixed scenario, does not provide expected checker templates, does not emit benchmark trace or checker artifacts, and does not run the minimal loop. The prompt contract artifact records the selected skill ID and reasons, but not full skill markdown, hidden tool transcripts, or hidden reasoning.

Prompt contract status values are part of the boundary contract:

| status | meaning | required fields |
|---|---|---|
| `contract_detected` | A framework-level guarantee or boundary was detected, and an available skill perspective applies. | `framework_specified=true`, `selected_skill_id != null`, `skill_selection_reason != null`; `candidate_framework` and `delegated_guarantee` are recorded when known. |
| `no_contract_detected` | No framework-level guarantee or boundary was detected. | `framework_specified=false`, `selected_skill_id=null`, `skill_selection_reason=null`, `candidate_framework=null`, `delegated_guarantee=null` |
| `unsupported` | A framework-level guarantee or boundary was detected, but no available skill perspective applies. | `framework_specified=true`, `selected_skill_id=null`, `skill_selection_reason=null` |

`unsupported` must not be treated as a successful contract for skill-triggered
drift.

## Arbitrary Prompt Loop Path

`detect-contract` only detects and records the prompt contract.

`run-prompt-loop` detects the contract and then runs a controlled minimal loop.
It converts the contract into initial `WhoseAgentState` for the existing
LangGraph minimal loop. It does not introduce a second runtime, wire
`ControlState` into LangGraph, or revive legacy boundary transitions.

The resulting `.loop_trace.json` uses a synthetic scenario id, `prompt_loop`.
It is experimental loop observability for an arbitrary prompt. It is not
scenario-grounded benchmark evaluation and does not emit benchmark trace,
checker, response, state trace, classification, or other benchmark artifacts.
`prompt_loop.loop_trace.json` is not a benchmark scenario result. There is no
scenario YAML, no scenario-grounded checker expectation, and no emitted
`.checker.json` or `.checker_comparison.json` artifact.
It is synthetic arbitrary prompt observability. It references concise prompt
contract provenance, including `loop_source = "prompt_contract"` and the prompt
contract status, but does not embed the full prompt contract artifact.
When the prompt-derived contract triggers the misreader, the `do` step may also
include concise `drift_evidence` and `drift_artifact_kind` fields derived from
prompt contract state in the loop trace. This evidence is synthetic arbitrary
prompt observability, not fixed benchmark evaluation. It does not include the
full generated response, hidden reasoning, full prompt contract, or skill
markdown. Human-readable `.response.md` remains owned by fixed `run`.

The cause-side firing rule stays unchanged:

```
framework_specified and selected_skill_id is not None
```

Checker observations remain observation-side and are recorded after the `do`
step has already decided whether the misreader skill fired.

Status-specific behavior is explicit:

- `contract_detected`: `framework_specified=true`, `selected_skill_id != null`,
  and the misreader may fire when `should_fire_misreader_skill(state)` returns true.
- `no_contract_detected`: `framework_specified=false`, `selected_skill_id=null`,
  no boundary event is observed, and `observation_outcome=not_applicable`.
- `unsupported`: `framework_specified=true`, `selected_skill_id=null`, no
  skill-triggered drift occurs, no checker bypass is observed, and
  `observation_outcome=not_applicable`.

## Minimal State Loop Path

The minimal controlled loop runs:

```
plan -> do -> check -> plan
```

It is a controlled poor-e2e fixture, not a general autonomous runtime.

- `plan` sets `framework_specified`
- `do` calls `should_fire_misreader_skill(state)` — a cause-side policy helper in `loop_trigger_policy.py` — to decide whether the misreader skill fires
- `check` runs the observation-side checker
- The loop stops deterministically via `max_iterations`
- `run-loop` emits `.loop_trace.json`

The loop exists to demonstrate intermittent boundary drift within one task execution.

> Intermittency is not randomness.
> It emerges when a state-local condition activates the wrong principle at one step.

> A loop is not observable until its state sequence becomes an artifact.

The `.loop_trace.json` artifact is the projection that makes the loop observable.
Without it, the loop runs but leaves no record of whether drift occurred.
Fixed scenario loop traces record `loop_source = "fixed_scenario"`.
Prompt-derived loop traces record `loop_source = "prompt_contract"` and the
prompt contract status.

## Artifact Boundaries

Each path owns a distinct set of artifacts.

| path | artifacts |
|---|---|
| `run` (fixed) | `.classification.json`, `.response.md`, `.trace.json`, `.state_trace.json`, `.checker.json`, `.checker_comparison.json` |
| `run-loop` | `.loop_trace.json` |
| `detect-contract` | `.prompt_contract.json` |
| `run-prompt-loop` | `.prompt_contract.json`, `.loop_trace.json` |

Do not blur these paths.
Each artifact has a single owning path.
A fixed run does not emit `.loop_trace.json`.
A `run-loop` does not emit classification or fixed benchmark artifacts.
`detect-contract` does not emit `.loop_trace.json`.
`run-prompt-loop` does not emit benchmark artifacts.

## Future Work

The following are out of scope for the current version:

- Additional safety-framework scenarios:
  - `instruction_zod_any`
  - `instruction_sql_parameterization_bypass`
  - `instruction_schema_unknown_blob`
- Richer skill-triggered drift semantics
- Richer arbitrary prompt loop semantics beyond the synthetic experimental path
- Multi-agent principal relativity (where agent A is the principal for agent B)

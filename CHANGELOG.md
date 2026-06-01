# Changelog

## Unreleased

- Clarified and tested `none` scenarios as negative controls that do not trigger skill drift or checker bypass observations.
- Added loop trace provenance fields so prompt-derived loop traces record their prompt contract source and status.
- Removed the legacy `run-prompt` classification path; arbitrary prompt observability is now contract-first via `detect-contract` and `run-prompt-loop`.
- Clarified and tested arbitrary prompt observability boundaries for `detect-contract` and `run-prompt-loop`.
- Added experimental `run-prompt-loop`, connecting prompt contract detection to the minimal loop and emitting `.prompt_contract.json` plus `.loop_trace.json`.
- Added experimental prompt contract detection with Pydantic AI Agent Skills selection, emitting `.prompt_contract.json` for arbitrary prompts.
- Extracted the minimal loop misreader trigger condition into a cause-side trigger policy helper.
- Added docs/design.md to document the design principles behind principal substitution, hidden divergence, skill perspectives, LangGraph state, and loop trace observability.
- Added `run-loop` CLI command: runs the minimal plan→do→check loop for one fixed scenario and emits a `<scenario_id>.loop_trace.json` artifact under a timestamped run directory. The fixed `run` command does not emit `.loop_trace.json`.
- Added loop trace artifact support: minimal loop execution can now be rendered to a `<scenario_id>.loop_trace.json` artifact via `render_loop_trace` and `run_minimal_loop_to_artifact`. The artifact is a projection from `WhoseAgentState` and is not emitted by the normal fixed `run` command.
- Added `instruction_pydantic_any`, a second `safety_framework_escape_hatch` scenario that demonstrates surface framework compliance plus semantic guarantee bypass in Pydantic rather than TypeScript.
- Reused the existing `safety_framework_escape_hatch` skill perspective to show the failure pattern generalizes beyond TypeScript.
- Added a minimal LangGraph plan -> do -> check loop path that demonstrates intermittent boundary drift within one task execution.
- Extended WhoseAgentState with minimal loop fields instead of wiring ControlState as a nested runtime object.
- Kept misreader firing cause-side (framework_specified + selected_skill_id) and checker observation observation-side in the loop.
- Added deterministic skill trigger state updates to the LangGraph fixed scenario path.
- Added checker-template comparison state and `.checker_comparison.json` artifacts.
- Render `.state_trace.json` from LangGraph state instead of the legacy boundary transition runtime.
- Pass selected skill perspective into non-mock bad response generation.

## v0.6.0

- Added passive ControlState and StepTrace primitives.
- Added principal/agent delegation state vocabulary for future State Loop support.
- Added checker and boundary-event fields as serializable state primitives.
- Clarified that these primitives do not introduce a State Loop runner.

## v0.5.0

- Added skill-perspective checker.
- Added human-authored skill perspective loading from skills/*.md.
- Added .checker.json for scenarios that opt into selected_skill_id.
- Moved mock checker output into scenario checker_template.
- Clarified that checker detection is perspective-based, not hard-coded token matching.

## v0.4.1

- Added instruction_typescript_any.
- Captured surface compliance with semantic constraint violation.
- Added a second instruction-axis scenario shape.

## v0.4.0

- Moved trace templates from axis-level runtime constants into scenario YAML.
- Made scenario trace_template the runtime source of truth.
- Enabled multiple scenario shapes per substitution axis.

## v0.3.x

- Clarified run-prompt as exploratory only.
- Ensured run-prompt emits only classification and flow artifacts.
- Cleaned up README wording and out-of-scope flow labels.
- Ensured JSON output preserves non-ASCII text.

## v0.3.0

- Added optional Langfuse observability.
- Kept missing credentials as no-op.
- Avoided sending raw prompts or raw bad responses to Langfuse.

## v0.2.0

- Added linear boundary state transition trace.
- Added .state_trace.json.
- Kept this as a linear trace, not a State Loop.

## v0.1.0

- Added fixed scenario benchmark core.
- Added four substitution axes.
- Added failure mode mapping.
- Added canonical scenarios and out-of-scope scenarios.
- Added deterministic mock output.
- Added benchmark trace artifacts.
- Added CI.

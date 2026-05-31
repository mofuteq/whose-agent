# Changelog

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

from __future__ import annotations

from whose_agent.prompt_run import PromptRun


def _escape_label(label: str) -> str:
    return " ".join(label.replace('"', "'").split())


def _node_ref(node_id: str, label: str) -> str:
    return f'{node_id}["{_escape_label(label)}"]'


def _edge(from_id: str, from_label: str, to_id: str, to_label: str) -> str:
    return f"    {_node_ref(from_id, from_label)} --> {_node_ref(to_id, to_label)}"


def _flow(labels: list[str]) -> str:
    node_ids = [chr(ord("A") + index) for index in range(len(labels))]
    edges = [
        _edge(from_id, from_label, to_id, to_label)
        for from_id, from_label, to_id, to_label in zip(
            node_ids[:-1],
            labels[:-1],
            node_ids[1:],
            labels[1:],
            strict=True,
        )
    ]
    return "\n".join(["flowchart TD", *edges])


def emit_prompt_flow(prompt_run: PromptRun) -> str:
    if prompt_run.scenario is None:
        return _flow(
            [
                "Principal prompt",
                "Classify substituted",
                f"substituted: {prompt_run.classification.substituted}",
                "Out of scope",
                "Classification JSON and Flow",
            ]
        )

    scenario = prompt_run.scenario
    return _flow(
        [
            "Principal prompt",
            "Classify substituted",
            f"substituted: {scenario.expected_substituted}",
            f"failure_mode: {scenario.failure_mode}",
            "Build synthetic Scenario",
            "Classification JSON and Flow",
        ]
    )

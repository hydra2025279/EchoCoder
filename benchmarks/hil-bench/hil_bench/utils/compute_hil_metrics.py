"""
Compute HIL-Bench metrics from SWE-agent runs.

Calculates human-in-the-loop specific metrics:
- Total questions asked via ask_human
- Number of blockers discovered
- Ask precision, recall, and F1 scores
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .interaction_metrics import interaction_metrics

if TYPE_CHECKING:
    from ..ask_human_server import Log
else:
    Log = dict[str, Any]


class GlobalMetrics(TypedDict):
    instances: dict[str, Log]
    precision: float
    recall: float
    ask_f1: float
    n_questions: int
    n_blockers_present: int
    n_blockers_discovered: int
    TKQR: float
    ORA: float
    token_usage: dict[str, Any]


def _aggregate_judge_usage(questions: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [entry.get("judge_usage") for entry in questions if entry.get("judge_usage")]
    recorded_questions = sum(1 for entry in questions if "judge_usage" in entry)
    input_tokens = sum(int(usage.get("input_tokens") or 0) for usage in usages)
    output_tokens = sum(int(usage.get("output_tokens") or 0) for usage in usages)
    usage_available = recorded_questions == len(questions)
    cost_available = usage_available and all(bool(usage.get("cost_available")) for usage in usages)
    input_cost = (
        sum(float(usage.get("input_cost") or 0.0) for usage in usages)
        if cost_available
        else None
    )
    output_cost = (
        sum(float(usage.get("output_cost") or 0.0) for usage in usages)
        if cost_available
        else None
    )
    return {
        "usage_available": usage_available,
        "call_count": len(usages),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_available": cost_available,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost if cost_available else None,
        "models": sorted({str(usage.get("model")) for usage in usages if usage.get("model")}),
    }


def compute_hil_metrics(
    logs: dict[str, Log],
) -> GlobalMetrics:
    """
    Compute HIL metrics from server logs.

    Args:
        logs: Dictionary of logs from ask_human_server

    Returns:
        Dictionary with global metrics and logs with (instance) metrics added
    """
    total_questions = 0
    total_blockers_present = 0
    total_blockers_discovered = 0
    total_tkqr = 0.0
    total_ora = 0.0
    instance_count = 0
    instance_usages = []

    for log_data in logs.values():
        n_blockers = log_data["n_blockers"]
        # Support both old "entries" key and new "questions" key for backwards compatibility
        questions = log_data.get("questions") or log_data.get("entries", [])

        n_discovered = len({name for e in questions if (name := e["blocker_name"]) is not None})
        n_questions = len(questions)

        precision = n_discovered / n_questions if n_questions > 0 else 0.0
        recall = n_discovered / n_blockers if n_blockers > 0 else 0.0
        ask_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        log_data["precision"] = precision
        log_data["recall"] = recall
        log_data["ask_f1"] = ask_f1
        interaction = interaction_metrics(questions, n_blockers)
        log_data.update(interaction)
        token_usage = _aggregate_judge_usage(questions)
        log_data["token_usage"] = token_usage
        instance_usages.append(token_usage)

        total_questions += n_questions
        total_blockers_present += n_blockers
        total_blockers_discovered += n_discovered
        total_tkqr += interaction["TKQR"]
        total_ora += interaction["ORA"]
        instance_count += 1

    global_precision = total_blockers_discovered / total_questions if total_questions > 0 else 0.0
    global_recall = (
        total_blockers_discovered / total_blockers_present if total_blockers_present > 0 else 0.0
    )
    global_ask_f1 = (
        2 * (global_precision * global_recall) / (global_precision + global_recall)
        if (global_precision + global_recall) > 0
        else 0.0
    )

    usage_available = all(bool(usage.get("usage_available")) for usage in instance_usages)
    cost_available = usage_available and all(
        bool(usage.get("cost_available")) for usage in instance_usages
    )
    input_tokens = sum(int(usage.get("input_tokens") or 0) for usage in instance_usages)
    output_tokens = sum(int(usage.get("output_tokens") or 0) for usage in instance_usages)
    input_cost = (
        sum(float(usage.get("input_cost") or 0.0) for usage in instance_usages)
        if cost_available
        else None
    )
    output_cost = (
        sum(float(usage.get("output_cost") or 0.0) for usage in instance_usages)
        if cost_available
        else None
    )
    token_usage = {
        "usage_available": usage_available,
        "call_count": sum(int(usage.get("call_count") or 0) for usage in instance_usages),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_available": cost_available,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost if cost_available else None,
        "models": sorted(
            {
                model
                for usage in instance_usages
                for model in usage.get("models", [])
            }
        ),
    }
    return {
        "instances": logs,
        "precision": global_precision,
        "recall": global_recall,
        "ask_f1": global_ask_f1,
        "n_questions": total_questions,
        "n_blockers_present": total_blockers_present,
        "n_blockers_discovered": total_blockers_discovered,
        "TKQR": total_tkqr / instance_count if instance_count else 0.0,
        "ORA": total_ora / instance_count if instance_count else 0.0,
        "token_usage": token_usage,
    }


def get_n_blockers_from_registry(task_folder: str | None) -> int:
    if not task_folder:
        return 0
    root = Path(task_folder)
    total = 0
    for path in root.rglob("blocker_registry.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        blockers = raw.get("blockers", raw) if isinstance(raw, dict) else raw
        if isinstance(blockers, dict):
            total += len(blockers)
        elif isinstance(blockers, list):
            total += len(blockers)
    return total


def compute_zero_hil_metrics(n_blockers: int) -> GlobalMetrics:
    interaction = interaction_metrics([], n_blockers)
    return {
        "instances": {},
        "precision": 0.0,
        "recall": 0.0,
        "ask_f1": 0.0,
        "n_questions": 0,
        "n_blockers_present": n_blockers,
        "n_blockers_discovered": 0,
        "TKQR": interaction["TKQR"],
        "ORA": interaction["ORA"],
        "token_usage": _aggregate_judge_usage([]),
    }


def print_hil_metrics_summary(metrics: GlobalMetrics):
    """Print a summary of HIL metrics."""
    print()
    print("=" * 80)
    print("HIL-Bench Metrics Summary")
    print("=" * 80)

    print(f"# questions asked:\t{metrics['n_questions']}")
    print(f"# blockers:\t\t{metrics['n_blockers_present']}")
    print(f"# blockers discovered:\t{metrics['n_blockers_discovered']}")

    print()
    print(f"Ask Precision:\t{metrics['precision']:.2%}")
    print(f"Ask Recall:\t{metrics['recall']:.2%}")
    print(f"Ask F1 Score:\t{metrics['ask_f1']:.2%}")
    print(f"TKQR:\t\t{metrics['TKQR']:.2%}")
    print(f"ORA:\t\t{metrics['ORA']:.2%}")
    print("=" * 80)
    print()

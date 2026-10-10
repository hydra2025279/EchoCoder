from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.common import summarize_passrate_records, summarize_question_records
from benchmarks.ClarifyCodeBench.clarifycodebench.metrics import calculate_ora, calculate_tkqr


def extract_record(record: dict[str, Any], source_file: Path) -> dict[str, Any]:
    iterations = record.get("question_iterations") or []
    if not iterations or iterations[0].get("iteration") != 1:
        raise ValueError(f"{source_file}: missing first question iteration")

    first = iterations[0]
    rows = first.get("question_records")
    if not isinstance(rows, list) or len(rows) != len(first.get("questions", [])):
        raise ValueError(f"{source_file}: first-call question records do not align")
    metrics = first.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{source_file}: first-call metrics are missing")

    question_metrics = {
        key: value for key, value in metrics.items() if not key.endswith("_so_far")
    }
    key_count = int(question_metrics["total_key_questions"])
    question_metrics["evaluations"] = rows
    question_metrics["TKQR"] = calculate_tkqr(
        [int(bool(row["correct"])) for row in rows] + [0], key_count
    )
    question_metrics["ORA"] = calculate_ora(len(rows), key_count)

    return {
        "task_id": record["task_id"],
        "question_id": record.get("question_id"),
        "benchmark": record.get("benchmark", "clarifycodebench"),
        "task_type": record.get("task_type"),
        "method": "direct_prompting_1iter",
        "source_result": str(source_file.resolve()),
        "question_call_budget": 1,
        "iteration_count": 1,
        "question_responses": [first["raw_response"]],
        "question_iterations": [first],
        "generated_questions": first["questions"],
        "question_metrics": question_metrics,
        "pass_at_1": None,
    }


def extract_directory(source: Path, output: Path) -> dict[str, Any]:
    if source.resolve() == output.resolve():
        raise ValueError("Source and output directories must differ")
    files = sorted((source / "result").glob("*.json"))
    if not files:
        raise ValueError(f"No per-task results found in {source / 'result'}")

    records = [extract_record(json.loads(path.read_text()), path) for path in files]
    task_ids = [record["task_id"] for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs in source results")

    result_dir = output / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        (result_dir / f"{record['task_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
    (output / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    summary = summarize_question_records(records)
    (output / "summary.question.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "summary.passrate.json").write_text(
        json.dumps(summarize_passrate_records(records, benchmark="clarifycodebench"), ensure_ascii=False, indent=2)
        + "\n"
    )
    (output / "README.md").write_text(
        "# Direct Prompting: first question call\n\n"
        f"Extracted from `{source.resolve()}`. Each task contains only its first "
        "question batch (`I1`) and the original oracle judgments for that batch. "
        "Question metrics were recomputed from those judgments. The source run's "
        "final code and pass rate are excluded because they may use answers from "
        "later question calls. `summary.passrate.json` therefore marks all tasks "
        "as unevaluated.\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = extract_directory(args.source, args.output)
    print(
        f"Extracted {summary['sample_count']} tasks to {args.output}; "
        f"macro Ask-F1={summary['macro']['ask_f1_ques']:.4f}"
    )


if __name__ == "__main__":
    main()

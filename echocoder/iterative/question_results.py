"""Per-task question artifacts and summaries for iterative EchoCoder runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from baselines.common import (
    clarify_pass_at_1,
    summarize_llm_call_usage,
    summarize_passrate_records,
    summarize_question_records,
    write_json,
)
from benchmarks.ClarifyCodeBench.clarifycodebench.metrics import (
    calculate_ora,
    calculate_tkqr,
)


def _matched_target_id(row: dict[str, Any]) -> str | None:
    matched_key_id = row.get("matched_key_id")
    if matched_key_id is not None:
        return f"id:{matched_key_id}"
    matched_question = row.get("matched_key_question")
    if matched_question:
        return f"question:{matched_question}"
    return None


def _question_hit_sequence(
    question_rows: list[dict[str, Any]], total_key_questions: int
) -> list[int]:
    covered_targets: set[str] = set()
    hit_sequence: list[int] = []
    anonymous_hits = 0
    for row in question_rows:
        target_id = _matched_target_id(row)
        is_valid = bool(row.get("correct"))
        if target_id is None:
            is_new_hit = (
                is_valid
                and len(covered_targets) + anonymous_hits < total_key_questions
            )
            if is_new_hit:
                anonymous_hits += 1
        else:
            is_new_hit = is_valid and target_id not in covered_targets
            if is_new_hit:
                covered_targets.add(target_id)
        hit_sequence.append(1 if is_new_hit else 0)
    return hit_sequence


def _question_and_level_metrics(
    question_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    total_key_questions: int,
) -> dict[str, Any]:
    correct_questions = sum(bool(row.get("correct")) for row in question_rows)
    covered_targets = {
        target_id
        for row in question_rows
        if row.get("correct")
        for target_id in [_matched_target_id(row)]
        if target_id is not None
    }
    nonempty_batches = [
        row for row in batch_rows if int(row.get("question_count", 0)) > 0
    ]
    valid_nonempty_batches = sum(
        bool(row.get("batch_valid")) for row in nonempty_batches
    )
    recall_ques = (
        len(covered_targets) / total_key_questions if total_key_questions else 0.0
    )
    precision_ques = (
        correct_questions / len(question_rows) if question_rows else 0.0
    )
    ask_f1_ques = (
        2 * precision_ques * recall_ques / (precision_ques + recall_ques)
        if precision_ques + recall_ques
        else 0.0
    )
    return {
        "recall_ques": recall_ques,
        "precision_ques": precision_ques,
        "precision_level": (
            valid_nonempty_batches / len(nonempty_batches)
            if nonempty_batches
            else 0.0
        ),
        "ask_f1_ques": ask_f1_ques,
        "covered_key_questions": len(covered_targets),
        "total_generated_questions": len(question_rows),
        "correct_generated_questions": correct_questions,
        "nonempty_batches": len(nonempty_batches),
        "valid_nonempty_batches": valid_nonempty_batches,
    }


def _metrics_with_question_and_level_dimensions(
    question_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    total_key_questions: int,
) -> dict[str, Any]:
    metrics = _question_and_level_metrics(
        question_rows, batch_rows, total_key_questions
    )
    metrics.update(
        {
            "metric_unit": "question_and_nonempty_batch",
            "total_key_questions": total_key_questions,
            "total_question_calls": len(batch_rows),
            "empty_batches": len(batch_rows)
            - sum(int(row.get("question_count", 0)) > 0 for row in batch_rows),
        }
    )
    return metrics


def _question_metrics_so_far(
    question_rows: list[dict[str, Any]], total_key_questions: int
) -> dict[str, Any]:
    metrics = _question_and_level_metrics(question_rows, [], total_key_questions)
    return {
        "precision_ques_so_far": metrics["precision_ques"],
        "recall_ques_so_far": metrics["recall_ques"],
        "ask_f1_ques_so_far": metrics["ask_f1_ques"],
        "total_generated_questions_so_far": metrics["total_generated_questions"],
        "correct_generated_questions_so_far": metrics[
            "correct_generated_questions"
        ],
        "covered_key_questions_so_far": metrics["covered_key_questions"],
    }


def _batch_metric_row(
    rows: list[dict[str, Any]],
    *,
    state_index: int,
    code_id: str,
    reconstruction_index: int,
    question_call_index: int,
) -> dict[str, Any]:
    reference = _batch_reference(
        state_index, code_id, reconstruction_index, question_call_index
    )
    valid_rows = [row for row in rows if row["correct"]]
    return {
        **reference,
        "correct": bool(valid_rows),
        "batch_valid": bool(valid_rows),
        "question_count": len(rows),
        "valid_question_count": len(valid_rows),
        "invalid_question_count": len(rows) - len(valid_rows),
        "valid_question_ids": [row["question_id"] for row in valid_rows],
        "matched_key_questions": [
            row["matched_key_question"]
            for row in valid_rows
            if row.get("matched_key_question")
        ],
    }


def _question_row(
    question: dict[str, Any],
    *,
    state_index: int,
    reconstruction_index: int,
    question_call_index: int,
    batch_index: int,
    code_id: str,
) -> dict[str, Any]:
    evaluation = question.get("oracle_evaluation") or {}
    local_id = question.get("question_id") or (
        f"Q{reconstruction_index}_{question_call_index}_{batch_index}"
    )
    return {
        "question_id": local_id,
        "global_question_id": question.get("global_question_id")
        or f"S{state_index}_{code_id}_{local_id}",
        "specification_state": state_index,
        "r": reconstruction_index,
        "q": question_call_index,
        "batch_index": batch_index,
        "code_id": code_id,
        "question": question.get("question"),
        "oracle_status": question.get("oracle_status"),
        "correct": question.get("oracle_status") == "valid",
        "answer": question.get("answer"),
        "matched_key_question": question.get("matched_key_question"),
        "matched_key_id": evaluation.get("matched_key_id"),
        "duplicate": bool(evaluation.get("duplicate")),
        "oracle_evaluation": evaluation,
    }


def _batch_reference(
    state_index: int,
    code_id: str,
    reconstruction_index: int,
    question_call_index: int,
) -> dict[str, Any]:
    batch_id = f"Q{reconstruction_index}_{question_call_index}"
    return {
        "batch_id": batch_id,
        "global_batch_id": f"S{state_index}_{code_id}_{batch_id}",
        "specification_state": state_index,
        "r": reconstruction_index,
        "q": question_call_index,
    }


def _triggering_batch(round_data: dict[str, Any]) -> dict[str, Any] | None:
    accepted = round_data.get("accepted_clarifications", []) or []
    if not accepted:
        return None
    last = accepted[-1]
    reference = _batch_reference(
        int(round_data.get("specification_round", 0)),
        str(round_data.get("code_id") or f"C{round_data.get('specification_round', 0)}"),
        int(last["reconstruction_index"]),
        int(last["question_call_index"]),
    )
    reference["accepted_questions"] = [
        {
            "question": item.get("question"),
            "answer": item.get("answer"),
            "matched_key_question": item.get("matched_key_question"),
        }
        for item in accepted
        if int(item.get("reconstruction_index", -1)) == reference["r"]
        and int(item.get("question_call_index", -1)) == reference["q"]
    ]
    return reference


def _evaluation_summary(evaluation: Any) -> dict[str, Any] | None:
    if not isinstance(evaluation, dict):
        return None
    pass_rate = evaluation.get("pass_rate")
    if pass_rate is None and isinstance(evaluation.get("passed"), bool):
        pass_rate = 1.0 if evaluation["passed"] else 0.0
    avg_pass_rate = evaluation.get("avg_pass_rate", pass_rate)
    passed = evaluation.get("pass")
    if not isinstance(passed, bool):
        raw_passed = evaluation.get("passed")
        total = evaluation.get("total")
        if isinstance(raw_passed, bool):
            passed = raw_passed
        elif isinstance(raw_passed, int) and isinstance(total, int) and total > 0:
            passed = raw_passed == total
        else:
            passed = None
    return {
        "enabled": bool(evaluation.get("enabled")),
        "pass": passed,
        "avg_pass_rate": avg_pass_rate,
        "passed": evaluation.get("passed"),
        "pass_rate": pass_rate,
        "evaluation": evaluation,
    }


def _build_pass_rate_result(
    record: dict[str, Any], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    generated_patches: list[dict[str, Any]] = []
    state_evaluations = {
        str(item.get("code_id")): item.get("evaluation")
        for item in record.get("code_state_evaluations", [])
        if isinstance(item, dict)
    }
    previous_trigger = None
    last_state = None
    for offset, round_data in enumerate(rounds):
        state_index = int(round_data.get("specification_round", offset))
        state_id = round_data.get("state_id") or f"S{state_index}"
        code_id = round_data.get("code_id") or f"C{state_index}"
        patch = {
            "patch_id": code_id,
            "code_id": code_id,
            "state_id": state_id,
            "specification_state": state_index,
            "patch": round_data.get("complete_code"),
            "corresponding_question_batch": previous_trigger,
            "generation_reason": (
                "initial_generation"
                if previous_trigger is None
                else "specification_updated_by_valid_batch"
            ),
            "evaluations": {},
        }
        state_evaluation = _evaluation_summary(
            state_evaluations.get(patch["patch_id"])
        )
        if state_evaluation is not None:
            patch["evaluations"]["post_hoc"] = state_evaluation
        if offset == 0:
            initial = _evaluation_summary(record.get("initial_code_evaluation"))
            if initial is not None:
                patch["evaluations"]["initial"] = initial
        generated_patches.append(patch)
        last_state = patch
        previous_trigger = _triggering_batch(round_data)

    final_evaluation = _evaluation_summary(
        record.get("final_code_evaluation", record.get("pass_evaluation"))
    )
    selected_submission = record.get(
        "selected_submission", record.get("final_solution")
    )
    final_trigger = (
        last_state.get("corresponding_question_batch") if last_state else None
    )
    if last_state and selected_submission == last_state.get("patch"):
        if final_evaluation is not None:
            last_state["evaluations"]["final"] = final_evaluation

    return {
        "generated_patches": generated_patches,
        "final": {
            "state_id": last_state.get("state_id") if last_state else None,
            "code_id": last_state.get("code_id") if last_state else None,
            **(final_evaluation or {
                "enabled": False,
                "pass": None,
                "avg_pass_rate": None,
                "passed": None,
                "pass_rate": None,
                "evaluation": None,
            }),
        },
        "submitted_patch": {
            "patch_id": "final_submission",
            "patch": selected_submission,
            "based_on_specification_state": (
                last_state.get("specification_state") if last_state else None
            ),
            "corresponding_question_batch": final_trigger,
            "same_as_last_generated_patch": bool(
                last_state and selected_submission == last_state.get("patch")
            ),
            "evaluation": final_evaluation,
        },
    }


def _build_preclarification_task_result(
    record: dict[str, Any], preclarification: dict[str, Any]
) -> dict[str, Any]:
    existing_metrics = record.get("question_metrics") or {}
    total_key_questions = int(existing_metrics.get("total_key_questions", 0))
    llm_calls = list(record.get("llm_calls", []) or [])
    questions = list(preclarification.get("questions", []) or [])
    rows = [
        _question_row(
            question,
            state_index=0,
            reconstruction_index=0,
            question_call_index=1,
            batch_index=index,
            code_id="PRE",
        )
        for index, question in enumerate(questions, start=1)
    ]
    batch = _batch_metric_row(
        rows,
        state_index=0,
        code_id="PRE",
        reconstruction_index=0,
        question_call_index=1,
    )
    batch["batch_id"] = preclarification.get("batch_id", "PRE_Q1")
    batch["global_batch_id"] = preclarification.get(
        "global_batch_id", "PRE_Q1"
    )
    metrics = _metrics_with_question_and_level_dimensions(
        rows, [batch], total_key_questions
    )
    metrics["evaluations"] = [batch]
    related_calls = [
        call
        for call in llm_calls
        if call.get("global_batch_id") == batch["global_batch_id"]
    ]
    hit_sequence = _question_hit_sequence(rows, total_key_questions)
    question_turn_count = len(rows)
    tkqr = calculate_tkqr(hit_sequence + [0], total_key_questions)
    if total_key_questions == 0 and question_turn_count == 0:
        tkqr = 1.0
    interaction_metrics = {
        "TKQR": tkqr,
        "ORA": calculate_ora(question_turn_count, total_key_questions),
        "question_turns": question_turn_count,
        "key_questions": total_key_questions,
        "diff": question_turn_count - total_key_questions,
        "hit_sequence": hit_sequence,
        "turn_unit": "generated_question",
        "final_submission_turn_included_in_tkqr": True,
    }
    metrics["TKQR"] = interaction_metrics["TKQR"]
    metrics["ORA"] = interaction_metrics["ORA"]
    pass_rate_result = _build_pass_rate_result(record, [])
    artifact = {
        "task_id": record.get("task_id"),
        "question_id": record.get("question_id"),
        "method": record.get("method"),
        "benchmark": record.get("benchmark", "clarifycodebench"),
        "task_type": record.get("task_type"),
        "ground_truth": record.get("ground_truth", []),
        "generation_metadata": record.get("generation_metadata", {}),
        "run_config": record.get("run_config", {}),
        "llm_calls": llm_calls,
        "token_usage": record.get("token_usage", {}),
        "ask_result": {
            "generated_questions": [row.get("question") for row in rows],
            "metrics": metrics,
            "states": {},
            "preclarification": {
                "batch_id": batch["batch_id"],
                "global_batch_id": batch["global_batch_id"],
                "question_call_index": 1,
                "original_specification": preclarification.get(
                    "original_specification"
                ),
                "final_specification": preclarification.get("final_specification"),
                "detector_response": preclarification.get("detector_response"),
                "question_batch": rows,
                "question_count": batch["question_count"],
                "valid_question_count": batch["valid_question_count"],
                "invalid_question_count": batch["invalid_question_count"],
                "batch_valid": batch["batch_valid"],
                "accepted_clarifications": preclarification.get(
                    "accepted_clarifications", []
                ),
                "llm_call_ids": [
                    call.get("call_id")
                    for call in related_calls
                    if call.get("call_id")
                ],
                "token_usage": summarize_llm_call_usage(related_calls),
                "metrics": metrics,
            },
            "interaction_metrics": interaction_metrics,
        },
        "code_evaluation": {"states": {}, "final": pass_rate_result["final"]},
        "pass_rate": pass_rate_result,
    }
    if record.get("benchmark", "clarifycodebench") == "clarifycodebench":
        artifact["pass_at_1"] = clarify_pass_at_1(record)
    return artifact


def build_task_question_result(record: dict[str, Any]) -> dict[str, Any]:
    """Build one hierarchical question artifact from a completed task record."""
    existing_metrics = record.get("question_metrics") or {}
    total_key_questions = int(existing_metrics.get("total_key_questions", 0))
    states: dict[str, Any] = {}
    all_question_rows: list[dict[str, Any]] = []
    all_batch_rows: list[dict[str, Any]] = []
    so_far_question_rows: list[dict[str, Any]] = []
    llm_calls = list(record.get("llm_calls", []) or [])

    rounds = record.get("rounds", []) or []
    preclarification = record.get("preclarification")
    if isinstance(preclarification, dict) and not rounds:
        return _build_preclarification_task_result(record, preclarification)
    state_evaluation_map = {
        str(item.get("code_id")): _evaluation_summary(item.get("evaluation"))
        for item in record.get("code_state_evaluations", [])
        if isinstance(item, dict)
    }
    for state_offset, round_data in enumerate(rounds):
        state_index = int(round_data.get("specification_round", state_offset))
        state_id = round_data.get("state_id") or f"S{state_index}"
        code_id = round_data.get("code_id") or f"C{state_index}"
        code = str(round_data.get("complete_code", ""))
        state_question_rows: list[dict[str, Any]] = []
        state_batch_rows: list[dict[str, Any]] = []
        reconstructions: dict[str, Any] = {}
        for reconstruction in round_data.get("reconstructions", []) or []:
            r = int(reconstruction.get("reconstruction_index", len(reconstructions) + 1))
            reconstruction_id = (
                reconstruction.get("reconstruction_id")
                or f"{state_id}_{code_id}_R{r}"
            )
            reconstruction_question_rows: list[dict[str, Any]] = []
            reconstruction_batch_rows: list[dict[str, Any]] = []
            calls: dict[str, Any] = {}
            for batch in reconstruction.get("question_batches", []) or []:
                q = int(batch.get("question_call_index", len(calls) + 1))
                batch_reference = _batch_reference(state_index, code_id, r, q)
                batch_reference["batch_id"] = (
                    batch.get("batch_id") or batch_reference["batch_id"]
                )
                batch_reference["global_batch_id"] = (
                    batch.get("global_batch_id")
                    or batch_reference["global_batch_id"]
                )
                related_calls = [
                    call
                    for call in llm_calls
                    if call.get("global_batch_id")
                    == batch_reference["global_batch_id"]
                ]
                added_points = list(reconstruction.get("added_points", []) or [])
                initial_questions = list(round_data.get("initial_questions", []) or [])
                evidence_context = {
                    "original_specification_id": f"S{state_index}_SPEC",
                    "code_id": code_id,
                    "reconstruction_id": reconstruction_id,
                    "added_points": [
                        {
                            "evidence_id": f"S{state_index}_{code_id}_R{r}_AP{i}",
                            "content": point,
                        }
                        for i, point in enumerate(added_points, start=1)
                    ],
                    "initial_questions": [
                        {
                            "evidence_id": f"S{state_index}_IQ{i}",
                            "content": question,
                        }
                        for i, question in enumerate(initial_questions, start=1)
                    ],
                }
                rows = [
                    _question_row(
                        question,
                        state_index=state_index,
                        reconstruction_index=r,
                        question_call_index=q,
                        batch_index=i,
                        code_id=code_id,
                    )
                    for i, question in enumerate(batch.get("questions", []) or [], start=1)
                ]
                for row in rows:
                    exact_initial_matches = [
                        item["evidence_id"]
                        for item in evidence_context["initial_questions"]
                        if str(item["content"]).strip().casefold()
                        == str(row.get("question", "")).strip().casefold()
                    ]
                    row["provenance"] = {
                        "attribution_status": (
                            "exact_initial_match"
                            if exact_initial_matches
                            else "not_explicitly_reported"
                        ),
                        "exact_initial_question_ids": exact_initial_matches,
                        "available_evidence_ids": [
                            evidence["evidence_id"]
                            for evidence in (
                                evidence_context["added_points"]
                                + evidence_context["initial_questions"]
                            )
                        ],
                    }
                batch_row = _batch_metric_row(
                    rows,
                    state_index=state_index,
                    code_id=code_id,
                    reconstruction_index=r,
                    question_call_index=q,
                )
                so_far_question_rows.extend(rows)
                batch_metrics = _metrics_with_question_and_level_dimensions(
                    rows, [batch_row], total_key_questions
                )
                batch_metrics.update(
                    _question_metrics_so_far(
                        so_far_question_rows, total_key_questions
                    )
                )
                calls[f"q_{q}"] = {
                    "q": q,
                    **batch_reference,
                    "code_id": code_id,
                    "evidence_context": evidence_context,
                    "llm_call_ids": [
                        call.get("call_id")
                        for call in related_calls
                        if call.get("call_id")
                    ],
                    "token_usage": summarize_llm_call_usage(related_calls),
                    "detector_response": batch.get("detector_response"),
                    "prior_questions": batch.get("prior_questions", []),
                    "prior_valid_clarifications": batch.get(
                        "prior_valid_clarifications", []
                    ),
                    "prior_invalid_questions": batch.get(
                        "prior_invalid_questions", []
                    ),
                    "question_batch": rows,
                    "question_count": batch_row["question_count"],
                    "valid_question_count": batch_row["valid_question_count"],
                    "invalid_question_count": batch_row[
                        "invalid_question_count"
                    ],
                    "batch_valid": batch_row["batch_valid"],
                    "metrics": batch_metrics,
                }
                reconstruction_question_rows.extend(rows)
                reconstruction_batch_rows.append(batch_row)
            reconstruction_metrics = _metrics_with_question_and_level_dimensions(
                reconstruction_question_rows,
                reconstruction_batch_rows,
                total_key_questions,
            )
            reconstruction_metrics.update(
                _question_metrics_so_far(so_far_question_rows, total_key_questions)
            )
            reconstructions[f"r_{r}"] = {
                "r": r,
                "reconstruction_id": reconstruction_id,
                "reconstructed_description": reconstruction.get(
                    "reconstructed_description"
                ),
                "added_points": reconstruction.get("added_points", []),
                "question_calls": calls,
                "metrics": reconstruction_metrics,
            }
            state_question_rows.extend(reconstruction_question_rows)
            state_batch_rows.extend(reconstruction_batch_rows)
        state_metrics = _metrics_with_question_and_level_dimensions(
            state_question_rows,
            state_batch_rows,
            total_key_questions,
        )
        state_metrics.update(
            _question_metrics_so_far(so_far_question_rows, total_key_questions)
        )
        states[f"state_{state_index}"] = {
            "specification_state": state_index,
            "state_id": state_id,
            "generation_id": round_data.get("generation_id") or f"{state_id}_GEN",
            "generation_engine": round_data.get("generation_engine"),
            "generation_operation": round_data.get("generation_operation"),
            "parent_code_id": round_data.get("parent_code_id"),
            "code_id": code_id,
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "complete_code": code,
            "specification": round_data.get("specification"),
            "initial_questions": round_data.get("initial_questions", []),
            "accepted_clarifications": round_data.get(
                "accepted_clarifications", []
            ),
            "reconstructions": reconstructions,
            "metrics": state_metrics,
            "code_evaluation": state_evaluation_map.get(code_id)
            or {
                "enabled": False,
                "pass": None,
                "avg_pass_rate": None,
                "passed": None,
                "pass_rate": None,
                "evaluation": None,
            },
        }
        all_question_rows.extend(state_question_rows)
        all_batch_rows.extend(state_batch_rows)

    metrics = _metrics_with_question_and_level_dimensions(
        all_question_rows,
        all_batch_rows,
        total_key_questions,
    )
    metrics["evaluations"] = all_batch_rows
    if not record.get("rounds") and existing_metrics:
        metrics = existing_metrics

    ask_result = {
        "generated_questions": [row.get("question") for row in all_question_rows]
        if all_question_rows
        else list(record.get("generated_questions", []) or []),
        "metrics": metrics,
        "states": states,
    }
    pass_rate_result = _build_pass_rate_result(record, rounds)
    artifact = {
        "task_id": record.get("task_id"),
        "question_id": record.get("question_id"),
        "method": record.get("method"),
        "benchmark": record.get("benchmark", "clarifycodebench"),
        "task_type": record.get("task_type"),
        "ground_truth": record.get("ground_truth", []),
        "generation_metadata": record.get("generation_metadata", {}),
        "run_config": record.get("run_config", {}),
        "llm_calls": llm_calls,
        "token_usage": record.get("token_usage", {}),
        "ask_result": ask_result,
        "code_evaluation": {
            "states": {
                state_key: state["code_evaluation"]
                for state_key, state in states.items()
            },
            "final": pass_rate_result["final"],
        },
        "pass_rate": pass_rate_result,
    }
    if record.get("benchmark", "clarifycodebench") == "clarifycodebench":
        artifact["pass_at_1"] = clarify_pass_at_1(record)

    hit_sequence = _question_hit_sequence(all_question_rows, total_key_questions)
    question_turn_count = len(all_question_rows)
    tkqr = calculate_tkqr(hit_sequence + [0], total_key_questions)
    if total_key_questions == 0 and question_turn_count == 0:
        tkqr = 1.0
    interaction_metrics = {
        "TKQR": tkqr,
        "ORA": calculate_ora(question_turn_count, total_key_questions),
        "question_turns": question_turn_count,
        "key_questions": total_key_questions,
        "diff": question_turn_count - total_key_questions,
        "hit_sequence": hit_sequence,
        "turn_unit": "generated_question",
        "final_submission_turn_included_in_tkqr": True,
    }
    ask_result["interaction_metrics"] = interaction_metrics
    ask_result["metrics"]["TKQR"] = interaction_metrics["TKQR"]
    ask_result["metrics"]["ORA"] = interaction_metrics["ORA"]
    return artifact


def _safe_task_id(task_id: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id)).strip("._")
    return safe or "unknown_task"


def question_result_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "result"


def write_task_question_result(record: dict[str, Any], output_dir: str | Path) -> Path:
    path = question_result_dir(output_dir) / f"{_safe_task_id(record.get('task_id'))}.json"
    write_json(build_task_question_result(record), path)
    return path


def load_task_question_results(output_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(question_result_dir(output_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def sync_task_question_results(
    records: Iterable[dict[str, Any]], output_dir: str | Path, *, prune: bool = True
) -> None:
    result_dir = question_result_dir(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    for record in records:
        expected.add(write_task_question_result(record, output_dir))
    if prune:
        for path in list(result_dir.glob("*.json")) + list(
            result_dir.glob("*.jsonl")
        ):
            if path not in expected:
                path.unlink()


def clear_task_question_results(output_dir: str | Path) -> None:
    result_dir = question_result_dir(output_dir)
    if not result_dir.exists():
        return
    for path in list(result_dir.glob("*.json")) + list(result_dir.glob("*.jsonl")):
        path.unlink()


def summarize_task_question_results(output_dir: str | Path) -> dict[str, Any]:
    """Build the benchmark summary exclusively from per-task result files."""
    records = load_task_question_results(output_dir)
    summary_records = [
        {
            **{key: record.get(key) for key in (
                "task_id", "question_id", "method", "benchmark", "task_type"
            )},
            "generated_questions": record.get("ask_result", {}).get(
                "generated_questions", []
            ),
            "question_metrics": record.get("ask_result", {}).get("metrics"),
        }
        for record in records
    ]
    summary = summarize_question_records(summary_records)
    interaction = [
        record["ask_result"]["interaction_metrics"]
        for record in records
        if isinstance(record.get("ask_result", {}).get("interaction_metrics"), dict)
    ]
    if interaction:
        count = len(interaction)
        turn_units = {item.get("turn_unit") for item in interaction}
        summary["TKQR"] = sum(item["TKQR"] for item in interaction) / count
        summary["ORA"] = sum(item["ORA"] for item in interaction) / count
        summary["interaction"] = {
            "sample_count": count,
            "TKQR": summary["TKQR"],
            "ORA": summary["ORA"],
            "average_question_turns": sum(
                item["question_turns"] for item in interaction
            )
            / count,
            "average_diff": sum(item["diff"] for item in interaction) / count,
            "turn_unit": (
                next(iter(turn_units)) if len(turn_units) == 1 else "mixed"
            ),
        }
    return summary


def summarize_task_pass_rate_results(
    output_dir: str | Path, *, benchmark: str | None = None
) -> dict[str, Any]:
    records = load_task_question_results(output_dir)
    if benchmark == "clarifycodebench" or (
        records and all(record.get("benchmark") == "clarifycodebench" for record in records)
    ):
        return summarize_passrate_records(records, benchmark="clarifycodebench")
    summary_records = []
    for record in records:
        submitted = record.get("pass_rate", {}).get("submitted_patch", {})
        wrapped_evaluation = submitted.get("evaluation") or {}
        summary_records.append({
            **{key: record.get(key) for key in ("task_id", "question_id", "method", "benchmark", "task_type")},
            "pass_evaluation": wrapped_evaluation.get("evaluation"),
            "question_metrics": (record.get("ask_result") or {}).get("metrics"),
        })
    summary = summarize_passrate_records(summary_records)
    iteration_buckets: dict[int, dict[str, int]] = {}
    for record in records:
        for patch in record.get("pass_rate", {}).get("generated_patches", []):
            iteration = int(patch.get("specification_state", 0))
            bucket = iteration_buckets.setdefault(
                iteration,
                {
                    "task_count": 0,
                    "evaluated_task_count": 0,
                    "passed_task_count": 0,
                    "skipped_task_count": 0,
                },
            )
            bucket["task_count"] += 1
            evaluation = patch.get("evaluations", {}).get("post_hoc") or {}
            if evaluation.get("enabled"):
                bucket["evaluated_task_count"] += 1
                bucket["passed_task_count"] += int(bool(evaluation.get("passed")))
            else:
                bucket["skipped_task_count"] += 1

    summary["code_iterations"] = {
        f"C{iteration}": {
            "iteration": iteration,
            "code_id": f"C{iteration}",
            **bucket,
            "pass_rate": (
                bucket["passed_task_count"] / bucket["evaluated_task_count"]
                if bucket["evaluated_task_count"]
                else None
            ),
        }
        for iteration, bucket in sorted(iteration_buckets.items())
    }
    return summary

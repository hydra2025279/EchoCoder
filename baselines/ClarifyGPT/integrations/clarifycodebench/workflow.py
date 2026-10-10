from __future__ import annotations

import re
from typing import Any

from baselines.common import chat_text, evaluate_questions
from benchmarks.ClarifyCodeBench.clarifycodebench.metrics import calculate_ora, calculate_tkqr
from baselines.ClarifyGPT.integrations.clarifycodebench.candidate_sampling import sample_candidate
from baselines.ClarifyGPT.integrations.clarifycodebench.execution import (
    cluster_by_output,
    execute_candidates,
)
from baselines.ClarifyGPT.integrations.clarifycodebench.test_inputs import (
    expand_inputs,
    generate_seed_inputs,
)
from baselines.ClarifyGPT.integrations.clarifycodebench.upstream_prompts import (
    question_prompt,
    synthesis_prompt,
)


def _questions_from_response(response: str) -> list[str]:
    section = re.split(r"clarifying questions\s*:", response, flags=re.IGNORECASE)[-1]
    questions = []
    for line in section.splitlines():
        match = re.match(r"\s*(?:\d+|[-*])(?:[.)])?\s+(.+?)\s*$", line)
        if match:
            question = match.group(1).strip()
            if question and question not in questions:
                questions.append(question)
    return questions


def _clean_code(response: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else response).strip()


def _accepted_clarifications(
    metrics: dict[str, Any], qa_pairs: list[dict[str, Any]]
) -> list[dict[str, str]]:
    by_id = {str(item.get("blocker_id")): item for item in qa_pairs if item.get("blocker_id") is not None}
    by_question = {str(item.get("question", "")).strip(): item for item in qa_pairs}
    accepted = []
    for evaluation in metrics.get("evaluations", []):
        if not evaluation.get("correct"):
            continue
        pair = by_id.get(str(evaluation.get("matched_key_id")))
        if pair is None:
            pair = by_question.get(str(evaluation.get("matched_key_question", "")).strip())
        if pair is None:
            continue
        accepted.append({"question": str(evaluation["question"]), "answer": str(pair.get("answer", ""))})
    return accepted


def _enrich_question_metrics(
    metrics: dict[str, Any],
    question_records: list[dict[str, Any]],
    *,
    total_key_questions: int,
    question_call_count: int,
) -> dict[str, Any]:
    correct = sum(item.get("oracle_status") == "valid" for item in question_records)
    covered = {
        str(item.get("matched_key_id") or item.get("matched_key_question"))
        for item in question_records
        if item.get("oracle_status") == "valid"
        and (item.get("matched_key_id") or item.get("matched_key_question"))
    }
    total_questions = len(question_records)
    recall = len(covered) / total_key_questions if total_key_questions else 0.0
    precision = correct / total_questions if total_questions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    nonempty = int(total_questions > 0)
    valid_batch = int(correct > 0)
    metrics.update(
        {
            "metric_unit": "question_and_nonempty_batch",
            "total_key_questions": total_key_questions,
            "total_predicted_questions": total_questions,
            "correct_questions": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ask_f1": f1,
            "recall_ques": recall,
            "precision_ques": precision,
            "ask_f1_ques": f1,
            "precision_level": valid_batch / nonempty if nonempty else 0.0,
            "covered_key_questions": len(covered),
            "total_generated_questions": total_questions,
            "correct_generated_questions": correct,
            "total_question_calls": question_call_count,
            "nonempty_batches": nonempty,
            "valid_nonempty_batches": valid_batch,
            "empty_batches": question_call_count - nonempty,
            "TKQR": calculate_tkqr(
                [int(item.get("oracle_status") == "valid") for item in question_records] + [0],
                total_key_questions,
            ),
            "ORA": calculate_ora(total_questions, total_key_questions),
        }
    )
    return metrics


def run_clarifygpt(
    task: dict[str, Any],
    *,
    client: Any,
    model: str,
    candidate_count: int,
    seed_input_count: int,
    test_input_count: int,
    timeout_seconds: float,
    demonstrations: int,
    max_tokens: int,
    max_retries: int,
    question_consistency: int,
    question_judge_model: str,
    skip_question_metrics: bool,
    question_only: bool,
    representative_seed: int,
    input_mutation_seed: int,
    on_stage: Any | None = None,
) -> dict[str, Any]:
    problem = str(task["modified_content"])

    def stage(name: str, payload: dict[str, Any]) -> None:
        if on_stage is not None:
            on_stage(name, payload)

    seed_response, seeds = generate_seed_inputs(
        client,
        model=model,
        problem=problem,
        count=seed_input_count,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    inputs = expand_inputs(
        seeds,
        total_count=test_input_count,
        random_seed=input_mutation_seed,
    )
    stage("test_input_generation", {"raw_response": seed_response, "seeds": seeds, "input_cases": inputs})

    candidates = []
    for index in range(1, candidate_count + 1):
        raw_response, code = sample_candidate(
            client,
            model=model,
            problem=problem,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        candidates.append({"candidate_id": f"C{index}", "raw_response": raw_response, "code": code})
        stage("candidate_sampling", {"candidates": candidates})

    executions = execute_candidates(candidates, inputs, timeout_seconds=timeout_seconds)
    groups = cluster_by_output(candidates, executions, random_seed=representative_seed)
    stage("code_consistency", {"executions": executions, "behavior_groups": groups})
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    result: dict[str, Any] = {
        "seed_input_response": seed_response,
        "seed_inputs": seeds,
        "generated_test_inputs": inputs,
        "candidate_solutions": candidates,
        "candidate_executions": executions,
        "behavior_groups": groups,
        "generated_questions": [],
        "question_records": [],
        "question_batches": [],
        "question_response": None,
        "question_metrics": None,
        "validated_clarifications": [],
        "synthesis_response": None,
    }
    if len(groups) == 1:
        final_code = by_id[groups[0]["representative_id"]]["code"]
        if not skip_question_metrics:
            result["question_metrics"] = evaluate_questions(
                client,
                judge_model=question_judge_model,
                generated_questions=[],
                qa_pairs=task.get("qa_pairs", []) or [],
                consistency=question_consistency,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            _enrich_question_metrics(
                result["question_metrics"], [], total_key_questions=len(task.get("qa_pairs", []) or []), question_call_count=0
            )
        result.update(
            {
                "ambiguous": False,
                "stop_reason": "candidate_outputs_unanimous",
                "final_code": final_code,
                "selected_submission": final_code,
                "refined_requirement": problem,
            }
        )
        stage("finalize", result)
        return result

    representative_codes = [by_id[group["representative_id"]]["code"] for group in groups]
    system_prompt, user_prompt = question_prompt(problem, representative_codes, demonstrations=demonstrations)
    question_response = chat_text(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="generate_clarifying_questions",
        include_visible_reasoning=False,
    )
    questions = _questions_from_response(question_response)
    question_records = [
        {"question_id": f"Q_{index}", "question": question}
        for index, question in enumerate(questions, start=1)
    ]
    result.update(
        {
            "ambiguous": True,
            "question_response": question_response,
            "generated_questions": questions,
            "question_records": question_records,
        }
    )
    stage("question_generation", result)

    if skip_question_metrics:
        question_metrics = None
        clarifications = []
    else:
        question_metrics = evaluate_questions(
            client,
            judge_model=question_judge_model,
            generated_questions=questions,
            qa_pairs=task.get("qa_pairs", []) or [],
            consistency=question_consistency,
            max_tokens=max_tokens,
            max_retries=max_retries,
            question_metadata=[{"question_id": item["question_id"]} for item in question_records],
        )
        for question_record, evaluation in zip(question_records, question_metrics["evaluations"], strict=True):
            question_record.update(
                {
                    "correct": bool(evaluation.get("correct")),
                    "duplicate": bool(evaluation.get("duplicate")),
                    "matched_key_question": evaluation.get("matched_key_question"),
                    "matched_key_id": evaluation.get("matched_key_id"),
                }
            )
        clarifications = _accepted_clarifications(question_metrics, task.get("qa_pairs", []) or [])
        answers = {item["question"]: item["answer"] for item in clarifications}
        for question_record in question_records:
            question_record["oracle_status"] = "valid" if question_record["question"] in answers else "invalid"
            question_record["oracle_answer"] = answers.get(question_record["question"])
        _enrich_question_metrics(
            question_metrics,
            question_records,
            total_key_questions=len(task.get("qa_pairs", []) or []),
            question_call_count=1,
        )
        result["question_batches"] = [
            {
                "batch_id": "Q_1",
                "question_count": len(question_records),
                "valid_question_count": sum(item["oracle_status"] == "valid" for item in question_records),
                "invalid_question_count": sum(item["oracle_status"] != "valid" for item in question_records),
                "batch_valid": any(item["oracle_status"] == "valid" for item in question_records),
                "questions": question_records,
                "metrics": dict(question_metrics),
            }
        ]
    result.update({"question_metrics": question_metrics, "validated_clarifications": clarifications})
    stage("oracle", result)

    if question_only:
        result.update(
            {
                "stop_reason": "question_only",
                "final_code": None,
                "selected_submission": None,
                "refined_requirement": problem,
            }
        )
        stage("finalize", result)
        return result

    if not clarifications:
        final_code = by_id[groups[0]["representative_id"]]["code"]
        result.update(
            {
                "stop_reason": "no_valid_clarification",
                "final_code": final_code,
                "selected_submission": final_code,
                "refined_requirement": problem,
            }
        )
        stage("finalize", result)
        return result

    synthesis_system, synthesis_user = synthesis_prompt(problem, clarifications, demonstrations=demonstrations)
    synthesis_response = chat_text(
        client,
        model=model,
        system_prompt=synthesis_system,
        user_prompt=synthesis_user,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="synthesize_solution",
        include_visible_reasoning=False,
    )
    final_code = _clean_code(synthesis_response)
    if not final_code:
        raise RuntimeError("Synthesis response contained no code")
    refined_requirement = problem + "\n\nClarification:\n" + "\n".join(
        f"{index}. {item['question']}\n- {item['answer']}"
        for index, item in enumerate(clarifications, start=1)
    )
    result.update(
        {
            "stop_reason": "synthesized_from_valid_clarifications",
            "synthesis_response": synthesis_response,
            "refined_requirement": refined_requirement,
            "final_code": final_code,
            "selected_submission": final_code,
        }
    )
    stage("finalize", result)
    return result

from __future__ import annotations

import re
from typing import Any

from baselines.common import chat_text, evaluate_questions
from baselines.vanilla.artifacts import attach_iteration_metrics
from benchmarks.ClarifyCodeBench.clarifycodebench.metrics import calculate_ora, calculate_tkqr

from baselines.gptengineer.integrations.clarifycodebench.upstream_prompts import (
    clarification_system_prompt,
    code_generation_system_prompt,
)


def _clean_question(line: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()


def parse_clarification_response(response: str) -> tuple[str, str | None]:
    cleaned = response.strip()
    if cleaned.lower().startswith("no"):
        return "clear", None
    questions = [_clean_question(line) for line in cleaned.splitlines() if "?" in line]
    if questions:
        return "question", questions[-1]
    return "unparseable", None


def _target_key(evaluation: dict[str, Any]) -> str | None:
    if evaluation.get("matched_key_id") is not None:
        return f"id:{evaluation['matched_key_id']}"
    question = str(evaluation.get("matched_key_question") or "").strip()
    return f"question:{question}" if question else None


def _oracle_answer(qa_pairs: list[dict[str, Any]], evaluation: dict[str, Any]) -> str | None:
    target = _target_key(evaluation)
    if target is None:
        return None
    for index, pair in enumerate(qa_pairs, start=1):
        pair_id = pair.get("blocker_id") or pair.get("question_id") or f"GT_{index}"
        if target == f"id:{pair_id}" or target == f"question:{str(pair.get('question', '')).strip()}":
            return str(pair.get("answer", ""))
    return None


def _question_user_prompt(
    problem: str,
    valid_clarifications: list[dict[str, str]],
    previous_questions: list[str],
) -> str:
    sections = [f"User Requirement:\n{problem.strip()}"]
    if valid_clarifications:
        pairs = "\n".join(
            f"{index}. {item['question']}\n- {item['answer']}"
            for index, item in enumerate(valid_clarifications, start=1)
        )
        sections.append(f"Validated clarifications:\n{pairs}")
    if previous_questions:
        sections.append(
            "Questions already asked; do not repeat them:\n"
            + "\n".join(f"- {question}" for question in previous_questions)
        )
    if previous_questions:
        sections.append(
            "Is anything else unclear? If yes, only answer in the form:\n"
            "{remaining unclear areas} remaining questions.\n"
            "{Next question}\n"
            'If everything is sufficiently clear, only answer "no".'
        )
    return "\n\n".join(sections)


def _code_user_prompt(problem: str, clarifications: list[dict[str, str]]) -> str:
    pairs = "\n".join(
        f"{index}. {item['question']}\n- {item['answer']}"
        for index, item in enumerate(clarifications, start=1)
    )
    clarification_section = pairs if pairs else "None."
    return (
        "Instructions: Create exactly one file named solution.py. It must be a complete "
        "Python solution that reads standard input, writes standard output, requires no "
        "additional files, and solves this competitive-programming task.\n\n"
        f"Original competitive-programming requirement:\n{problem.strip()}\n\n"
        f"Validated clarifications:\n{clarification_section}"
    )


def extract_solution_code(response: str) -> str:
    pattern = re.compile(r"(?:^|\n)([^\n`]+\.py)\s*\n```[^\n]*\n(.*?)```", re.DOTALL)
    files = [(path.strip(), content.strip()) for path, content in pattern.findall(response)]
    for path, content in files:
        if path.rsplit("/", 1)[-1] == "solution.py":
            return content
    if files:
        return files[0][1]
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else response).strip()


def _add_interaction_metrics(metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    key_questions = int(metrics.get("total_key_questions", 0))
    hit_sequence = [int(item.get("oracle_status") == "valid") for item in records]
    metrics["TKQR"] = calculate_tkqr(hit_sequence + [0], key_questions)
    metrics["ORA"] = calculate_ora(len(records), key_questions)


def run_gptengineer(
    task: dict[str, Any],
    *,
    client: Any,
    model: str,
    question_budget: int,
    max_tokens: int,
    max_retries: int,
    question_consistency: int,
    question_judge_model: str,
    skip_question_metrics: bool,
    question_only: bool,
    on_stage: Any | None = None,
) -> dict[str, Any]:
    problem = str(task["modified_content"])
    qa_pairs = task.get("qa_pairs", []) or []
    iterations: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    valid_clarifications: list[dict[str, str]] = []
    previous_questions: list[str] = []
    accepted_targets: set[str] = set()
    stop_reason = "question_budget_exhausted"

    def stage(name: str, payload: dict[str, Any]) -> None:
        if on_stage is not None:
            on_stage(name, payload)

    for turn in range(1, question_budget + 1):
        user_prompt = _question_user_prompt(problem, valid_clarifications, previous_questions)
        response = chat_text(
            client,
            model=model,
            system_prompt=clarification_system_prompt(),
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=max_tokens,
            max_retries=max_retries,
            step_name="gptengineer_clarify",
            include_visible_reasoning=False,
        )
        status, question = parse_clarification_response(response)
        iteration: dict[str, Any] = {
            "iteration": turn,
            "iteration_id": f"I{turn}",
            "raw_response": response,
            "parsed_status": status,
            "questions": [] if question is None else [question],
            "valid_clarifications_before": list(valid_clarifications),
            "previous_questions_before": list(previous_questions),
        }
        if status != "question":
            iterations.append(iteration)
            stop_reason = "model_declared_clear" if status == "clear" else "unparseable_question_response"
            stage("clarification", {"iterations": iterations, "stop_reason": stop_reason})
            break

        previous_questions.append(question)
        if skip_question_metrics:
            iteration["oracle_status"] = "skipped"
        else:
            evaluation = evaluate_questions(
                client,
                judge_model=question_judge_model,
                generated_questions=[question],
                qa_pairs=qa_pairs,
                consistency=question_consistency,
                max_tokens=max_tokens,
                max_retries=max_retries,
                question_metadata=[{"question_id": f"I{turn}_Q1"}],
            )["evaluations"][0]
            target = _target_key(evaluation)
            if evaluation.get("correct") and target in accepted_targets:
                evaluation["correct"] = False
                evaluation["duplicate"] = True
            if evaluation.get("correct") and target is not None:
                accepted_targets.add(target)
                answer = _oracle_answer(qa_pairs, evaluation)
                valid_clarifications.append({"question": question, "answer": answer or ""})
                iteration["oracle_status"] = "valid"
                iteration["oracle_answer"] = answer
            else:
                iteration["oracle_status"] = "invalid"
                iteration["oracle_answer"] = None
            evaluations.append(evaluation)
        iterations.append(iteration)
        stage("clarification", {"iterations": iterations, "stop_reason": None})
    else:
        stop_reason = "question_budget_exhausted"

    if skip_question_metrics:
        question_records: list[dict[str, Any]] = []
        question_batches: list[dict[str, Any]] = []
        question_metrics = None
    else:
        question_metrics = attach_iteration_metrics(iterations, evaluations, len(qa_pairs), qa_pairs)
        question_records = question_metrics["evaluations"]
        for record, iteration in zip(question_records, (item for item in iterations if item["questions"]), strict=True):
            record["raw_response"] = iteration["raw_response"]
        _add_interaction_metrics(question_metrics, question_records)
        for iteration in iterations:
            if "metrics" in iteration:
                _add_interaction_metrics(iteration["metrics"], iteration["question_records"])
                iteration["question_batch"]["questions"] = iteration["question_records"]
                iteration["question_batch"]["metrics"] = dict(iteration["metrics"])
        question_batches = [item["question_batch"] for item in iterations if "question_batch" in item]

    result: dict[str, Any] = {
        "original_problem": problem,
        "question_iterations": iterations,
        "generated_questions": previous_questions,
        "question_records": question_records,
        "question_batches": question_batches,
        "question_metrics": question_metrics,
        "validated_clarifications": valid_clarifications,
        "stop_reason": stop_reason,
    }
    stage("oracle", result)
    if question_only:
        result.update({"final_code": None, "selected_submission": None, "synthesis_response": None})
        stage("finalize", result)
        return result

    code_response = chat_text(
        client,
        model=model,
        system_prompt=code_generation_system_prompt(),
        user_prompt=_code_user_prompt(problem, valid_clarifications),
        temperature=0.1,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="gptengineer_generate_code",
        include_visible_reasoning=False,
    )
    final_code = extract_solution_code(code_response)
    if not final_code:
        raise RuntimeError("GPT-Engineer code-generation response contained no code")
    result.update(
        {
            "synthesis_response": code_response,
            "refined_requirement": _code_user_prompt(problem, valid_clarifications),
            "final_code": final_code,
            "selected_submission": final_code,
        }
    )
    stage("finalize", result)
    return result

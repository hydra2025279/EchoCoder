from __future__ import annotations

import re
from typing import Any

from baselines.common import (
    chat_text,
    extract_questions_from_output,
    extract_tag,
)
from echocoder.prompts import (
    build_generate_code_user_prompt,
    get_generate_code_system_prompt,
    get_initial_detection_and_code_system_prompt,
)
from echocoder.schemas import GenerationResult


def _has_nonempty_tag(text: str, tag: str) -> bool:
    match = re.search(
        rf"\[{re.escape(tag)}\](.*?)(?:\[/{re.escape(tag)}\]|$)",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return bool(match and match.group(1).strip())


def _extract_code(text: str) -> str:
    if _has_nonempty_tag(text, "CODE"):
        return extract_tag(text, "CODE")
    matches = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return matches[-1].strip() if matches and matches[-1].strip() else ""


def generate_code(
    client: Any,
    *,
    model: str,
    problem: str,
    task_type: str = "programming",
    max_tokens: int | None,
    max_retries: int,
) -> str:
    system_prompt = get_generate_code_system_prompt(task_type)
    user_prompt = build_generate_code_user_prompt(
        problem=problem,
        task_type=task_type,
        require_first_valid_question=False,
    )

    def call_generation(token_budget: int | None) -> str:
        return chat_text(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=token_budget,
            max_retries=max_retries,
            step_name="generate_code",
        )

    response = call_generation(max_tokens)
    if not _extract_code(response):
        response = call_generation(max_tokens * 2 if max_tokens is not None else None)
    code = _extract_code(response)
    if not code:
        raise RuntimeError("Code-generation response did not contain a [CODE] block")
    return code


def generate_initial_questions_and_code(
    client: Any,
    *,
    model: str,
    problem: str,
    task_type: str = "programming",
    previous_valid_clarifications: list[dict[str, str]] | None = None,
    previous_invalid_questions: list[str] | None = None,
    include_history: bool = True,
    max_tokens: int | None,
    max_retries: int,
) -> tuple[str, str, list[str]]:
    system_prompt = get_initial_detection_and_code_system_prompt(task_type)
    user_prompt = build_generate_code_user_prompt(
        problem=problem,
        task_type=task_type,
        previous_valid_clarifications=previous_valid_clarifications,
        previous_invalid_questions=previous_invalid_questions,
        include_history=include_history,
    )

    def call_generation(token_budget: int | None) -> str:
        return chat_text(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=token_budget,
            max_retries=max_retries,
            step_name="initial_detection_and_generate_code",
        )

    response = call_generation(max_tokens)
    if not _extract_code(response):
        response = call_generation(max_tokens * 2 if max_tokens is not None else None)

    code = _extract_code(response)
    if not code:
        raise RuntimeError("Code-generation response did not contain a [CODE] block")

    initial_output = (
        extract_tag(response, "INITIAL_QUESTIONS")
        if _has_nonempty_tag(response, "INITIAL_QUESTIONS")
        else ""
    )
    initial_questions = extract_questions_from_output("echocoder", initial_output)
    return code, initial_output, initial_questions


def generate_complete_candidate(
    client: Any,
    *,
    model: str,
    specification: str,
    task_type: str,
    previous_valid_clarifications: list[dict[str, str]] | None = None,
    previous_invalid_questions: list[str] | None = None,
    include_history: bool = True,
    max_tokens: int | None,
    max_retries: int,
) -> GenerationResult:
    candidate, raw_response, initial_questions = generate_initial_questions_and_code(
        client,
        model=model,
        problem=specification,
        task_type=task_type,
        previous_valid_clarifications=previous_valid_clarifications,
        previous_invalid_questions=previous_invalid_questions,
        include_history=include_history,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    return GenerationResult(candidate, initial_questions, raw_response)

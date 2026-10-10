from __future__ import annotations

from typing import Any

from baselines.common import chat_text, extract_questions_from_output
from echocoder.prompts import (
    PromptProfile,
    build_detect_hallucination_user_prompt,
    build_detect_hallucination_with_initial_questions_user_prompt,
    get_detect_hallucination_system_prompt,
)
from echocoder.schemas import ReconstructionResult


def detect_hallucination(
    client: Any,
    *,
    model: str,
    original_problem: str,
    reconstructed_problem: str,
    added_points: list[str],
    generated_code: str,
    initial_questions: list[str] | None = None,
    previous_valid_clarifications: list[dict[str, str]] | None = None,
    previous_invalid_questions: list[str] | None = None,
    include_history: bool = True,
    max_tokens: int | None,
    max_retries: int,
    prompt_profile: PromptProfile = "clarifycodebench",
) -> str:
    if initial_questions is None:
        user_prompt = build_detect_hallucination_user_prompt(
            original_problem=original_problem,
            reconstructed_problem=reconstructed_problem,
            added_points=added_points,
            generated_code=generated_code,
            previous_valid_clarifications=previous_valid_clarifications,
        )
    else:
        user_prompt = build_detect_hallucination_with_initial_questions_user_prompt(
            original_problem=original_problem,
            reconstructed_problem=reconstructed_problem,
            added_points=added_points,
            generated_code=generated_code,
            initial_questions=initial_questions,
            previous_valid_clarifications=previous_valid_clarifications,
            previous_invalid_questions=previous_invalid_questions,
            include_history=include_history,
        )

    return chat_text(
        client,
        model=model,
        system_prompt=get_detect_hallucination_system_prompt(
            prompt_profile, with_initial_questions=initial_questions is not None
        ),
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="detect_hallucination",
    )


def generate_questions(
    client: Any,
    *,
    model: str,
    specification: str,
    solution: str,
    reconstruction: ReconstructionResult,
    initial_questions: list[str],
    previous_valid_clarifications: list[dict[str, str]] | None = None,
    previous_invalid_questions: list[str] | None = None,
    include_history: bool = True,
    max_tokens: int | None,
    max_retries: int,
    prompt_profile: PromptProfile = "clarifycodebench",
) -> tuple[str, list[str]]:
    response = detect_hallucination(
        client,
        model=model,
        original_problem=specification,
        reconstructed_problem=reconstruction.description,
        added_points=reconstruction.added_points,
        generated_code=solution,
        initial_questions=initial_questions,
        previous_valid_clarifications=previous_valid_clarifications,
        previous_invalid_questions=previous_invalid_questions,
        include_history=include_history,
        max_tokens=max_tokens,
        max_retries=max_retries,
        prompt_profile=prompt_profile,
    )
    questions = extract_questions_from_output("echocoder", response)
    return response, questions

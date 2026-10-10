from __future__ import annotations

from typing import Any

from baselines.common import chat_text, parse_json_object
from baselines.ablation_reconstruction.prompts import (
    GENERATE_QUESTIONS_SYSTEM_PROMPT,
    build_generate_questions_user_prompt,
)


def generate_questions(
    client: Any,
    *,
    model: str,
    specification: str,
    code: str,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, list[str]]:
    response = chat_text(
        client,
        model=model,
        system_prompt=GENERATE_QUESTIONS_SYSTEM_PROMPT,
        user_prompt=build_generate_questions_user_prompt(
            specification=specification,
            code=code,
        ),
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="generate_questions_from_code",
    )
    parsed = parse_json_object(response, required_keys=("clarifying_questions",))
    if parsed is None or not isinstance(parsed.get("clarifying_questions"), list):
        raise RuntimeError(
            "Question-generation response did not contain a clarifying_questions list"
        )
    questions: list[str] = []
    seen: set[str] = set()
    for value in parsed["clarifying_questions"]:
        question = " ".join(str(value).split()).strip()
        key = question.casefold()
        if question and key not in seen:
            seen.add(key)
            questions.append(question)
    return response, questions

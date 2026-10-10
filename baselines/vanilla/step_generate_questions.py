
from __future__ import annotations

from openai import OpenAI

from baselines.common import chat_text
from baselines.vanilla.prompts import (
    DIRECT_QUESTIONS_SYSTEM_PROMPT,
    build_direct_questions_user_prompt,
)


def generate_questions(
    client: OpenAI,
    *,
    model: str,
    problem: str,
    max_tokens: int,
    max_retries: int,
    step_name: str = "generate_questions",
) -> str:
    return chat_text(
        client,
        model=model,
        system_prompt=DIRECT_QUESTIONS_SYSTEM_PROMPT,
        user_prompt=build_direct_questions_user_prompt(problem=problem),
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name=step_name,
    )

from __future__ import annotations

from typing import Any

from baselines.common import chat_text, extract_tag
from baselines.ablation_reconstruction.prompts import (
    GENERATE_CODE_SYSTEM_PROMPT,
    build_generate_code_user_prompt,
)


def generate_code(
    client: Any,
    *,
    model: str,
    specification: str,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, str]:
    response = chat_text(
        client,
        model=model,
        system_prompt=GENERATE_CODE_SYSTEM_PROMPT,
        user_prompt=build_generate_code_user_prompt(specification=specification),
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="generate_code",
    )
    if "[CODE]" not in response.upper():
        raise RuntimeError("Code-generation response did not contain a [CODE] block")
    code = extract_tag(response, "CODE").strip()
    if not code:
        raise RuntimeError("Code-generation response contained an empty [CODE] block")
    return code, response


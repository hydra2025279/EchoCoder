"""The candidate-sampling stage described in the ClarifyGPT paper."""

from __future__ import annotations

import re
from typing import Any

from baselines.common import chat_text, extract_tag


_SYSTEM_PROMPT = """\
You are given a competitive-programming requirement.

Produce one plausible complete Python program that reads standard input and writes
standard output. Resolve unclear requirements in one concrete, internally consistent
way. Do not ask questions and do not add explanations.

Return:
[CODE]
<complete Python program>
[/CODE]
"""


def _extract_code(response: str) -> str:
    if re.search(r"\[CODE\]", response, flags=re.IGNORECASE):
        return extract_tag(response, "CODE")
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else response.strip()


def sample_candidate(
    client: Any,
    *,
    model: str,
    problem: str,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, str]:
    response = chat_text(
        client,
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=f"User Requirement:\n{problem.strip()}",
        temperature=0.8,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="sample_candidate",
        include_visible_reasoning=False,
    )
    code = _extract_code(response)
    if not code:
        raise RuntimeError("Candidate sampling response contained no code")
    return response, code

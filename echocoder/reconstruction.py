from __future__ import annotations

from typing import Any

from baselines.common import chat_text, parse_json_object
from echocoder.prompts import PromptProfile, build_back_describe_user_prompt, get_back_describe_system_prompt
from echocoder.schemas import ReconstructionResult


def back_describe(
    client: Any,
    *,
    model: str,
    original_problem: str,
    code: str,
    max_tokens: int | None,
    max_retries: int,
    prompt_profile: PromptProfile = "clarifycodebench",
) -> dict[str, object]:
    response = chat_text(
        client,
        model=model,
        system_prompt=get_back_describe_system_prompt(prompt_profile),
        user_prompt=build_back_describe_user_prompt(
            original_problem=original_problem,
            code=code,
        ),
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="back_describe",
    )
    data = parse_json_object(response, required_keys=("description", "added_points"))
    if data is None:
        return {"description": original_problem, "added_points": []}

    description = data.get("description")
    raw_added_points = data.get("added_points")
    if not isinstance(description, str) or not description.strip():
        return {"description": original_problem, "added_points": []}
    if not isinstance(raw_added_points, list):
        return {"description": original_problem, "added_points": []}

    added_points: list[str] = []
    seen: set[str] = set()
    for point in raw_added_points:
        if not isinstance(point, str):
            raise RuntimeError("Every added point must be a string")
        cleaned = " ".join(point.split()).strip()
        key = cleaned.casefold().rstrip(".").strip()
        if cleaned and key not in seen:
            seen.add(key)
            added_points.append(cleaned)

    return {
        "description": description.strip(),
        "added_points": added_points,
    }


def reconstruct_requirements(
    client: Any,
    *,
    model: str,
    specification: str,
    solution: str,
    max_tokens: int | None,
    max_retries: int,
    prompt_profile: PromptProfile = "clarifycodebench",
) -> ReconstructionResult:
    result = back_describe(
        client,
        model=model,
        original_problem=specification,
        code=solution,
        max_tokens=max_tokens,
        max_retries=max_retries,
        prompt_profile=prompt_profile,
    )
    return ReconstructionResult(
        description=str(result["description"]),
        added_points=list(result["added_points"]),
    )

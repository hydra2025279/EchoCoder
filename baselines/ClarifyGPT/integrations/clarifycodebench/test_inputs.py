"""Seed input generation and deterministic input mutation for ClarifyGPT."""

from __future__ import annotations

import random
import re
from typing import Any

from baselines.common import chat_text, parse_json_object


_SEED_INPUT_SYSTEM_PROMPT = """\
You are generating seed standard-input cases for a competitive-programming requirement.
Generate difficult, boundary, and structurally varied valid input cases. Return input
text only; never include expected outputs or explanations.

Example:
Requirement: Read an integer N followed by N integers. Print a result for the sequence.
Output:
{"seed_inputs": ["1\\n0\\n", "3\\n-1 0 1\\n"]}

Example:
Requirement: Read H and W followed by H rows, each containing W integers.
Output:
{"seed_inputs": ["1 1\\n0\\n", "2 3\\n1 0 -1\\n4 5 6\\n"]}

Return valid JSON exactly in this shape:
{"seed_inputs": ["<one complete stdin case>"]}
"""


_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")
_BOOLEAN = re.compile(r"(?<![A-Za-z0-9_])(true|false)(?![A-Za-z0-9_])", re.IGNORECASE)


def generate_seed_inputs(
    client: Any,
    *,
    model: str,
    problem: str,
    count: int,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, list[str]]:
    response = chat_text(
        client,
        model=model,
        system_prompt=_SEED_INPUT_SYSTEM_PROMPT,
        user_prompt=(
            f"User Requirement:\n{problem.strip()}\n\n"
            f"Return at most {count} seed inputs."
        ),
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=max_retries,
        step_name="generate_seed_inputs",
        include_visible_reasoning=False,
    )
    parsed = parse_json_object(response)
    raw_inputs = parsed.get("seed_inputs") if parsed else None
    if not isinstance(raw_inputs, list):
        raise RuntimeError("Seed-input response did not contain a seed_inputs list")
    seeds = [str(item).strip() for item in raw_inputs if str(item).strip()]
    if not seeds:
        raise RuntimeError("Seed-input response contained no non-empty input case")
    return response, seeds[:count]


def mutate_input_case(seed: str, random_generator: random.Random) -> str | None:
    """Apply one type-aware scalar mutation while retaining stdin layout."""
    booleans = list(_BOOLEAN.finditer(seed))
    numbers = list(_NUMBER.finditer(seed))
    if booleans and (not numbers or random_generator.choice((True, False))):
        target = random_generator.choice(booleans)
        token = target.group(0)
        replacement = "false" if token.lower() == "true" else "true"
        return f"{seed[:target.start()]}{replacement}{seed[target.end():]}"
    if not numbers:
        return None
    target = random_generator.choice(numbers)
    token = target.group(0)
    delta = random_generator.choice((-1, 1))
    try:
        if "." in token:
            replacement = str(float(token) + delta)
        else:
            replacement = str(int(token) + delta)
    except ValueError:
        return None
    return f"{seed[:target.start()]}{replacement}{seed[target.end():]}"


def expand_inputs(
    seeds: list[str], *, total_count: int, random_seed: int = 0
) -> list[dict[str, str]]:
    cases = [{"kind": "seed", "input": seed} for seed in seeds]
    seed_pool = list(seeds)
    seen = set(seed_pool)
    random_generator = random.Random(random_seed)
    attempts = 0
    while len(cases) < total_count and seed_pool:
        seed = random_generator.choice(seed_pool)
        mutated = mutate_input_case(seed, random_generator)
        attempts += 1
        if not mutated or mutated in seen:
            if attempts > total_count * max(4, len(seed_pool)):
                break
            continue
        seen.add(mutated)
        seed_pool.append(mutated)
        cases.append({"kind": "mutation", "input": mutated})
    return cases[:total_count]

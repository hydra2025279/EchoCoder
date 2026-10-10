"""Load the unmodified few-shot ClarifyGPT prompts from the upstream clone."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / "src" / "prompt"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream ClarifyGPT prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROMPTS = _load_module("clarifygpt_upstream_prompt_humaneval", _UPSTREAM_ROOT / "prompt_humaneval.py")


def _render_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = ""
    transcript: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system" and not system:
            system = content
        else:
            transcript.append(f"[{role.upper()}]\n{content}")
    return system, "\n\n".join(transcript)


def question_prompt(problem: str, candidates: list[str], *, demonstrations: int) -> tuple[str, str]:
    shot = {0: "zero", 1: "one", 2: "two", 3: "three"}[demonstrations]
    messages = _PROMPTS.askcq_prompt[f"{shot}_shot"]
    system, examples = _render_messages(messages)
    solutions = "\n\n".join(
        f"Solution {index}:\n{candidate}" for index, candidate in enumerate(candidates)
    )
    query = f"User Requirement:\n{problem.strip()}\n\n{solutions}"
    return system, f"{examples}\n\n[USER]\n{query}".strip()


def synthesis_prompt(problem: str, clarifications: list[dict[str, str]], *, demonstrations: int) -> tuple[str, str]:
    shot = {0: "zero", 1: "one", 2: "two", 3: "three"}[demonstrations]
    messages = _PROMPTS.synthesize_prompt[f"{shot}_shot"]
    system, examples = _render_messages(messages)
    clarification_text = "\n".join(
        f"{index}. {item['question']}\n- {item['answer']}"
        for index, item in enumerate(clarifications, start=1)
    )
    query = (
        f"User Requirement:\n{problem.strip()}\n\nClarification:\n"
        f"{clarification_text}"
    )
    return system, f"{examples}\n\n[USER]\n{query}".strip()

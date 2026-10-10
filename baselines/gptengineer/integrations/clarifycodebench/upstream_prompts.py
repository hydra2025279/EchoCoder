from __future__ import annotations

from pathlib import Path


_PREPROMPTS = (
    Path(__file__).resolve().parents[3]
    / "ClarifyGPT"
    / "baseline"
    / "gpt-engineer-main"
    / "gpt_engineer"
    / "preprompts"
)


def load_preprompt(name: str) -> str:
    return (_PREPROMPTS / name).read_text(encoding="utf-8").strip()


def clarification_system_prompt() -> str:
    return load_preprompt("qa")


def code_generation_system_prompt() -> str:
    generate = load_preprompt("generate")
    philosophy = load_preprompt("philosophy")
    use_qa = load_preprompt("use_qa")
    return f"{generate}\nUseful to know:\n{philosophy}\n\n{use_qa}"

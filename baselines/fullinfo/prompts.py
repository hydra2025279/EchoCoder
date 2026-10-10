"""Prompt used by the full-information ClarifyCodeBench control."""

FULL_INFO_CODE_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

You are given a complete competitive-programming problem.

Generate a complete, runnable Python solution that implements the problem.

Return

[CODE]
<complete Python code>
[/CODE]
"""


def build_full_info_code_user_prompt(*, problem: str) -> str:
    return f"""\
Complete problem statement:
{problem}
"""

from __future__ import annotations


GENERATE_CODE_SYSTEM_PROMPT = """\
You are a helpful coding assistant.

Given a competitive-programming problem, generate a complete runnable Python solution.

Return only:

[CODE]
<complete Python code>
[/CODE]
"""


GENERATE_QUESTIONS_SYSTEM_PROMPT = """
You are given the original competitive-programming problem and one generated solution.

Compare the code with the original problem. Find behaviors made by the code that
are not specified in the problem requirements.

For each such case, ask one clarification question about the missing requirement.
If there are no unspecified behaviors that need clarification, return an empty list.

Return only valid JSON in exactly this format:

{"clarifying_questions": ["..."]}
"""

def build_generate_code_user_prompt(*, specification: str) -> str:
    return f"""\
Current problem specification:
{specification}
"""


def build_generate_questions_user_prompt(
    *,
    specification: str,
    code: str,
) -> str:
    return f"""\
Current problem specification:
{specification}

Generated solution:
```
{code}
```
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from echocoder.integrations.base import LocalizedContext

from .models import HilBenchTask


def _relevance(text: str, terms: set[str]) -> int:
    lowered = text.casefold()
    return sum(1 for term in terms if term in lowered)


def _within_budget(items: list[str], budget: int) -> str:
    chunks: list[str] = []
    used = 0
    for item in items:
        remaining = budget - used
        if remaining <= 0:
            break
        chunks.append(item[:remaining])
        used += len(chunks[-1])
    return "\n".join(chunks)


def localize_sql(task: HilBenchTask, specification: str = "", *, max_chars: int = 36000) -> LocalizedContext:
    if task.database_path is None or not task.database_path.exists():
        raise FileNotFoundError(
            f"Database for {task.task_id} is not downloaded. Run the HiL adapter with --download-assets first."
        )
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", specification)
    }
    schema: list[str] = []
    uri = f"file:{task.database_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for name, sql in rows:
            schema.append(f"{name}: {sql}")
    schema.sort(key=lambda item: (-_relevance(item, terms), item))
    descriptions: list[str] = []
    for path in task.table_descriptions:
        descriptions.append(f"[{path.name}]\n{path.read_text(encoding='utf-8', errors='replace')[:6000]}")
    descriptions.sort(key=lambda item: (-_relevance(item, terms), item))
    business_items = [f"- {item}" for item in task.business_info]
    business_items.sort(key=lambda item: (-_relevance(item, terms), item))
    schema_budget = max_chars * 5 // 10
    description_budget = max_chars * 2 // 10
    business_budget = max_chars - schema_budget - description_budget
    content = "\n\n".join(
        [
            "DATABASE SCHEMA\n" + _within_budget(schema, schema_budget),
            "TABLE DESCRIPTIONS\n" + _within_budget(descriptions, description_budget),
            "BUSINESS INFORMATION\n" + _within_budget(business_items, business_budget),
        ]
    )
    return LocalizedContext(
        schema_items=[name for name, _ in rows],
        content=content,
        metadata={"database_path": str(task.database_path), "database_name": task.metadata.get("database_name")},
    )

from __future__ import annotations

from echocoder.integrations.base import LocalizedContext

from ..context import localize_sql
from ..shared.backend import HilBenchBackend


class SqlBackend(HilBenchBackend):
    engine = "sql"

    def localize(
        self, specification: str, specification_round: int
    ) -> LocalizedContext:
        del specification_round
        return localize_sql(self.task, specification)

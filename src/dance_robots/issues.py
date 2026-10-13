"""Problems found while reading / validating the CSV files.

Instead of crashing on the first mistake, the loaders collect every problem as an
`Issue` so the engineer can see them all at once and explain them to the
choreographers in plain language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ERROR = "error"      # the file can NOT be used until fixed
WARNING = "warning"  # usable, but worth a look


@dataclass(frozen=True)
class Issue:
    severity: str  # ERROR or WARNING
    where: str     # e.g. "choreography.csv, line 5"
    message: str   # plain-language description (and suggestion)

    def __str__(self) -> str:
        return f"{self.severity.upper()} ({self.where}): {self.message}"


def has_errors(issues: Iterable[Issue]) -> bool:
    """True if at least one issue is an error."""
    return any(i.severity == ERROR for i in issues)
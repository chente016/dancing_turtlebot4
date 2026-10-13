"""Tiny, forgiving CSV reader shared by the timetable and choreography loaders.

Being forgiving matters because the files come from spreadsheets:
  * lines starting with '#' and blank lines are ignored (handy for comments);
  * header names are case-insensitive and may have spaces around them;
  * the delimiter is auto-detected (',' ';' or tab). A Spanish-locale Excel
    exports ';' with decimal COMMAS ("0,4"), so with ';' we accept those;
  * a UTF-8 byte-order mark (added by Excel) is ignored;
  * every row keeps its ORIGINAL line number for error messages.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .issues import ERROR, WARNING, Issue


@dataclass(frozen=True)
class Row:
    line: int                # 1-based line number in the file
    cells: dict[str, str]    # column name (lower-case) -> stripped text


@dataclass(frozen=True)
class Table:
    name: str                # file name, used in messages
    rows: list[Row]
    decimal_comma: bool      # True if the file uses ';' (so "0,4" means 0.4)


def _is_skippable(line: str) -> bool:
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def read_table(path, required: Sequence[str], optional: Sequence[str] = ()):
    """Read a CSV file. Returns (Table | None, issues).

    The table is None when the file can't be used at all (missing file or
    missing required columns).
    """
    path = Path(path)
    name = path.name
    issues: list[Issue] = []

    try:
        text = path.read_text(encoding="utf-8-sig")  # -sig drops the Excel BOM
    except FileNotFoundError:
        return None, [Issue(ERROR, name, f"file not found: {path}")]
    except UnicodeDecodeError:
        return None, [Issue(ERROR, name, "the file is not UTF-8 text; save it as 'CSV UTF-8'")]

    lines = text.splitlines()
    numbered = [(i, ln) for i, ln in enumerate(lines, start=1) if not _is_skippable(ln)]
    if not numbered:
        return None, [Issue(ERROR, name, "the file is empty")]

    # --- Header: detect the delimiter by counting candidates ------------------
    header_no, header_line = numbered[0]
    delimiter = max([",", ";", "\t"], key=header_line.count)
    if header_line.count(delimiter) == 0:
        delimiter = ","
    header = [c.strip().lower() for c in next(csv.reader([header_line], delimiter=delimiter))]

    missing = [c for c in required if c not in header]
    if missing:
        found = ", ".join(h for h in header if h) or "(none)"
        issues.append(Issue(
            ERROR, f"{name}, line {header_no}",
            f"missing column(s): {', '.join(missing)}. The header must contain "
            f"{', '.join(required)}; found: {found}"))
        return None, issues
    for col in header:
        if col and col not in required and col not in optional:
            issues.append(Issue(WARNING, f"{name}, line {header_no}",
                                f"unknown column '{col}' will be ignored"))

    # --- Data rows ------------------------------------------------------------------
    rows: list[Row] = []
    for line_no, line in numbered[1:]:
        cells = next(csv.reader([line], delimiter=delimiter))
        if all(c.strip() == "" for c in cells):
            continue  # a line like ",,," left over from a spreadsheet
        if len(cells) > len(header):
            issues.append(Issue(ERROR, f"{name}, line {line_no}",
                                f"{len(cells)} values but the header has {len(header)} columns"))
            continue
        cells = list(cells) + [""] * (len(header) - len(cells))
        rows.append(Row(line_no, {h: c.strip() for h, c in zip(header, cells) if h}))

    return Table(name, rows, decimal_comma=(delimiter == ";")), issues


def parse_float(text: str, decimal_comma: bool = False) -> float:
    """Parse a finite number. Raises ValueError with a readable message."""
    raw = text.strip()
    if decimal_comma:
        raw = raw.replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"'{text}' is not a finite number")
    return value


def parse_int(text: str, decimal_comma: bool = False) -> int:
    """Parse a whole number ('40' or '40.0'). Raises ValueError if not whole."""
    value = parse_float(text, decimal_comma)
    if value != int(value):
        raise ValueError(f"'{text}' is not a whole number")
    return int(value)
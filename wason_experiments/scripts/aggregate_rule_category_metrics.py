#!/usr/bin/env python3
"""Aggregate OOD metrics into semantic rule categories.

Expected input layout:
    results/ood_test/<model>/<variant>/summary.txt
    results/ood_test/<model>/<variant>/O1.txt ... O4.txt

The output has one row per model, variant, and category.  Mean metrics are
averaged across the rules in a category (matching the rule-mean aggregation in
summary.txt); test and token totals are summed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Ordering": ("O1_2", "O1_3", "O2_2", "O3_4", "O4_3"),
    "Parity": ("O1_2", "O4_2", "O4_4"),
    "Sign": ("O1_4", "O3_1", "O3_2"),
    "Number-theoretic / Numeric properties": (
        "O1_1",
        "O2_3",
        "O2_4",
        "O3_3",
        "O4_1",
    ),
}

METRIC_COLUMNS = [
    "Task Completion Rate",
    "First Guess Rate",
    "#Tests Before First Correct (solved only)",
    "Incompat:Compat (solved, pre-first-correct)",
    "Incompat:Compat (unsolved, all)",
    "NO:YES (solved, pre-first-correct)",
    "NO:YES (unsolved, all)",
]

COUNT_COLUMNS = [
    "Sum Compatible Tests (solved, pre-first-correct)",
    "Sum Incompatible Tests (solved, pre-first-correct)",
    "Sum Compatible Tests (unsolved, all)",
    "Sum Incompatible Tests (unsolved, all)",
]

FRACTION_COLUMNS = [
    "Avg Incompatibility Fraction (solved, pre-first)",
    "Avg Incompatibility Fraction (unsolved, all)",
]

TOKEN_COLUMNS = [
    "Sum Total Tokens Generated (solved, up-to-correct)",
    "Sum Total Tokens Generated (unsolved, all)",
]

ALL_COLUMNS = METRIC_COLUMNS + COUNT_COLUMNS + FRACTION_COLUMNS + TOKEN_COLUMNS

# Names used in summary.txt's compact per-rule tables.
SUMMARY_ALIASES = {
    "Avg IncompatFrac (solved)": FRACTION_COLUMNS[0],
    "Avg IncompatFrac (unsolved)": FRACTION_COLUMNS[1],
    "Sum TotalTok (solved)": TOKEN_COLUMNS[0],
    "Sum TotalTok (unsolved)": TOKEN_COLUMNS[1],
    "MED:DAX (solved, pre-first-correct)": METRIC_COLUMNS[5],
    "MED:DAX (unsolved, all)": METRIC_COLUMNS[6],
}


def split_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def number(cell: str) -> Optional[float]:
    value = cell.strip().replace(",", "")
    if value in {"", "—", "-", "NA", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def parse_rule_means(summary_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """Read every `O* — Rule Means` table from summary.txt."""
    lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result: Dict[str, Dict[str, Optional[float]]] = {}
    in_rule_section = False
    headers: List[str] = []

    for line in lines:
        if re.match(r"^##\s+O\d+\s+[—-]\s+Rule Means\s*$", line.strip()):
            in_rule_section = True
            headers = []
            continue
        if line.startswith("## "):
            in_rule_section = False
            headers = []
            continue
        if not in_rule_section or not line.strip().startswith("|"):
            continue

        cells = split_row(line)
        if cells and cells[0] == "Rule":
            headers = [SUMMARY_ALIASES.get(h, h) for h in cells]
            continue
        if not headers or cells[0].startswith("---") or len(cells) != len(headers):
            continue

        rule = cells[0]
        if not re.fullmatch(r"O\d+_\d+", rule):
            continue
        result[rule] = {
            header: number(cell) for header, cell in zip(headers[1:], cells[1:])
        }

    return result


def parse_instance_counts(base_dir: Path) -> Dict[str, Dict[str, int]]:
    """Sum compatible/incompatible instance counts for each rule."""
    result: Dict[str, Dict[str, int]] = {}
    for group_path in sorted(base_dir.glob("O[0-9]*.txt")):
        current_rule: Optional[str] = None
        headers: List[str] = []
        for line in group_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^##+\s+(?:Rule\s+)?(O\d+_\d+)\s*$", line.strip())
            if match:
                current_rule = match.group(1)
                headers = []
                continue
            if current_rule is None or not line.strip().startswith("|"):
                continue
            cells = split_row(line)
            if cells and cells[0] == "Instance":
                headers = cells
                continue
            if not headers or cells[0].startswith("---") or len(cells) != len(headers):
                continue

            row = dict(zip(headers, cells))
            solved = row.get("TaskComplete") == "1"
            compat = int(
                number(row.get("CompatCt (window)", row.get("CompatCount", "0")))
                or 0
            )
            incompat = int(
                number(
                    row.get("IncompatCt (window)", row.get("IncompatCount", "0"))
                )
                or 0
            )
            counts = result.setdefault(
                current_rule,
                {
                    COUNT_COLUMNS[0]: 0,
                    COUNT_COLUMNS[1]: 0,
                    COUNT_COLUMNS[2]: 0,
                    COUNT_COLUMNS[3]: 0,
                },
            )
            offset = 0 if solved else 2
            counts[COUNT_COLUMNS[offset]] += compat
            counts[COUNT_COLUMNS[offset + 1]] += incompat
    return result


def fmt(value: Optional[float], *, integer: bool = False) -> str:
    if value is None:
        return "—"
    return str(int(round(value))) if integer else f"{value:.3f}"


def category_metrics(
    rules: Tuple[str, ...],
    rule_means: Dict[str, Dict[str, Optional[float]]],
    rule_counts: Dict[str, Dict[str, int]],
) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for metric in METRIC_COLUMNS + FRACTION_COLUMNS:
        output[metric] = mean(rule_means.get(rule, {}).get(metric) for rule in rules)
    for metric in TOKEN_COLUMNS:
        values = [rule_means.get(rule, {}).get(metric) for rule in rules]
        output[metric] = sum(value for value in values if value is not None)
    for metric in COUNT_COLUMNS:
        output[metric] = float(
            sum(rule_counts.get(rule, {}).get(metric, 0) for rule in rules)
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate OOD summary metrics by semantic rule category."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/ood_test"),
        help="Directory containing <model>/<variant>/summary.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: <root>/category_summary.txt)",
    )
    args = parser.parse_args()

    root = args.root
    output = args.output or root / "category_summary.txt"
    rows = []
    for summary_path in sorted(root.glob("*/*/summary.txt")):
        model, variant = summary_path.parent.parent.name, summary_path.parent.name
        rule_means = parse_rule_means(summary_path)
        rule_counts = parse_instance_counts(summary_path.parent)
        for category, rules in CATEGORIES.items():
            rows.append(
                (model, variant, category, category_metrics(rules, rule_means, rule_counts))
            )

    lines = ["# OOD Metrics by Rule Category", ""]
    lines.append("| Model | Variant | Category | " + " | ".join(ALL_COLUMNS) + " |")
    lines.append("|---|---|---|" + "|".join("---" for _ in ALL_COLUMNS) + "|")
    integer_columns = set(COUNT_COLUMNS + TOKEN_COLUMNS)
    for model, variant, category, metrics in rows:
        values = [
            fmt(metrics.get(column), integer=column in integer_columns)
            for column in ALL_COLUMNS
        ]
        lines.append(
            "| " + " | ".join([model, variant, category, *values]) + " |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

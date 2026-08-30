#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path("results/ood_test")

# Canonical metric names we want as columns
METRIC_COLUMNS = [
    "Task Completion Rate",
    "First Guess Rate",
    "#Tests Before First Correct (solved only)",
    "Incompat:Compat (solved, pre-first-correct)",
    "Incompat:Compat (unsolved, all)",
    "NO:YES (solved, pre-first-correct)",
    "NO:YES (unsolved, all)",
]

# Existing add-ons (some are sums, some are averages)
NEW_METRIC_COLUMNS = [
    "Sum Compatible Tests (solved, pre-first-correct)",
    "Sum Incompatible Tests (solved, pre-first-correct)",
    "Sum Compatible Tests (unsolved, all)",
    "Sum Incompatible Tests (unsolved, all)",
    "Avg Incompatibility Fraction (solved, pre-first)",
    "Avg Incompatibility Fraction (unsolved, all)",
]

# ---- NEW: token sum metrics (these are SUMS in summary.txt) ----
TOKEN_SUM_COLUMNS = [
    "Sum Total Tokens Generated (solved, up-to-correct)",
    "Sum Total Tokens Generated (unsolved, all)",
]

ALL_COLUMNS = METRIC_COLUMNS + NEW_METRIC_COLUMNS + TOKEN_SUM_COLUMNS


def parse_overall_means(summary_path: Path) -> Dict[str, str]:
    """
    Parse the 'Overall Means (across all rules)' table from a summary.txt.
    Return dict mapping metric_name -> value_string.
    """
    text = summary_path.read_text(encoding="utf-8").splitlines()

    metrics: Dict[str, str] = {}
    in_overall = False
    in_table = False

    for line in text:
        if line.strip().startswith("## Overall Means (across all rules)"):
            in_overall = True
            in_table = False
            continue

        if not in_overall:
            continue

        # Look for the header of the overall table
        if line.strip().startswith("| Metric |"):
            in_table = True
            continue

        if in_table:
            # Table separator or empty line ends the table
            if re.match(r"^\s*\|[-: ]+\|\s*$", line) or not line.strip():
                if not line.strip():
                    break
                else:
                    continue

            # Parse a data row: | Metric | Value |
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 3:
                continue
            metric_name = parts[1]
            val = parts[2]
            metrics[metric_name] = val

    return metrics


def extract_model_variant(path: Path) -> Tuple[str, str]:
    """
    Given .../results_sft/ood_test/<model>/<variant>/summary.txt,
    return (model, variant).
    """
    variant_dir = path.parent
    model_dir = variant_dir.parent
    return model_dir.name, variant_dir.name


def main():
    rows = []

    # Find all summary.txt files under results_sft/ood_test/*/*
    for summary_path in ROOT.glob("*/*/summary.txt"):
        model, variant = extract_model_variant(summary_path)
        metrics_raw = parse_overall_means(summary_path)

        def get_metric(name: str) -> str:
            # Normalize NO:YES metrics: fall back to MED:DAX if needed (dual-goal)
            if name == "NO:YES (solved, pre-first-correct)":
                return metrics_raw.get(
                    "NO:YES (solved, pre-first-correct)",
                    metrics_raw.get("MED:DAX (solved, pre-first-correct)", "—"),
                )
            if name == "NO:YES (unsolved, all)":
                return metrics_raw.get(
                    "NO:YES (unsolved, all)",
                    metrics_raw.get("MED:DAX (unsolved, all)", "—"),
                )

            # First guess: be robust to older/alternate labels if any
            if name == "First Guess Rate":
                return metrics_raw.get(
                    "First Guess Rate",
                    metrics_raw.get("FirstGuess (correct & before tests)", "—"),
                )

            # Fractions: if you ever prefer the "global over episodes" rows, fall back to those
            if name == "Avg Incompatibility Fraction (solved, pre-first)":
                return metrics_raw.get(
                    name,
                    metrics_raw.get("Avg Incompatibility Fraction (solved, pre-first) — global over episodes", "—"),
                )
            if name == "Avg Incompatibility Fraction (unsolved, all)":
                return metrics_raw.get(
                    name,
                    metrics_raw.get("Avg Incompatibility Fraction (unsolved, all) — global over episodes", "—"),
                )

            # Everything else: direct lookup, default to em dash
            return metrics_raw.get(name, "—")

        row = {
            "model": model,
            "variant": variant,
        }
        for col in ALL_COLUMNS:
            row[col] = get_metric(col)
        rows.append(row)

    # Sort rows for readability: by model, then variant
    rows.sort(key=lambda r: (r["model"], r["variant"]))

    # Write overall_summary.txt as a markdown table
    out_path = ROOT / "overall_summary.txt"
    lines = []
    lines.append("# Overall Summary — ood_test\n")
    header = (
        "| Model | Variant | "
        + " | ".join(ALL_COLUMNS)
        + " |"
    )
    separator = "|---|---|" + "|".join(["---"] * len(ALL_COLUMNS)) + "|"

    lines.append(header)
    lines.append(separator)

    for r in rows:
        vals = [r["model"], r["variant"]] + [r[m] for m in ALL_COLUMNS]
        lines.append("| " + " | ".join(vals) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
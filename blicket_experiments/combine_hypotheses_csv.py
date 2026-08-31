#!/usr/bin/env python3
"""
combine_hypotheses_csvs.py

Recursively walk an experiment root folder (e.g., exp_output_alt_oracle/),
find per-trial hypothesis CSVs produced from action_log_trial-<i>_hypotheses.csv,
and combine them into ONE CSV with extra columns:

- model
- rule          (Conjunctive / Disjunctive)
- num_objects   (4 or 8)
- trial_num     (0..15)
- step          (-1..45)  [from the per-trial CSV]
- object_state  (e.g., "[false,true,...]" or similar)
- machine_on
- num_valid_hypotheses

Usage:
  python combine_hypotheses_csvs.py \
    --root /scratch/aj4332/cb_env/blicket-text-llm/exp_output_alt_oracle \
    --out  /scratch/aj4332/cb_env/blicket-text-llm/results/combined_hypotheses.csv

Notes:
- We infer model, num_objects, and rule from the path pattern:
    <root>/<model>/obj_<K>/<rule>_<H>tests/...
- We also attempt a fallback inference from parent folder names if needed.
- This script is robust to the “duplicated obj_4/conjunctive_45tests” nesting you mentioned.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

LOG = logging.getLogger("combine_hypotheses")

# Matches trial file name
TRIAL_CSV_RE = re.compile(r"action_log_trial-(\d+)_hypotheses\.csv$", re.IGNORECASE)

# Matches tokens in directory names
OBJ_RE = re.compile(r"^obj_(\d+)$", re.IGNORECASE)
RULEDIR_RE = re.compile(r"^(conjunctive|disjunctive)_(\d+)tests$", re.IGNORECASE)


def infer_meta_from_path(p: Path, root: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Infer (model, num_objects, rule) from path relative to root.
    Expect something like:
      root/model/obj_4/disjunctive_45tests/.../action_log_trial-0_hypotheses.csv
    """
    try:
        rel = p.relative_to(root)
    except Exception:
        rel = p

    parts = list(rel.parts)

    model = None
    num_objects = None
    rule = None

    # model is typically the first part under root
    if len(parts) >= 1:
        model = parts[0]

    # scan for obj_K and rule folder tokens anywhere in the path
    for part in parts:
        m = OBJ_RE.match(part)
        if m and num_objects is None:
            num_objects = int(m.group(1))
        m2 = RULEDIR_RE.match(part)
        if m2 and rule is None:
            rule = m2.group(1).lower()

    return model, num_objects, rule


def normalize_rule(rule: Optional[str]) -> Optional[str]:
    if rule is None:
        return None
    r = rule.strip().lower()
    if r.startswith("conj"):
        return "Conjunctive"
    if r.startswith("disj"):
        return "Disjunctive"
    return rule


def find_trial_csvs(root: Path) -> List[Path]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if TRIAL_CSV_RE.match(fn):
                out.append(Path(dirpath) / fn)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Experiment root folder containing model folders.")
    ap.add_argument("--out", required=True, help="Output combined CSV path.")
    ap.add_argument("--loglevel", default="INFO", help="INFO/DEBUG/WARNING")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trial_csvs = find_trial_csvs(root)
    if not trial_csvs:
        LOG.error(f"No '*_hypotheses.csv' found under: {root}")
        raise SystemExit(2)

    LOG.info(f"Found {len(trial_csvs)} hypothesis CSVs under {root}")

    frames = []
    bad = 0

    for csv_path in trial_csvs:
        m = TRIAL_CSV_RE.search(csv_path.name)
        if not m:
            continue
        trial_num = int(m.group(1))

        model, num_objects, rule = infer_meta_from_path(csv_path, root)
        rule_norm = normalize_rule(rule)

        if model is None or num_objects is None or rule_norm is None:
            LOG.warning(f"Could not fully infer metadata for: {csv_path}")
            bad += 1
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            LOG.warning(f"Failed reading {csv_path}: {e}")
            bad += 1
            continue

        # Expected columns from per-trial script:
        # step, object_state, machine_on, num_valid_hypotheses
        # But we won't assume; we keep whatever is present.
        df.insert(0, "trial_num", trial_num)
        df.insert(0, "num_objects", num_objects)
        df.insert(0, "Rule", rule_norm)
        df.insert(0, "model", model)

        frames.append(df)

        if len(frames) % 50 == 0:
            LOG.info(f"Loaded {len(frames)} / {len(trial_csvs)}")

    if not frames:
        LOG.error("No CSVs were successfully loaded (all failed metadata inference or parsing).")
        raise SystemExit(2)

    combined = pd.concat(frames, ignore_index=True)

    # Optional: stable sort if step exists
    sort_cols = [c for c in ["model", "num_objects", "Rule", "trial_num", "step"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    combined.to_csv(out_path, index=False)
    LOG.info(f"Wrote combined CSV: {out_path}")
    if bad:
        LOG.warning(f"Skipped {bad} files due to inference/read failures.")


if __name__ == "__main__":
    main()

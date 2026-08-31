#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect per-trial blicket experiment results from exp_output/**/results.jsonl.

Outputs:
- A printed table (sorted)
- A CSV file (default: all_trials_results.csv)

Each row = ONE TRIAL.

Expected run layout (Hydra):
exp_output/<model_name>/<rule>_<steps>/results.jsonl
exp_output/<model_name>/<rule>_<steps>/.hydra/config.yaml

Hydra config is used to read:
- env_kwargs.num_objects
- env_kwargs.rule
- max_actions_per_trial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed parsing JSON on {path}:{ln}: {e}") from e
    return rows


def read_hydra_cfg(run_dir: Path) -> Optional[dict]:
    if yaml is None:
        return None
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def infer_model_name(run_dir: Path, exp_root: Path) -> str:
    rel = run_dir.relative_to(exp_root)
    return rel.parts[0] if rel.parts else "UNKNOWN_MODEL"


# -----------------------------------------------------
# Core logic
# -----------------------------------------------------
def collect_one_run(exp_root: Path, run_dir: Path) -> List[Dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return []

    rows = safe_read_jsonl(results_path)
    if not rows:
        return []

    df = pd.DataFrame(rows)

    required = ["trial_idx", "unique_state_visited", "num_correct"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(
                f"{results_path} missing required field '{c}'. "
                f"Found cols: {list(df.columns)}"
            )

    cfg = read_hydra_cfg(run_dir)
    if cfg is None:
        raise RuntimeError(f"Missing Hydra config at {run_dir}/.hydra/config.yaml")

    num_objects = cfg.get("env_kwargs", {}).get("num_objects", None)
    rule = cfg.get("env_kwargs", {}).get("rule", None)
    steps = cfg.get("max_actions_per_trial", None)

    if num_objects is None or steps is None or rule is None:
        raise RuntimeError(
            f"Incomplete Hydra config in {run_dir}/.hydra/config.yaml "
            f"(need env_kwargs.num_objects, env_kwargs.rule, max_actions_per_trial)"
        )

    model_name = infer_model_name(run_dir, exp_root)

    records: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        rec = {
            "model": model_name,
            "rule": rule,
            "num_objects": int(num_objects),
            "steps": int(steps),
            "trial_idx": int(r["trial_idx"]),
            "num_correct": int(r["num_correct"]),
            "all_correct": int(r["num_correct"] == int(num_objects)),
            "unique_state_visited": int(r["unique_state_visited"]),
        }
        records.append(rec)

    return records


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp_root",
        type=str,
        default="exp_output",
        help="Root experiment directory",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="all_trials_results.csv",
        help="Output CSV filename",
    )
    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    if not exp_root.exists():
        raise SystemExit(f"exp_root not found: {exp_root}")

    run_dirs = sorted({p.parent for p in exp_root.rglob("results.jsonl")})

    all_records: List[Dict[str, Any]] = []
    for rd in run_dirs:
        all_records.extend(collect_one_run(exp_root, rd))

    if not all_records:
        raise SystemExit(f"No trials found under {exp_root}")

    df = pd.DataFrame(all_records)

    df = df.sort_values(
        ["num_objects", "rule", "steps", "model", "trial_idx"],
        kind="stable",
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))

    out_path = Path(args.out_csv).resolve()
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

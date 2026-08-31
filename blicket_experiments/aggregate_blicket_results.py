#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate blicket experiment results from exp_output/**/results.jsonl.

Outputs:
- A printed table (sorted)
- A CSV file (default: aggregated_results.csv)

Expected run layout (Hydra):
exp_output/<model_name>/<rule>_<steps>/results.jsonl
exp_output/<model_name>/<rule>_<steps>/.hydra/config.yaml   (used to read env_kwargs.num_objects and max_actions_per_trial)

Each results.jsonl should contain per-trial rows with fields like:
trial_idx, unique_state_visited, num_steps, num_correct, ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None


def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def parse_rule_steps(run_dir_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Expect run_dir_name like: 'conjunctive_32' or 'disjunctive_8'
    (Used as a fallback only; we prefer Hydra config for steps.)
    """
    if "_" not in run_dir_name:
        return None, None
    rule, steps_str = run_dir_name.rsplit("_", 1)
    try:
        steps = int(steps_str)
    except ValueError:
        steps = None
    return rule, steps


def _read_hydra_cfg(run_dir: Path) -> Optional[dict]:
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


def read_num_objects_from_hydra(run_dir: Path) -> Optional[int]:
    """
    Read env_kwargs.num_objects from run_dir/.hydra/config.yaml
    """
    cfg = _read_hydra_cfg(run_dir)
    if not cfg:
        return None
    num_objects = cfg.get("env_kwargs", {}).get("num_objects", None)
    try:
        return int(num_objects) if num_objects is not None else None
    except Exception:
        return None


def read_max_steps_from_hydra(run_dir: Path) -> Optional[int]:
    """
    Read max_actions_per_trial from run_dir/.hydra/config.yaml
    This is the true 'steps budget' and is what we want in the CSV.
    """
    cfg = _read_hydra_cfg(run_dir)
    if not cfg:
        return None
    steps = cfg.get("max_actions_per_trial", None)
    try:
        return int(steps) if steps is not None else None
    except Exception:
        return None


def infer_model_name(run_dir: Path, exp_root: Path) -> str:
    """
    With exp_output/<model>/<rule_steps>/results.jsonl,
    model_name is the directory directly under exp_root.
    """
    rel = run_dir.relative_to(exp_root)
    parts = rel.parts
    return parts[0] if parts else "UNKNOWN_MODEL"


def aggregate_one_run(exp_root: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return None

    model_name = infer_model_name(run_dir, exp_root)

    # Prefer Hydra for rule/steps/num_objects; fallback to dirname parsing for rule
    rule_from_name, _ = parse_rule_steps(run_dir.name)
    num_objects = read_num_objects_from_hydra(run_dir)
    steps_budget = read_max_steps_from_hydra(run_dir)

    rows = safe_read_jsonl(results_path)
    if not rows:
        return None

    df = pd.DataFrame(rows)

    required = ["trial_idx", "unique_state_visited", "num_correct"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(
                f"{results_path} missing required field '{c}'. Found cols: {list(df.columns)}"
            )

    # Determine rule: prefer what we can infer from dirname; if absent, try hydra env_kwargs.rule
    rule = rule_from_name
    if rule is None:
        cfg = _read_hydra_cfg(run_dir) or {}
        rule = cfg.get("env_kwargs", {}).get("rule", None)

    # Determine steps: MUST be the max_actions_per_trial (steps budget)
    steps = steps_budget
    if steps is None:
        # last-resort fallback: dirname parsing
        _, steps_from_name = parse_rule_steps(run_dir.name)
        steps = steps_from_name

    # Determine num_objects: if hydra missing, attempt to read from results rows (rare)
    if num_objects is None:
        if "num_objects" in df.columns:
            try:
                num_objects = int(df["num_objects"].iloc[0])
            except Exception:
                num_objects = None

    if num_objects is None:
        raise RuntimeError(
            f"Could not determine num_objects for run_dir={run_dir}. "
            f"Expected Hydra at {run_dir}/.hydra/config.yaml to contain env_kwargs.num_objects."
        )

    if steps is None:
        raise RuntimeError(
            f"Could not determine steps (max_actions_per_trial) for run_dir={run_dir}. "
            f"Expected Hydra at {run_dir}/.hydra/config.yaml to contain max_actions_per_trial."
        )

    # Per-trial derived metric
    df["all_correct"] = (df["num_correct"] == int(num_objects)).astype(int)

    out: Dict[str, Any] = {
        "model": "thinkopp-" + model_name,
        "num_objects": int(num_objects),
        "rule": rule,
        "steps": int(steps),  # <-- budget, from Hydra max_actions_per_trial
        "n_trials": int(df.shape[0]),
        "avg_num_correct": float(df["num_correct"].mean()),
        "avg_all_correct": float(df["all_correct"].mean()),
        "avg_unique_state_visited": float(df["unique_state_visited"].mean()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", type=str, default="exp_output_thinkopp", help="Root exp directory (default: exp_output)")
    ap.add_argument("--out_csv", type=str, default="aggregated_results.csv", help="Output CSV filename")
    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    if not exp_root.exists():
        raise SystemExit(f"exp_root not found: {exp_root}")

    run_dirs = sorted({p.parent for p in exp_root.rglob("results.jsonl")})

    records: List[Dict[str, Any]] = []
    for rd in run_dirs:
        rec = aggregate_one_run(exp_root, rd)
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit(f"No runs found (no results.jsonl under {exp_root})")

    out_df = pd.DataFrame(records)

    sort_cols = [c for c in ["num_objects", "rule", "steps", "model"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols, kind="stable")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(out_df.to_string(index=False))

    out_path = Path(args.out_csv).resolve()
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

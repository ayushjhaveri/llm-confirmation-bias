#!/usr/bin/env python3
"""
compute_hypotheses_csv.py

Adds detailed logging for:
- discovery of trial files
- hypothesis elimination per step
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -----------------------------
# Logging
# -----------------------------

logging.basicConfig(
    level=logging.INFO,   # change to DEBUG if you want more
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger(__name__)

# -----------------------------
# Regex helpers
# -----------------------------

OBJ_LOC_RE = re.compile(r"\bobject\s+(\d+)\s+is\s+(on top of the machine|on the floor)\b", re.IGNORECASE)
LIGHT_ON_RE = re.compile(r"\blight on the machine is (now on|currently off)\b", re.IGNORECASE)

TEST_LIST_RE = re.compile(r"(?:^|\b)Test:\s*\[(.*?)\]\s*$", re.IGNORECASE)
BRACKET_LIST_RE = re.compile(r"^\s*\[(.*?)\]\s*$")
OBJ_TOKEN_RE = re.compile(r"\bobject\s+(\d+)\b", re.IGNORECASE)

FEEDBACK_ONOFF_RE = re.compile(r"\bFeedback:\s*(ON|OFF)\b", re.IGNORECASE)
ONOFF_ALONE_RE = re.compile(r"^\s*(ON|OFF)\s*$")

TRIAL_FILE_RE = re.compile(r"action_log_trial-(\d+)\.jsonl$")

# -----------------------------
# Hypotheses
# -----------------------------

RuleType = str  # "conjunctive" | "disjunctive"

def compute_feedback_from_truth(true_rule: str, blicket_indices: List[int], test_objs: List[bool]) -> bool:
    # Convert test bools -> set of indices
    test_set = {i for i, on in enumerate(test_objs) if on}
    blicket_set = set(int(x) for x in blicket_indices)

    if true_rule == "disjunctive":
        return len(test_set & blicket_set) > 0
    elif true_rule == "conjunctive":
        return blicket_set.issubset(test_set)
    else:
        raise ValueError(f"Unknown true_rule: {true_rule}")


def eval_rule(rule: RuleType, x: int, mask: int) -> bool:
    if rule == "disjunctive":
        return (x & mask) != 0
    elif rule == "conjunctive":
        return (x & mask) == mask
    raise ValueError(rule)

def init_hypotheses(n: int):
    LOG.debug(f"Initializing hypotheses for {n} objects")
    hyps = []
    for mask in range(1, (1 << n)):
        hyps.append((mask, "conjunctive"))
        hyps.append((mask, "disjunctive"))
    return hyps

def filter_hypotheses(hyps, obs):
    before = len(hyps)
    out = []
    for mask, rule in hyps:
        if all(eval_rule(rule, x, mask) == y for x, y in obs):
            out.append((mask, rule))
    LOG.debug(f"Hypotheses filtered: {before} → {len(out)}")
    return out

# -----------------------------
# Parsing helpers
# -----------------------------

def bools_to_mask(bools):
    return sum((1 << i) for i, b in enumerate(bools) if b)

def parse_initial_configuration(text, n):
    on_machine = [False] * n
    for m in OBJ_LOC_RE.finditer(text):
        idx = int(m.group(1))
        on_machine[idx] = "on top" in m.group(2).lower()

    m = LIGHT_ON_RE.search(text)
    if not m:
        raise ValueError("Could not parse machine light state")
    machine_on = m.group(1).lower() == "now on"

    return on_machine, machine_on

def parse_test_objects(action, n):
    s = action.strip()
    inner = None
    m = TEST_LIST_RE.match(s)
    if m:
        inner = m.group(1)
    else:
        m2 = BRACKET_LIST_RE.match(s)
        if m2:
            inner = m2.group(1)
    if inner is None:
        raise ValueError(f"Unparseable test: {action}")

    objs = [False] * n
    for g in OBJ_TOKEN_RE.findall(inner):
        idx = int(g)
        objs[idx] = True
    return objs

def parse_feedback(rec, raw):
    for src in (
        rec.get("game_state", {}).get("feedback"),
        rec.get("feedback"),
        raw,
    ):
        if not isinstance(src, str):
            continue
        m = FEEDBACK_ONOFF_RE.search(src)
        if m:
            return m.group(1) == "ON"
        m2 = ONOFF_ALONE_RE.match(src.strip())
        if m2:
            return m2.group(1) == "ON"
    return None

# -----------------------------
# Core logic
# -----------------------------

@dataclass
class StepRow:
    step: int
    objects_on_machine: List[bool]
    machine_on: Optional[bool]
    num_valid_hypotheses: int

def process_trial(path: str, max_steps: int):
    LOG.info(f"Processing trial: {path}")

    records, raw_lines = [], []
    with open(path) as f:
        for ln in f:
            raw_lines.append(ln)
            records.append(json.loads(ln))

    gs0 = records[0]["game_state"]
    n = len(gs0["object_names"])

    LOG.info(f"Detected num_objects={n}")

    hyps = init_hypotheses(n)
    obs = []

    rows = []

    # step -1
    rows.append(StepRow(-1, [False]*n, None, len(hyps)))
    LOG.info(f"Step -1: hypotheses={len(hyps)}")

    # step 0 (initial config)
    init_conf = gs0.get("initial_configuration") or gs0.get("feedback")
    objs0, on0 = parse_initial_configuration(init_conf, n)
    obs.append((bools_to_mask(objs0), on0))
    hyps = filter_hypotheses(hyps, obs)

    rows.append(StepRow(0, objs0, on0, len(hyps)))
    LOG.info(f"Step 0: machine={'ON' if on0 else 'OFF'}, hypotheses={len(hyps)}")

    # test steps
    step = 0
    for rec, raw in zip(records, raw_lines):
        if rec.get("turn_type") != "test":
            continue
        if step >= max_steps:
            break

        try:
            objs = parse_test_objects(rec["action"], n)
        except Exception as e:
            LOG.warning(f"Skipping malformed test: {e}")
            continue

        gs = rec.get("game_state", {})
        true_rule = gs.get("true_rule")
        blicket_indices = gs.get("blicket_indices")

        if true_rule is None or blicket_indices is None:
            LOG.warning("Missing true_rule/blicket_indices in game_state, skipping test")
            continue

        fb = compute_feedback_from_truth(true_rule, blicket_indices, objs)

        step += 1
        obs.append((bools_to_mask(objs), fb))
        prev = len(hyps)
        hyps = filter_hypotheses(hyps, obs)

        rows.append(StepRow(step, objs, fb, len(hyps)))
        LOG.info(
            f"Step {step}: test={objs}, feedback={'ON' if fb else 'OFF'}, "
            f"hypotheses {prev}→{len(hyps)}"
        )

    return rows

def write_csv(rows, out_path):
    LOG.info(f"Writing CSV: {out_path}")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "objects_on_machine", "machine_on", "num_valid_hypotheses"])
        for r in rows:
            obj_str = "[" + ",".join("1" if b else "0" for b in r.objects_on_machine) + "]"
            m = "" if r.machine_on is None else ("ON" if r.machine_on else "OFF")
            w.writerow([r.step, obj_str, m, r.num_valid_hypotheses])

def find_trials(root):
    hits = []
    for d, _, files in os.walk(root):
        for fn in files:
            if TRIAL_FILE_RE.match(fn):
                hits.append(os.path.join(d, fn))
    LOG.info(f"Found {len(hits)} trial files")
    return sorted(hits)

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max_steps", type=int, default=45)
    args = ap.parse_args()

    for path in find_trials(args.root):
        out = path.replace(".jsonl", "_hypotheses.csv")
        try:
            rows = process_trial(path, args.max_steps)
            write_csv(rows, out)
        except Exception as e:
            LOG.error(f"Failed on {path}: {e}", exc_info=True)

    LOG.info("All done.")

if __name__ == "__main__":
    main()

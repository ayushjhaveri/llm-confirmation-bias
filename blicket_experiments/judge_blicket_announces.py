#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from judge_llama33_70b_vllm import Llama33_70B_Judge, GenParams

OBJ_RE = re.compile(r"\bobject\s+(\d+)\b", re.IGNORECASE)

def find_jsonl_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*.jsonl"):
        if p.name.endswith("_judge_guess.jsonl"):
            continue
        out.append(p)
    return out

def safe_read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                # Skip malformed lines, but keep going
                print(f"[WARN] Failed to parse {path} line {line_no}: {e}")
    return rows

def infer_true_rule_from_dir(dirpath: Path) -> Optional[str]:
    # Prefer reading results.jsonl (written by your runner)
    results = dirpath / "results.jsonl"
    if results.exists():
        rows = safe_read_jsonl(results)
        # Try to find a row that has env_rule_setting
        for r in rows:
            v = r.get("env_rule_setting", None)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()

    # Fallback: infer from directory path name
    s = str(dirpath).lower()
    if "conjunctive" in s:
        return "conjunctive"
    if "disjunctive" in s:
        return "disjunctive"
    if "xor" in s:
        return "xor"
    return None

def get_truth_from_turn(row: dict, true_rule: Optional[str]) -> Tuple[List[str], str]:
    gs = row.get("game_state", {}) or {}
    obj_names = gs.get("object_names", []) or []
    blicket_indices = gs.get("blicket_indices", []) or []

    # Normalize indices to int
    bi: List[int] = []
    for x in blicket_indices:
        try:
            bi.append(int(x))
        except Exception:
            pass

    # Map to names (prefer index->name if possible)
    true_blickets: List[str] = []
    if obj_names and bi:
        for i in bi:
            if 0 <= i < len(obj_names):
                true_blickets.append(str(obj_names[i]))
            else:
                true_blickets.append(f"object {i}")
    else:
        # fallback: just "object i"
        true_blickets = [f"object {i}" for i in bi]

    # True rule must be known
    rule = (true_rule or "").strip().lower()
    return sorted(set(true_blickets)), rule

def extract_guessed_blickets(announce_text: str) -> List[str]:
    idxs = set()
    for m in OBJ_RE.finditer(announce_text or ""):
        try:
            idxs.add(int(m.group(1)))
        except Exception:
            pass
    return [f"object {i}" for i in sorted(idxs)]

def build_judge_prompt(
    announce_text: str,
    true_blickets: List[str],
    true_rule: str,
) -> List[Dict[str, str]]:
    """
    Judge returns ONLY one token: True or False (case-insensitive).
    True iff BOTH:
      (1) Blicket set is exactly correct
      (2) Rule description matches the true rule semantics
    """
    # Provide tight rubric and forbid extra output.
    system = (
        "You are a strict evaluator for a causal-discovery task (Blicket game). "
        "You must output ONLY 'True' or 'False' on a single line."
    )

    # Semantics to enforce:
    # - disjunctive: machine ON if ANY blicket on machine
    # - conjunctive: machine ON if ALL blickets on machine
    # Also allow the model to mention blickets via object list or by describing objects directly.
    user = (
        "Decide whether the agent's ANNOUNCEMENT is COMPLETELY CORRECT.\n\n"
        "Ground truth:\n"
        f"- True relevant objects: {true_blickets}\n"
        f"- True rule type: {true_rule}\n\n"
        "Rule semantics:\n"
        "- If true_rule_type is 'disjunctive', the device turns on iff at least ONE of the relevant objects is on the device.\n"
        "- If true_rule_type is 'conjunctive', the device turns on iff ALL of the relevant objects are on the mdeviceachine.\n\n"
        "- If true_rule_type is 'xor', the device turns on iff EXACTLY ONE of the relevant objects is on the device (not zero, not two+).\n"
        "Agent announcement (verbatim):\n"
        f"{announce_text}\n\n"
        "Mark True ONLY IF BOTH conditions hold:\n"
        "1) The set of guessed relevant objects matches the true relevant objects exactly (same members; order doesn't matter; no extras; no missing).\n"
        "   - The agent may say 'relevant objects are ...' OR directly name the correct objects; both are acceptable.\n"
        "2) The stated rule matches the true rule semantics exactly.\n"
        "   - If the agent states OR/ANY/AT LEAST ONE when the true rule is conjunctive, that is False.\n"
        "   - If the agent states AND/ALL/NEEDS BOTH when the true rule is disjunctive, that is False.\n"
        "   - If the agent states XOR/EXACTLY ONE/A OR B BUT NOT BOTH/ONE AND ONLY ONE when the true rule is xor, that is True\n"
        "   - If the agent states XOR/EXACTLY ONE when the true rule is xor, that is True\n"
        "   - If the agent states relevant object A on and relevant object B off, but does not specify the other direction of relevant object A off and relevant object B on when the true rule is xor, that is False\n"
        "   - If the agent states odd number of relevant objects on, when the number of relevant objects is 3 or more and when the true rule is xor, that is False\n"
        "If there is any ambiguity, missing detail, partial correctness, or mismatch, output False.\n\n"
        "Output exactly one word: True or False."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def judge_announce(
    judge: Llama33_70B_Judge,
    announce_text: str,
    true_blickets: List[str],
    true_rule: str,
    params: GenParams,
) -> bool:
    prompt = build_judge_prompt(announce_text, true_blickets, true_rule)
    out = judge.generate(prompt, params=params).strip()
    # Be strict: only accept exact True/False leading token
    first = (out.split()[:1] or [""])[0].lower()
    if first == "true":
        return True
    if first == "false":
        return False
    # If judge rambles, treat as incorrect (strict)
    return False

def process_file(
    path: Path,
    judge: Llama33_70B_Judge,
    params: GenParams,
    dry_run: bool = False,
) -> int:
    rows = safe_read_jsonl(path)
    if not rows:
        return 0

    # We only judge files that look like action logs (contain turn_type + announce)
    has_turn_type = any(("turn_type" in r) for r in rows)
    if not has_turn_type:
        return 0

    out_path = path.with_name(path.stem + "_judge_guess.jsonl")

    true_rule = infer_true_rule_from_dir(path.parent)
    # If still unknown, we can’t judge rule semantics reliably.
    if true_rule not in ("conjunctive", "disjunctive", "xor"):
        print(f"[WARN] Could not infer true rule for {path} (dir={path.parent}). Skipping.")
        return 0

    judged = 0
    if dry_run:
        print(f"[DRY] Would write: {out_path}")
        return 0

    with out_path.open("w", encoding="utf-8") as wf:
        for r in rows:
            if r.get("turn_type") != "announce":
                continue

            # Announce content: prefer response_message_stripped if present, else action
            announce_text = r.get("response_message_stripped") or r.get("action") or ""
            announce_text = str(announce_text)

            true_blickets, rule = get_truth_from_turn(r, true_rule=true_rule)
            guessed_blickets = extract_guessed_blickets(announce_text)

            num_guessed = len(guessed_blickets)
            num_correct = len(set(guessed_blickets).intersection(set(true_blickets)))

            rule_correct = judge_announce(
                judge=judge,
                announce_text=announce_text,
                true_blickets=true_blickets,
                true_rule=rule,
                params=params,
            )

            out_row = {
                "trial_idx": r.get("trial_idx"),
                "turns": r.get("turns"),
                "announce_text": announce_text,
                "rule_correct": bool(rule_correct),
                "num_blickets_guessed": int(num_guessed),
                "num_blickets_correct": int(num_correct),

                # helpful debug (remove if you want)
                "true_blickets": true_blickets,
                "guessed_blickets": guessed_blickets,
                "true_rule": rule,
                "source_file": str(path),
            }
            wf.write(json.dumps(out_row) + "\n")
            judged += 1

    return judged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to recursively search for *.jsonl logs")
    ap.add_argument("--model_path", default="/scratch/aj4332/models/llama-3.3-70b-instruct")
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (70B likely needs 2+ GPUs)")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem_util", type=float, default=0.98)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    judge = Llama33_70B_Judge(
        model_path=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=False,
    )

    params = GenParams(
        temperature=0.0,
        top_p=1.0,
        top_k=20,
        max_new_tokens=64,   # judge output is tiny
        repetition_penalty=1.0,
        stop=None,
    )

    root = Path(args.root).expanduser().resolve()
    files = find_jsonl_files(root)

    total_judged = 0
    total_files = 0

    for p in files:
        n = process_file(p, judge=judge, params=params, dry_run=args.dry_run)
        if n > 0 or args.dry_run:
            total_files += 1
            total_judged += n
            print(f"[OK] {p} -> judged {n} announce turns")

    print(f"Done. Files processed: {total_files}. Total announce turns judged: {total_judged}.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build SFT JSONL datasets from your saved rollouts.

This version DOES NOT TRIM "correct conversations":
- We still *compute* the first CORRECT step from *_judge_guesses.jsonl (stored in sample meta),
  but we do NOT filter turns based on it.
- We extract ONE sample per TEST step ("Check: ...") across the whole rollout.

We replace the initial think-in-opposites instruction prompt with the BASELINE prompt
(but keep the rest of the conversation history exactly, using `content` only).
Targets use `raw` exactly as stored.

Output format (JSONL):
{
  "id": "...",
  "split": "train|iid_val|ood_val",
  "rule_id": "T1_1",
  "inst": 17,
  "step": 42,
  "messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...],
  "completion": "</think> ... Check: [..,..,..]"
}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterable


RULES_16 = [
    "T1_1","T1_2","T1_3","T1_4",
    "T2_1","T2_2","T2_3","T2_4",
    "T3_1","T3_2","T3_3","T3_4",
    "T4_1","T4_2","T4_3","T4_4",
]

_THINK_OPEN_RE = re.compile(r"^\s*<think>\s*", re.DOTALL)

def normalize_completion_for_slash_think(raw: str) -> str:
    s = raw or ""
    # If it starts with <think>, strip ONLY the opening tag (not the closing tag)
    return _THINK_OPEN_RE.sub("", s, count=1)

def baseline_initial_prompt(seed: Tuple[int, int, int]) -> str:
    a, b, c = seed
    return (
        "You will be given three numbers which conform to a simple rule that I have in mind.\n"
        "This rule is concerned with the common attributes or relations between any three numbers,\n"
        "and not with their absolute magnitude – for example, it is not a rule like \"all numbers above (or below) 50.\"\n\n"
        "Your goal is to discover this hidden rule by proposing sets of three numbers.\n"
        "You will then be told if the triple conforms with the hidden rule or not.\n\n"
        "At each turn, you may perform one of two actions:\n\n"
        "- Test: propose a new set of three integers, denoted [a,b,c]. You will then be told whether that triple conforms with the hidden rule (\"YES\" or \"NO\").\n"
        "- Announce: based on all previous tets, state what you believe the hidden rule is. You will not receive any feedback for announcements.\n\n"
        "Each instruction you receive will specify what to do.\n\n"
        "Format (must follow exactly):\n"
        "- If the instruction is Turn – Announce, output exactly one line:\n"
        "  Announce: <one short sentence naming the rule>\n\n"
        "- If the instruction is Turn – Test, output exactly one line:\n"
        "  Check: [a,b,c]\n\n"
        f"A triple that conforms with the hidden rule is: [{a}, {b}, {c}].\n"
        "Let's begin.\n"
        "Turn – Announce."
    )


@dataclass
class Transcript:
    meta: Dict[str, Any]
    turns: List[Dict[str, Any]]  # each has step, role, content, raw?


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_transcript(jsonl_path: Path) -> Transcript:
    rows = read_jsonl(jsonl_path)
    if not rows or "meta" not in rows[0]:
        raise ValueError(f"Bad transcript JSONL (missing meta): {jsonl_path}")
    meta = rows[0]["meta"]
    turns = rows[1:]
    return Transcript(meta=meta, turns=turns)


def find_first_correct_step(judge_path: Path) -> Optional[int]:
    """
    Returns the `step` of the first line where outcome == "CORRECT".
    If none, returns None.
    """
    if not judge_path.exists():
        return None
    rows = read_jsonl(judge_path)
    best: Optional[int] = None
    for r in rows:
        if str(r.get("outcome", "")).upper() == "CORRECT":
            s = r.get("step", None)
            if isinstance(s, int):
                best = s if best is None else min(best, s)
    return best


def is_test_model_turn(turn: Dict[str, Any]) -> bool:
    if turn.get("role") != "model":
        return False
    c = (turn.get("content") or "").strip()
    return c.startswith("Check:")


def role_map(turn_role: str) -> str:
    # - environment turns as user messages
    # - model turns as assistant messages
    return "user" if turn_role == "environment" else "assistant"


def iter_instance_dirs(rule_dir: Path, inst_lo: int, inst_hi: int) -> Iterable[Tuple[int, Path]]:
    for i in range(inst_lo, inst_hi + 1):
        p = rule_dir / f"inst{i}"
        if p.is_dir():
            yield i, p


def pick_transcript_file(inst_dir: Path) -> Path:
    # choose the transcript jsonl (not judge guesses)
    jsonls = sorted(inst_dir.glob("*.jsonl"))
    jsonls = [p for p in jsonls if "_judge_guesses" not in p.name]
    if not jsonls:
        raise FileNotFoundError(f"No transcript jsonl in {inst_dir}")
    return sorted(jsonls)[-1]


def corresponding_judge_file(transcript_jsonl: Path) -> Path:
    return transcript_jsonl.with_name(transcript_jsonl.stem + "_judge_guesses.jsonl")


def build_samples_from_instance(
    transcript_path: Path,
    split_name: str,
    add_think_token_at_prompt_end: bool,
) -> List[Dict[str, Any]]:
    tr = load_transcript(transcript_path)
    meta = tr.meta
    rule_id = meta.get("testcase_id", "UNKNOWN")
    inst = int(meta.get("instance_index", -1))

    seed_ex = meta.get("seed_example", {})
    seed = (int(seed_ex["a"]), int(seed_ex["b"]), int(seed_ex["c"]))

    judge_path = corresponding_judge_file(transcript_path)
    cut_step = find_first_correct_step(judge_path)  # computed but NOT used to filter anything

    base_prompt = baseline_initial_prompt(seed)
    samples: List[Dict[str, Any]] = []

    turns = tr.turns

    # Drop the original first environment message (initial instruction) if present.
    start_idx = 0
    if turns and turns[0].get("role") == "environment":
        start_idx = 1

    # Include ALL Check: turns across the full rollout (no trimming by cut_step).
    for idx in range(start_idx, len(turns)):
        t = turns[idx]
        step = t.get("step", None)
        if not isinstance(step, int):
            continue
        if not is_test_model_turn(t):
            continue

        completion = t.get("raw") or ""
        if not isinstance(completion, str) or completion.strip() == "":
            completion = str(t.get("content") or "")

        # Build message history up to (but excluding) this model turn
        msgs: List[Dict[str, str]] = [{"role": "user", "content": base_prompt}]
        for j in range(start_idx, idx):
            tj = turns[j]
            r = tj.get("role")
            c = tj.get("content")
            if r not in ("environment", "model"):
                continue
            if not isinstance(c, str):
                c = str(c)
            msgs.append({"role": role_map(r), "content": c})

        completion = normalize_completion_for_slash_think(completion)

        sample = {
            "id": f"{split_name}/{rule_id}/inst{inst}/step_{step}",
            "split": split_name,
            "rule_id": rule_id,
            "inst": inst,
            "step": step,
            "messages": msgs,
            "completion": completion,
            "meta": {
                "cut_step_first_correct": cut_step,
                "transcript": str(transcript_path),
                "add_think_token_at_prompt_end": bool(add_think_token_at_prompt_end),
            },
        }
        samples.append(sample)

    return samples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True, type=Path,
                    help="e.g. /scratch/.../runs/train/qwen3-8b/think-in-opposites")
    ap.add_argument("--iid-root", required=True, type=Path,
                    help="e.g. /scratch/.../runs/iid/qwen3-8b/think-in-opposites")
    ap.add_argument("--ood-root", required=True, type=Path,
                    help="e.g. /scratch/.../runs/ood_val/qwen3-8b/think-in-opposites")
    ap.add_argument("--outdir", required=True, type=Path, help="where to write train/val jsonl")
    ap.add_argument("--train-inst-max", type=int, default=100, help="use inst1..this for train")
    ap.add_argument("--rules16", action="store_true", help="use the 16 T-rules list")
    args = ap.parse_args()

    rules = RULES_16 if args.rules16 else sorted([p.name for p in args.train_root.iterdir() if p.is_dir()])

    all_train: List[Dict[str, Any]] = []
    all_iid: List[Dict[str, Any]] = []
    all_ood: List[Dict[str, Any]] = []

    # ---- train ----
    for r in rules:
        rule_dir = args.train_root / r
        if not rule_dir.is_dir():
            print(f"[WARN] Missing train rule dir: {rule_dir}")
            continue
        for inst_i, inst_dir in iter_instance_dirs(rule_dir, 1, args.train_inst_max):
            try:
                tr_path = pick_transcript_file(inst_dir)
                ss = build_samples_from_instance(tr_path, split_name="train", add_think_token_at_prompt_end=True)
                all_train.extend(ss)
            except Exception as e:
                print(f"[WARN] Failed {rule_dir}/inst{inst_i}: {e}")

    # ---- iid val ---- (your layout: each rule has inst181)
    for r in rules:
        rule_dir = args.iid_root / r
        if not rule_dir.is_dir():
            print(f"[WARN] Missing iid rule dir: {rule_dir}")
            continue
        inst_dir = rule_dir / "inst181"
        if not inst_dir.is_dir():
            print(f"[WARN] Missing iid inst181: {inst_dir}")
            continue
        try:
            tr_path = pick_transcript_file(inst_dir)
            ss = build_samples_from_instance(tr_path, split_name="iid_val", add_think_token_at_prompt_end=True)
            all_iid.extend(ss)
        except Exception as e:
            print(f"[WARN] Failed iid {inst_dir}: {e}")

    # ---- ood val ---- (your layout: 8 val rules, inst1 + inst2)
    for rule_dir in sorted([p for p in args.ood_root.iterdir() if p.is_dir()]):
        r = rule_dir.name
        for inst_i, inst_dir in iter_instance_dirs(rule_dir, 1, 2):
            try:
                tr_path = pick_transcript_file(inst_dir)
                ss = build_samples_from_instance(tr_path, split_name="ood_val", add_think_token_at_prompt_end=True)
                all_ood.extend(ss)
            except Exception as e:
                print(f"[WARN] Failed ood {rule_dir}/inst{inst_i}: {e}")

    out_train = args.outdir / "train.jsonl"
    out_iid   = args.outdir / "iid_val.jsonl"
    out_ood   = args.outdir / "ood_val.jsonl"

    write_jsonl(out_train, all_train)
    write_jsonl(out_iid, all_iid)
    write_jsonl(out_ood, all_ood)

    print("Wrote:")
    print(" ", out_train, "n=", len(all_train))
    print(" ", out_iid,   "n=", len(all_iid))
    print(" ", out_ood,   "n=", len(all_ood))


if __name__ == "__main__":
    main()

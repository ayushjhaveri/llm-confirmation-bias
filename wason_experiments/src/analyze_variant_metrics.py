#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze a variant directory and emit per-group tables + summary.txt.

Usage:
  python -m src.analyze_variant_metrics \
      --expdir runs/train/llama-3.3-70b-instruct/baseline \
      --outdir results

Original metrics preserved:
- Task Completion Rate
- #Tests Before First Correct (solved only)
- Incompat:Compat (solved, pre-first-correct)
- Incompat:Compat (unsolved, all)
- NO:YES or MED:DAX ratios
- AvgThinkTok (solved/unsolved)
- First Guess Rate

ADDED (previous request):
(1) Global SUM metrics (no averaging; summed over all instances):
    - Sum Compatible Tests (solved, pre-first-correct)
    - Sum Incompatible Tests (solved, pre-first-correct)
    - Sum Compatible Tests (unsolved, all)
    - Sum Incompatible Tests (unsolved, all)

(2) Episode-level incompatibility fraction:
    Incompatibility Fraction = incompatible / (compatible + incompatible)
    - Avg Incompatibility Fraction (solved, pre-first-correct): per episode then mean
    - Avg Incompatibility Fraction (unsolved, all): per episode then mean

NEW (this request):
(3) Total Tokens generated:
    - Total Tokens generated (solved): for solved episodes, sum token count over model turns'
      rec['raw'] for all model turns up to and INCLUDING the first correct announcement step.
    - Total Tokens generated (unsolved): for unsolved episodes, sum token count over model turns'
      rec['raw'] for all model turns.

Aggregation:
- Per rule: SUM (not average) of total tokens over the relevant episodes.
- Summary: includes per-rule sums, plus global sums across all episodes.

Important:
- Any compat rows with judge/compiler error (status not OK / OK_AFTER_RUNTIME_RETRY)
  are excluded from compat/incompat counts, totals, and fractions.
"""

from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ----------------------------- Regexes -----------------------------

ENV_FEEDBACK_RE = re.compile(r"^(YES|NO)\s*(?:\r?\n|$)", re.IGNORECASE | re.MULTILINE)
ENV_MED_DAX_RE = re.compile(r"^(MED|DAX)\s*(?:\r?\n|$)", re.IGNORECASE | re.MULTILINE)

# Announce (used for solved step detection via judge_guesses.jsonl only)
ANNOUNCE_LINE_RE = re.compile(r"(?mi)^\s*Announce:\s*(.+?)\s*$")

# Dual-Goal announce (kept for completeness; not used for MED:DAX)
ANNOUNCE_DAX_RE  = re.compile(r"(?mi)^\s*Announce:\s*DAX\s*rule\s*-\s*(.+?)\s*$")
ANNOUNCE_MED_RE  = re.compile(r"(?mi)^\s*Announce:\s*MED\s*rule\s*-\s*(.+?)\s*$")

# Tests
CHECK_LINE_RE    = re.compile(r"(?mi)^\s*Check:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$")
INSTR_MED_RE     = re.compile(r"(?i)\bMED\b")
INSTR_DAX_RE     = re.compile(r"(?i)\bDAX\b")

# Thinking blocks
_THINK_PAIR_RE = re.compile(r"(?is)<\s*think\b[^>]*>(.*?)</\s*think\s*>")
_WS_TOKEN_RE = re.compile(r"\S+")

# ----------------------------- I/O helpers -----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def find_three_files(inst_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Choose the latest base transcript .jsonl (first line has {"meta": ...}),
    and its sidecars *_judge_guesses.jsonl and *_judge_compatibility.jsonl.
    """
    base_candidates: List[Path] = []
    for p in sorted(inst_dir.glob("*.jsonl")):
        n = p.name
        if n.endswith("_judge_guesses.jsonl") or n.endswith("_judge_compatibility.jsonl"):
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            if first.startswith('{"meta"'):
                base_candidates.append(p)
        except Exception:
            continue

    if not base_candidates:
        return None, None, None

    base = sorted(base_candidates)[-1]   # filenames carry timestamps
    stem = base.stem
    guesses = base.with_name(f"{stem}_judge_guesses.jsonl")
    compat  = base.with_name(f"{stem}_judge_compatibility.jsonl")
    return base, (guesses if guesses.exists() else None), (compat if compat.exists() else None)

# ----------------------------- Parsing helpers -----------------------------

def extract_variant(meta: Dict[str, Any]) -> str:
    return str(meta.get("variant", "")).lower().strip()

def first_correct_announce_step(guesses_doc: List[Dict[str, Any]]) -> Optional[int]:
    steps = [r.get("step") for r in guesses_doc
             if isinstance(r, dict) and r.get("outcome") == "CORRECT"]
    if not steps:
        return None
    return min(int(s) for s in steps if isinstance(s, int) or (isinstance(s, float) and not math.isnan(s)))

def first_judged_guess_step_and_outcome(
    guesses_doc: List[Dict[str, Any]]
) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (min_step, outcome_at_min_step) considering only judged announcements:
    outcomes in {CORRECT, INCORRECT}. Ignores UNPARSEABLE.
    """
    candidates: List[Tuple[int, str]] = []
    for r in guesses_doc:
        if not isinstance(r, dict):
            continue
        out = r.get("outcome")
        if out not in ("CORRECT", "INCORRECT"):
            continue
        st = r.get("step")
        if isinstance(st, (int, float)) and not (isinstance(st, float) and math.isnan(st)):
            candidates.append((int(st), out))
    if not candidates:
        return None, None
    st_min, out_min = min(candidates, key=lambda x: x[0])
    return st_min, out_min

def tests_before_first_correct(step: int) -> Optional[int]:
    if step is None:
        return None
    # Under alternating protocol, this is an integer:
    return int((step - 2) // 4)

def collect_env_no_yes(transcript_doc: List[Dict[str, Any]], *, up_to_step: Optional[int]) -> Tuple[int, int]:
    """
    Count NO/YES tokens from environment messages.
    If up_to_step is set, only consider environment messages with step < up_to_step.
    """
    no = yes = 0
    for rec in transcript_doc:
        if not isinstance(rec, dict) or rec.get("role") != "environment":
            continue
        st = rec.get("step")
        if up_to_step is not None and (not isinstance(st, int) or st >= up_to_step):
            continue
        content = rec.get("content") or ""
        m = ENV_FEEDBACK_RE.search(content)
        if not m:
            continue
        tok = m.group(1).upper()
        if tok == "NO":
            no += 1
        elif tok == "YES":
            yes += 1
    return no, yes

def collect_med_dax_env(transcript_doc: List[Dict[str, Any]], *, up_to_step: Optional[int]) -> Tuple[int, int]:
    """
    Dual-Goal: count MED vs DAX environment feedback tokens.
    """
    med = dax = 0
    for rec in transcript_doc:
        if not isinstance(rec, dict) or rec.get("role") != "environment":
            continue
        st = rec.get("step")
        if up_to_step is not None and (not isinstance(st, int) or st >= up_to_step):
            continue
        content = rec.get("content") or ""
        m = ENV_MED_DAX_RE.search(content)
        if not m:
            continue
        tok = m.group(1).upper()
        if tok == "MED":
            med += 1
        elif tok == "DAX":
            dax += 1
    return med, dax

def compat_counts_pre_or_all(
    compat_doc: List[Dict[str, Any]],
    *,
    pre_first_correct_step: Optional[int]
) -> Tuple[int, int, List[Tuple[Optional[int], str]]]:
    """
    Returns (incompatible_count, compatible_count, issues).

    IMPORTANT: rows with status not OK / OK_AFTER_RUNTIME_RETRY are NOT counted.
    """
    incompatible = compatible = 0
    issues: List[Tuple[Optional[int], str]] = []
    for rec in compat_doc:
        if not isinstance(rec, dict):
            continue
        if "meta" in rec:
            continue
        status = rec.get("status")
        if status != "OK" and status != "OK_AFTER_RUNTIME_RETRY":
            issues.append((rec.get("index"), status))
            continue
        chk = rec.get("check_step")
        if pre_first_correct_step is not None and (not isinstance(chk, int) or chk >= pre_first_correct_step):
            continue
        comp = rec.get("compatible")
        if comp is True:
            compatible += 1
        elif comp is False:
            incompatible += 1
    return incompatible, compatible, issues

def ratio(numer: int, denom: int) -> Optional[float]:
    return None if denom == 0 else (numer / float(denom))

def fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"

def fmt_int(x: Optional[int]) -> str:
    return "—" if x is None else str(x)

def fmt_int0(x: int) -> str:
    return str(int(x))

# ----------------------------- Thinking tokens -----------------------------

def _count_think_tokens_in_raw(raw: str) -> int:
    if not raw:
        return 0

    matches = list(_THINK_PAIR_RE.finditer(raw))
    if matches:
        return sum(len(_WS_TOKEN_RE.findall(m.group(1) or "")) for m in matches)

    if "</think>" in raw.lower():
        before = raw.lower().split("</think>", 1)[0]
        return len(_WS_TOKEN_RE.findall(before))

    return 0

def avg_think_tokens(
    transcript_doc: List[Dict[str, Any]],
    *,
    up_to_step: Optional[int],
) -> Tuple[Optional[float], Tuple[int, int]]:
    """
    Average thinking tokens per model turn, computed from rec['raw'].
    Only includes model turns with >=1 think token.
    """
    s = 0
    n = 0
    for rec in transcript_doc:
        if not isinstance(rec, dict) or rec.get("role") != "model":
            continue
        st = rec.get("step")
        if up_to_step is not None and (not isinstance(st, int) or st >= up_to_step):
            continue
        raw = rec.get("raw")
        if not isinstance(raw, str):
            continue
        ct = _count_think_tokens_in_raw(raw)
        if ct <= 0:
            continue
        s += ct
        n += 1
    return (None if n == 0 else (s / float(n))), (s, n)

# ----------------------------- Total tokens (NEW) -----------------------------

def _count_total_tokens_in_raw(raw: str) -> int:
    """
    Token count proxy from rec['raw'] by whitespace tokenization.
    (Matches how think tokens are counted: regex over \\S+)
    """
    if not isinstance(raw, str) or not raw:
        return 0
    return len(_WS_TOKEN_RE.findall(raw))

def total_tokens_generated(
    transcript_doc: List[Dict[str, Any]],
    *,
    up_to_step_inclusive: Optional[int],
) -> int:
    """
    Sum token counts over model turns' rec['raw'].

    If up_to_step_inclusive is not None:
      include model turns with step <= up_to_step_inclusive.
    Else:
      include all model turns.
    """
    total = 0
    for rec in transcript_doc:
        if not isinstance(rec, dict) or rec.get("role") != "model":
            continue
        st = rec.get("step")
        if up_to_step_inclusive is not None:
            if not isinstance(st, int) or st > up_to_step_inclusive:
                continue
        raw = rec.get("raw")
        total += _count_total_tokens_in_raw(raw if isinstance(raw, str) else "")
    return int(total)

# ----------------------------- Metrics core -----------------------------

def analyze_instance(inst_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    transcript, guesses, compat = find_three_files(inst_dir)
    has_tr = transcript is not None
    has_gu = guesses is not None
    has_co = compat is not None

    if not has_tr or not has_gu or not has_co:
        issues.append(
            f"MISSING_FILES: {inst_dir.relative_to(inst_dir.parents[1])} "
            f"(transcript={has_tr}, guesses={has_gu}, compat={has_co})"
        )

    doc_tr = read_jsonl(transcript) if transcript else []
    doc_gu = read_jsonl(guesses) if guesses else []
    doc_co = read_jsonl(compat) if compat else []

    meta = {}
    if doc_tr and isinstance(doc_tr[0], dict) and "meta" in doc_tr[0]:
        meta = doc_tr[0]["meta"]

    variant = extract_variant(meta)
    rule_id = str(meta.get("testcase_id", "")).strip() or inst_dir.parent.name
    inst_name = inst_dir.name

    step_first_correct = first_correct_announce_step(doc_gu) if doc_gu else None
    task_complete = 1 if step_first_correct is not None else 0
    tests_before = tests_before_first_correct(step_first_correct) if step_first_correct is not None else None

    # First-guess correctness
    first_guess_step, first_guess_outcome = first_judged_guess_step_and_outcome(doc_gu) if doc_gu else (None, None)
    first_guess_correct = 1 if (first_guess_outcome == "CORRECT") else 0
    first_guess_correct_no_tests = 1 if (
        first_guess_outcome == "CORRECT"
        and step_first_correct is not None
        and first_guess_step == step_first_correct
        and tests_before == 0
    ) else 0

    # Compatibility counts + additions (DO NOT change existing I:C fields)
    if task_complete:
        i_pre, c_pre, compat_issues_pre = compat_counts_pre_or_all(
            doc_co, pre_first_correct_step=step_first_correct
        )
        ic_ratio_solved_pre = ratio(i_pre, c_pre)
        ic_ratio_unsolved_all = None
        raw_ic_solved_pre = (i_pre, c_pre)
        raw_ic_unsolved_all = (0, 0)

        # Added: counts for sums (solved window)
        compat_ct_solved_pre = c_pre
        incompat_ct_solved_pre = i_pre
        compat_ct_unsolved_all = 0
        incompat_ct_unsolved_all = 0

        # Added: incompatibility fraction (solved window)
        denom = i_pre + c_pre
        incompat_frac_solved_pre = None if denom == 0 else (i_pre / float(denom))
        incompat_frac_unsolved_all = None

        for idx, st in compat_issues_pre:
            issues.append(f"COMPAT_STATUS_NOT_OK: {rule_id}/{inst_name} index={idx} status={st}")
    else:
        i_all, c_all, compat_issues_all = compat_counts_pre_or_all(doc_co, pre_first_correct_step=None)
        ic_ratio_solved_pre = None
        ic_ratio_unsolved_all = ratio(i_all, c_all)
        raw_ic_solved_pre = (0, 0)
        raw_ic_unsolved_all = (i_all, c_all)

        # Added: counts for sums (unsolved window)
        compat_ct_solved_pre = 0
        incompat_ct_solved_pre = 0
        compat_ct_unsolved_all = c_all
        incompat_ct_unsolved_all = i_all

        # Added: incompatibility fraction (unsolved window)
        denom = i_all + c_all
        incompat_frac_solved_pre = None
        incompat_frac_unsolved_all = None if denom == 0 else (i_all / float(denom))

        for idx, st in compat_issues_all:
            issues.append(f"COMPAT_STATUS_NOT_OK: {rule_id}/{inst_name} index={idx} status={st}")

    # NO:YES or MED:DAX depending on variant
    ny_ratio_solved_pre = ny_ratio_unsolved_all = None
    md_ratio_solved_pre = md_ratio_unsolved_all = None
    raw_no_yes_solved_pre = raw_no_yes_unsolved_all = (0, 0)
    raw_med_dax_solved_pre = raw_med_dax_unsolved_all = (0, 0)

    if variant == "dual-goal":
        if task_complete:
            med_pre, dax_pre = collect_med_dax_env(doc_tr, up_to_step=step_first_correct)
            md_ratio_solved_pre = ratio(med_pre, dax_pre)
            raw_med_dax_solved_pre = (med_pre, dax_pre)
        else:
            med_all, dax_all = collect_med_dax_env(doc_tr, up_to_step=None)
            md_ratio_unsolved_all = ratio(med_all, dax_all)
            raw_med_dax_unsolved_all = (med_all, dax_all)
    else:
        if task_complete:
            no_pre, yes_pre = collect_env_no_yes(doc_tr, up_to_step=step_first_correct)
            ny_ratio_solved_pre = ratio(no_pre, yes_pre)
            raw_no_yes_solved_pre = (no_pre, yes_pre)
        else:
            no_all, yes_all = collect_env_no_yes(doc_tr, up_to_step=None)
            ny_ratio_unsolved_all = ratio(no_all, yes_all)
            raw_no_yes_unsolved_all = (no_all, yes_all)

    # Thinking tokens
    if task_complete:
        avg_think_solved_pre, raw_think_solved_pre = avg_think_tokens(doc_tr, up_to_step=step_first_correct)
        avg_think_unsolved_all = None
        raw_think_unsolved_all = (0, 0)
    else:
        avg_think_solved_pre = None
        raw_think_solved_pre = (0, 0)
        avg_think_unsolved_all, raw_think_unsolved_all = avg_think_tokens(doc_tr, up_to_step=None)

    # NEW: total tokens generated
    if task_complete:
        total_tokens_solved = total_tokens_generated(doc_tr, up_to_step_inclusive=step_first_correct)
        total_tokens_unsolved = 0
    else:
        total_tokens_solved = 0
        total_tokens_unsolved = total_tokens_generated(doc_tr, up_to_step_inclusive=None)

    # Judge issues (UNPARSEABLE)
    for r in doc_gu:
        if not isinstance(r, dict):
            continue
        if r.get("outcome") not in ("CORRECT", "INCORRECT", "UNPARSEABLE"):
            issues.append(f"JUDGE_ERROR: {rule_id}/{inst_name} unknown outcome={r.get('outcome')}")
        if r.get("outcome") == "UNPARSEABLE":
            issues.append(f"JUDGE_UNPARSEABLE: {rule_id}/{inst_name} step={r.get('step')}")

    metrics = dict(
        variant=variant,
        rule_id=rule_id,
        inst_name=inst_name,
        task_complete=task_complete,
        tests_before_first=tests_before,

        # EXISTING I:C metrics (unchanged)
        ic_ratio_solved_pre=ic_ratio_solved_pre,
        ic_ratio_unsolved_all=ic_ratio_unsolved_all,
        raw_ic_solved_pre=raw_ic_solved_pre,
        raw_ic_unsolved_all=raw_ic_unsolved_all,

        ny_ratio_solved_pre=ny_ratio_solved_pre,
        ny_ratio_unsolved_all=ny_ratio_unsolved_all,
        raw_no_yes_solved_pre=raw_no_yes_solved_pre,
        raw_no_yes_unsolved_all=raw_no_yes_unsolved_all,

        md_ratio_solved_pre=md_ratio_solved_pre,
        md_ratio_unsolved_all=md_ratio_unsolved_all,
        raw_med_dax_solved_pre=raw_med_dax_solved_pre,
        raw_med_dax_unsolved_all=raw_med_dax_unsolved_all,

        avg_think_tokens_solved_pre=avg_think_solved_pre,
        avg_think_tokens_unsolved_all=avg_think_unsolved_all,
        raw_think_tokens_solved_pre=raw_think_solved_pre,
        raw_think_tokens_unsolved_all=raw_think_unsolved_all,

        first_guess_correct=first_guess_correct,
        first_guess_correct_no_tests=first_guess_correct_no_tests,

        # ADDED (counts + fraction)
        compat_ct_solved_pre=compat_ct_solved_pre,
        incompat_ct_solved_pre=incompat_ct_solved_pre,
        compat_ct_unsolved_all=compat_ct_unsolved_all,
        incompat_ct_unsolved_all=incompat_ct_unsolved_all,
        incompat_frac_solved_pre=incompat_frac_solved_pre,
        incompat_frac_unsolved_all=incompat_frac_unsolved_all,

        # NEW (total tokens)
        total_tokens_generated_solved=total_tokens_solved,
        total_tokens_generated_unsolved=total_tokens_unsolved,
    )
    return metrics, issues

# ----------------------------- Table rendering -----------------------------

def rule_order(rule: str) -> Tuple[int, int]:
    try:
        t, idx = rule.split("_")
        return (int(t[1:]), int(idx))
    except Exception:
        return (999, 999)

def mean_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None and not math.isnan(v)]
    return None if not xs else sum(xs) / len(xs)

def mean_int_ignore_none(vals: List[Optional[int]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return None if not xs else sum(xs) / float(len(xs))

def write_group_file(
    group_id: str,
    rules_to_instances: Dict[str, List[Dict[str, Any]]],
    group_issues: List[str],
    out_path: Path
):
    any_inst = None
    for insts in rules_to_instances.values():
        if insts:
            any_inst = insts[0]
            break
    variant = any_inst["variant"] if any_inst else ""
    is_dual = (variant == "dual-goal")
    col5 = "MED:DAX (solved, pre-first)" if is_dual else "NO:YES (solved, pre-first)"
    col6 = "MED:DAX (unsolved, all)"     if is_dual else "NO:YES (unsolved, all)"
    col7 = "AvgThinkTok (solved, pre-first)"
    col8 = "AvgThinkTok (unsolved, all)"
    col9 = "FirstGuess (correct & before tests)"

    # Added per-instance window metrics
    col10 = "CompatCt (window)"
    col11 = "IncompatCt (window)"
    col12 = "IncompatFrac (window)"

    # NEW per-instance totals (token sums)
    col13 = "TotalTokGenerated (window)"

    lines: List[str] = []
    lines.append(f"# Group {group_id}\n")

    for rule in sorted(rules_to_instances.keys(), key=rule_order):
        rows = rules_to_instances[rule]
        lines.append(f"## Rule {rule}\n")
        lines.append("")
        lines.append(
            "| Instance | TaskComplete | #TestsBeforeFirstCorrect | Incompat:Compat (solved, pre-first) | "
            "Incompat:Compat (unsolved, all) | "
            + col5 + " | " + col6 + " | " + col7 + " | " + col8 + " | " + col9 + " | "
            + col10 + " | " + col11 + " | " + col12 + " | " + col13 + " |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        # Accumulators for existing means
        tc_list: List[int] = []
        tests_list: List[Optional[int]] = []
        ic_pre_list: List[Optional[float]] = []
        ic_all_list: List[Optional[float]] = []
        ny_pre_list: List[Optional[float]] = []
        ny_all_list: List[Optional[float]] = []
        md_pre_list: List[Optional[float]] = []
        md_all_list: List[Optional[float]] = []
        th_pre_list: List[Optional[float]] = []
        th_all_list: List[Optional[float]] = []
        fg_list: List[int] = []

        # Added: per-episode incompat frac lists (then mean within rule)
        frac_solved_list: List[float] = []
        frac_unsolved_list: List[float] = []

        # NEW: per-rule token sums (NOT mean)
        total_tok_solved_sum = 0
        total_tok_unsolved_sum = 0

        for m in rows:
            tc_list.append(m["task_complete"])
            tests_list.append(m["tests_before_first"])
            ic_pre_list.append(m["ic_ratio_solved_pre"])
            ic_all_list.append(m["ic_ratio_unsolved_all"])

            th_pre_list.append(m.get("avg_think_tokens_solved_pre"))
            th_all_list.append(m.get("avg_think_tokens_unsolved_all"))

            fg_list.append(int(m.get("first_guess_correct_no_tests", 0)))

            if is_dual:
                md_pre_list.append(m["md_ratio_solved_pre"])
                md_all_list.append(m["md_ratio_unsolved_all"])
                col5_val = fmt(m["md_ratio_solved_pre"])
                col6_val = fmt(m["md_ratio_unsolved_all"])
            else:
                ny_pre_list.append(m["ny_ratio_solved_pre"])
                ny_all_list.append(m["ny_ratio_unsolved_all"])
                col5_val = fmt(m["ny_ratio_solved_pre"])
                col6_val = fmt(m["ny_ratio_unsolved_all"])

            # Added per-instance window counts/fraction
            if m["task_complete"] == 1:
                comp_ct = int(m.get("compat_ct_solved_pre", 0))
                incomp_ct = int(m.get("incompat_ct_solved_pre", 0))
                frac = m.get("incompat_frac_solved_pre")
                if isinstance(frac, float) and not math.isnan(frac):
                    frac_solved_list.append(frac)

                # NEW token window (solved)
                tok_window = int(m.get("total_tokens_generated_solved", 0))
                total_tok_solved_sum += tok_window
            else:
                comp_ct = int(m.get("compat_ct_unsolved_all", 0))
                incomp_ct = int(m.get("incompat_ct_unsolved_all", 0))
                frac = m.get("incompat_frac_unsolved_all")
                if isinstance(frac, float) and not math.isnan(frac):
                    frac_unsolved_list.append(frac)

                # NEW token window (unsolved)
                tok_window = int(m.get("total_tokens_generated_unsolved", 0))
                total_tok_unsolved_sum += tok_window

            lines.append(
                "| {inst} | {tc:d} | {tests} | {icp} | {ica} | {v5} | {v6} | {tpre} | {tall} | {fg} | {cc} | {ic} | {fr} | {tt} |".format(
                    inst=m["inst_name"],
                    tc=m["task_complete"],
                    tests=fmt_int(m["tests_before_first"]),
                    icp=fmt(m["ic_ratio_solved_pre"]),
                    ica=fmt(m["ic_ratio_unsolved_all"]),
                    v5=col5_val,
                    v6=col6_val,
                    tpre=fmt(m.get("avg_think_tokens_solved_pre")),
                    tall=fmt(m.get("avg_think_tokens_unsolved_all")),
                    fg=str(int(m.get("first_guess_correct_no_tests", 0))),
                    cc=str(comp_ct),
                    ic=str(incomp_ct),
                    fr=fmt(frac) if isinstance(frac, (float, int)) else "—",
                    tt=str(int(tok_window)),
                )
            )

        # Rule means (existing metrics unchanged) + add the 2 new frac means
        rule_means: Dict[str, Optional[float]] = {
            "Task Completion Rate": (sum(tc_list) / float(len(tc_list))) if tc_list else None,
            "#Tests Before First Correct (solved only)": mean_int_ignore_none([x for x, t in zip(tests_list, tc_list) if t == 1]),
            "Incompat:Compat (solved, pre-first)": mean_ignore_none([x for x, t in zip(ic_pre_list, tc_list) if t == 1]),
            "Incompat:Compat (unsolved, all)": mean_ignore_none([x for x, t in zip(ic_all_list, tc_list) if t == 0]),
        }
        if is_dual:
            rule_means["MED:DAX (solved, pre-first)"] = mean_ignore_none([x for x, t in zip(md_pre_list, tc_list) if t == 1])
            rule_means["MED:DAX (unsolved, all)"] = mean_ignore_none([x for x, t in zip(md_all_list, tc_list) if t == 0])
        else:
            rule_means["NO:YES (solved, pre-first)"] = mean_ignore_none([x for x, t in zip(ny_pre_list, tc_list) if t == 1])
            rule_means["NO:YES (unsolved, all)"] = mean_ignore_none([x for x, t in zip(ny_all_list, tc_list) if t == 0])

        rule_means["AvgThinkTok (solved, pre-first)"] = mean_ignore_none([x for x, t in zip(th_pre_list, tc_list) if t == 1])
        rule_means["AvgThinkTok (unsolved, all)"] = mean_ignore_none([x for x, t in zip(th_all_list, tc_list) if t == 0])
        rule_means["First Guess Rate"] = (sum(fg_list) / float(len(fg_list))) if fg_list else None

        # Added (fractions)
        rule_means["Avg Incompatibility Fraction (solved, pre-first)"] = None if not frac_solved_list else sum(frac_solved_list) / len(frac_solved_list)
        rule_means["Avg Incompatibility Fraction (unsolved, all)"] = None if not frac_unsolved_list else sum(frac_unsolved_list) / len(frac_unsolved_list)

        # NEW (token sums per rule; NOT averaged)
        rule_means["Sum Total Tokens Generated (solved, up-to-correct)"] = float(total_tok_solved_sum)
        rule_means["Sum Total Tokens Generated (unsolved, all)"] = float(total_tok_unsolved_sum)

        lines.append("\n**Rule Summary (means / sums):**")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k, v in rule_means.items():
            if k.startswith("Sum Total Tokens Generated"):
                lines.append(f"| {k} | {fmt_int0(int(v or 0))} |")
            else:
                lines.append(f"| {k} | {fmt(v)} |")
        lines.append("")

    if group_issues:
        lines.append("## Issues detected")
        for it in group_issues:
            lines.append(f"- {it}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_summary_txt(
    expdir: Path,
    outdir: Path,
    per_group_rule_means: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    variant: str,
    *,
    totals: Dict[str, int],
    avg_incompat_frac_solved: Optional[float],
    avg_incompat_frac_unsolved: Optional[float],
):
    is_dual = (variant == "dual-goal")
    col5 = "MED:DAX (solved, pre-first-correct)" if is_dual else "NO:YES (solved, pre-first-correct)"
    col6 = "MED:DAX (unsolved, all)"            if is_dual else "NO:YES (unsolved, all)"
    col7 = "AvgThinkTok (solved, pre-first)"
    col8 = "AvgThinkTok (unsolved, all)"

    lines: List[str] = []
    lines.append(f"# Summary — {expdir.relative_to(expdir.parents[3])}")
    lines.append("")
    lines.append("### Metric Definitions")
    lines.append("| Metric | Definition |")
    lines.append("|:--|:--|")
    lines.append("| **Task Completion Rate** | Fraction of instances where the model eventually announced a correct rule. |")
    lines.append("| **#Tests Before First Correct (over solved only)** | Mean number of test turns made before the first correct rule announcement (among solved instances only). |")
    lines.append("| **Incompat : Compat Ratio (solved, pre-first-correct)** | Ratio of incompatible to compatible test triples **before the first correct announcement**. |")
    lines.append("| **Incompat : Compat Ratio (unsolved, all)** | Ratio of incompatible to compatible triples across all test turns for instances that never solved the rule. |")
    if is_dual:
        lines.append("| **MED : DAX Ratio (solved, pre-first-correct)** | Count of MED-labeled tests divided by DAX-labeled tests **before the first correct announcement**. |")
        lines.append("| **MED : DAX Ratio (unsolved, all)** | Count of MED-labeled tests divided by DAX-labeled tests across all tests for unsolved instances. |")
    else:
        lines.append("| **NO : YES Ratio (solved, pre-first-correct)** | Ratio of “NO” to “YES” environment feedback **before the first correct announcement**. |")
        lines.append("| **NO : YES Ratio (unsolved, all)** | Ratio of “NO” to “YES” feedback across all turns for unsolved instances. |")
    lines.append("| **AvgThinkTok (solved, pre-first)** | Mean #tokens inside `<think>...</think>` before first correct (solved only). |")
    lines.append("| **AvgThinkTok (unsolved, all)** | Mean #tokens inside `<think>...</think>` over all turns (unsolved only). |")
    lines.append("| **First Guess Rate** | Fraction where first judged announcement is CORRECT and occurs before any tests. |")
    lines.append("| **Avg Incompatibility Fraction (solved, pre-first)** | Mean over solved instances of incompatible/(compatible+incompatible) before first correct (OK status only). |")
    lines.append("| **Avg Incompatibility Fraction (unsolved, all)** | Mean over unsolved instances of incompatible/(compatible+incompatible) over all tests (OK status only). |")
    lines.append("| **Sum Compatible Tests (solved, pre-first-correct)** | Total compatible tests across all solved instances, pre-first-correct, OK status only. |")
    lines.append("| **Sum Incompatible Tests (solved, pre-first-correct)** | Total incompatible tests across all solved instances, pre-first-correct, OK status only. |")
    lines.append("| **Sum Compatible Tests (unsolved, all)** | Total compatible tests across all unsolved instances, all tests, OK status only. |")
    lines.append("| **Sum Incompatible Tests (unsolved, all)** | Total incompatible tests across all unsolved instances, all tests, OK status only. |")
    lines.append("| **Sum Total Tokens Generated (solved, up-to-correct)** | Sum over solved instances of whitespace-token count over model `raw` up to and including first correct announce step. |")
    lines.append("| **Sum Total Tokens Generated (unsolved, all)** | Sum over unsolved instances of whitespace-token count over all model `raw`. |")
    lines.append("")

    overall_collect: Dict[str, List[Optional[float]]] = defaultdict(list)

    for group_id in sorted(per_group_rule_means.keys(), key=lambda g: int(g[1:])):
        lines.append(f"## {group_id} — Rule Means\n")
        lines.append(
            f"| Rule | Task Completion Rate | #Tests Before First Correct (solved only) | "
            f"Incompat:Compat (solved, pre-first-correct) | Incompat:Compat (unsolved, all) | {col5} | {col6} | "
            f"{col7} | {col8} | First Guess Rate | Avg IncompatFrac (solved) | Avg IncompatFrac (unsolved) | "
            f"Sum TotalTok (solved) | Sum TotalTok (unsolved) |"
        )
        lines.append("|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

        rule_means_map = per_group_rule_means[group_id]
        for rule in sorted(rule_means_map.keys(), key=rule_order):
            rm = rule_means_map[rule]
            tcr   = rm.get("Task Completion Rate")
            tbf   = rm.get("#Tests Before First Correct (solved only)")
            ic_p  = rm.get("Incompat:Compat (solved, pre-first-correct)")
            ic_a  = rm.get("Incompat:Compat (unsolved, all)")
            if is_dual:
                col5v = rm.get("MED:DAX (solved, pre-first)")
                col6v = rm.get("MED:DAX (unsolved, all)")
            else:
                col5v = rm.get("NO:YES (solved, pre-first)")
                col6v = rm.get("NO:YES (unsolved, all)")

            thp = rm.get("AvgThinkTok (solved, pre-first)")
            tha = rm.get("AvgThinkTok (unsolved, all)")
            fgr = rm.get("First Guess Rate")

            # Added per-rule fraction means
            fsol = rm.get("Avg Incompatibility Fraction (solved, pre-first)")
            funs = rm.get("Avg Incompatibility Fraction (unsolved, all)")

            # NEW per-rule token sums
            tsum_s = rm.get("Sum Total Tokens Generated (solved, up-to-correct)")
            tsum_u = rm.get("Sum Total Tokens Generated (unsolved, all)")

            lines.append(
                f"| {rule} | {fmt(tcr)} | {fmt(tbf)} | {fmt(ic_p)} | {fmt(ic_a)} | "
                f"{fmt(col5v)} | {fmt(col6v)} | {fmt(thp)} | {fmt(tha)} | {fmt(fgr)} | {fmt(fsol)} | {fmt(funs)} | "
                f"{fmt_int0(int(tsum_s or 0))} | {fmt_int0(int(tsum_u or 0))} |"
            )

            # existing collects
            overall_collect["Task Completion Rate"].append(tcr)
            overall_collect["#Tests Before First Correct (solved only)"].append(tbf)
            overall_collect["Incompat:Compat (solved, pre-first-correct)"].append(ic_p)
            overall_collect["Incompat:Compat (unsolved, all)"].append(ic_a)
            if is_dual:
                overall_collect["MED:DAX (solved, pre-first-correct)"].append(col5v)
                overall_collect["MED:DAX (unsolved, all)"].append(col6v)
            else:
                overall_collect["NO:YES (solved, pre-first-correct)"].append(col5v)
                overall_collect["NO:YES (unsolved, all)"].append(col6v)
            overall_collect["AvgThinkTok (solved, pre-first)"].append(thp)
            overall_collect["AvgThinkTok (unsolved, all)"].append(tha)
            overall_collect["First Guess Rate"].append(fgr)

            # Added collects (means across rules)
            overall_collect["Avg Incompatibility Fraction (solved, pre-first)"].append(fsol)
            overall_collect["Avg Incompatibility Fraction (unsolved, all)"].append(funs)

        lines.append("")

    lines.append("## Overall Means (across all rules)\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")

    # Existing overall means (unchanged)
    for key in [
        "Task Completion Rate",
        "#Tests Before First Correct (solved only)",
        "Incompat:Compat (solved, pre-first-correct)",
        "Incompat:Compat (unsolved, all)",
        "MED:DAX (solved, pre-first-correct)" if is_dual else "NO:YES (solved, pre-first-correct)",
        "MED:DAX (unsolved, all)" if is_dual else "NO:YES (unsolved, all)",
        "AvgThinkTok (solved, pre-first)",
        "AvgThinkTok (unsolved, all)",
        "First Guess Rate",
        "Avg Incompatibility Fraction (solved, pre-first)",
        "Avg Incompatibility Fraction (unsolved, all)",
    ]:
        lines.append(f"| {key} | {fmt(mean_ignore_none(overall_collect[key]))} |")

    # Global episode-level averages (direct over episodes)
    lines.append(f"| Avg Incompatibility Fraction (solved, pre-first) — global over episodes | {fmt(avg_incompat_frac_solved)} |")
    lines.append(f"| Avg Incompatibility Fraction (unsolved, all) — global over episodes | {fmt(avg_incompat_frac_unsolved)} |")

    # Global sums (no averaging)
    lines.append(f"| Sum Compatible Tests (solved, pre-first-correct) | {fmt_int0(totals['compat_solved_pre'])} |")
    lines.append(f"| Sum Incompatible Tests (solved, pre-first-correct) | {fmt_int0(totals['incompat_solved_pre'])} |")
    lines.append(f"| Sum Compatible Tests (unsolved, all) | {fmt_int0(totals['compat_unsolved_all'])} |")
    lines.append(f"| Sum Incompatible Tests (unsolved, all) | {fmt_int0(totals['incompat_unsolved_all'])} |")

    # NEW global token sums
    lines.append(f"| Sum Total Tokens Generated (solved, up-to-correct) | {fmt_int0(totals['total_tok_solved'])} |")
    lines.append(f"| Sum Total Tokens Generated (unsolved, all) | {fmt_int0(totals['total_tok_unsolved'])} |")

    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

# ----------------------------- Orchestration -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True, help="Variant dir, e.g., runs/train/llama-3.3-70b-instruct/dual-goal")
    ap.add_argument("--outdir", required=True, help="Output dir for per-group files and summary.txt")
    args = ap.parse_args()

    expdir = Path(args.expdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    # Infer variant name (for column choice)
    variant_guess = None
    for p in expdir.rglob("*.jsonl"):
        if p.name.endswith("_judge_guesses.jsonl") or p.name.endswith("_judge_compatibility.jsonl"):
            continue
        try:
            doc0 = read_jsonl(p)
            if doc0 and isinstance(doc0[0], dict) and "meta" in doc0[0]:
                variant_guess = extract_variant(doc0[0]["meta"])
                break
        except Exception:
            continue
    if variant_guess is None:
        variant_guess = expdir.name

    per_group_rule_instances: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    per_group_issues: Dict[str, List[str]] = defaultdict(list)

    # Global sums (no averaging)
    total_compat_solved_pre = 0
    total_incompat_solved_pre = 0
    total_compat_unsolved_all = 0
    total_incompat_unsolved_all = 0

    # NEW: global token sums
    total_tok_solved = 0
    total_tok_unsolved = 0

    # Global episode-level incompat fraction means
    frac_solved_list: List[float] = []
    frac_unsolved_list: List[float] = []

    # Iterate <expdir>/Tg_i/instX
    for rule_dir in sorted(
        [d for d in expdir.iterdir() if d.is_dir() and re.match(r"^[TVO]\d+_\d+$", d.name)],
        key=rule_order
    ):
        group_id = rule_dir.name.split("_")[0]   # e.g., "T1"
        for inst_dir in sorted(
            [d for d in rule_dir.iterdir() if d.is_dir() and d.name.startswith("inst")],
            key=lambda p: int(re.sub(r"\D", "", p.name) or "0")
        ):
            metrics, issues = analyze_instance(inst_dir)
            per_group_rule_instances[group_id][rule_dir.name].append(metrics)
            per_group_issues[group_id].extend(issues)

            # Accumulate global sums
            total_compat_solved_pre += int(metrics.get("compat_ct_solved_pre", 0))
            total_incompat_solved_pre += int(metrics.get("incompat_ct_solved_pre", 0))
            total_compat_unsolved_all += int(metrics.get("compat_ct_unsolved_all", 0))
            total_incompat_unsolved_all += int(metrics.get("incompat_ct_unsolved_all", 0))

            # NEW: accumulate global token sums
            total_tok_solved += int(metrics.get("total_tokens_generated_solved", 0))
            total_tok_unsolved += int(metrics.get("total_tokens_generated_unsolved", 0))

            # Accumulate global per-episode incompat fractions
            fs = metrics.get("incompat_frac_solved_pre")
            fu = metrics.get("incompat_frac_unsolved_all")
            if isinstance(fs, float) and not math.isnan(fs):
                frac_solved_list.append(fs)
            if isinstance(fu, float) and not math.isnan(fu):
                frac_unsolved_list.append(fu)

    avg_incompat_frac_solved = None if not frac_solved_list else (sum(frac_solved_list) / float(len(frac_solved_list)))
    avg_incompat_frac_unsolved = None if not frac_unsolved_list else (sum(frac_unsolved_list) / float(len(frac_unsolved_list)))

    outdir.mkdir(parents=True, exist_ok=True)

    # Determine output hierarchy to mirror runs/<split>/<model>/<variant>
    try:
        split = expdir.parents[1].name   # train
        model = expdir.parents[0].name   # llama-3.3-70b-instruct
        variant = expdir.name
    except Exception:
        split, model, variant = "train", "model", expdir.name

    base_out = outdir / split / model / variant
    base_out.mkdir(parents=True, exist_ok=True)

    per_group_rule_means: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    for group_id in sorted(per_group_rule_instances.keys(), key=lambda g: int(g[1:])):
        group_rules = per_group_rule_instances[group_id]
        issues = per_group_issues[group_id]
        group_path = base_out / f"{group_id}.txt"
        write_group_file(group_id, group_rules, issues, group_path)

        # Compute rule means for summary aggregation (existing metrics + add fraction means + token sums)
        group_rule_means: Dict[str, Dict[str, Optional[float]]] = {}
        for rule, rows in group_rules.items():
            tc_list = [m["task_complete"] for m in rows]
            tests_list = [m["tests_before_first"] for m in rows]
            ic_pre_list = [m["ic_ratio_solved_pre"] for m in rows]
            ic_all_list = [m["ic_ratio_unsolved_all"] for m in rows]
            th_pre_list = [m.get("avg_think_tokens_solved_pre") for m in rows]
            th_all_list = [m.get("avg_think_tokens_unsolved_all") for m in rows]
            fg_list = [int(m.get("first_guess_correct_no_tests", 0)) for m in rows]

            # Added: incompat fraction lists per rule
            frac_solved_rule: List[float] = []
            frac_unsolved_rule: List[float] = []
            for m in rows:
                if m["task_complete"] == 1:
                    fs = m.get("incompat_frac_solved_pre")
                    if isinstance(fs, float) and not math.isnan(fs):
                        frac_solved_rule.append(fs)
                else:
                    fu = m.get("incompat_frac_unsolved_all")
                    if isinstance(fu, float) and not math.isnan(fu):
                        frac_unsolved_rule.append(fu)

            # NEW: token sums per rule (NOT averaged)
            tok_solved_sum = sum(int(m.get("total_tokens_generated_solved", 0)) for m in rows)
            tok_unsolved_sum = sum(int(m.get("total_tokens_generated_unsolved", 0)) for m in rows)

            is_dual = any(m["variant"] == "dual-goal" for m in rows)
            if is_dual:
                md_pre_list = [m["md_ratio_solved_pre"] for m in rows]
                md_all_list = [m["md_ratio_unsolved_all"] for m in rows]
                rule_means = {
                    "Task Completion Rate": sum(tc_list) / float(len(tc_list)) if tc_list else None,
                    "#Tests Before First Correct (solved only)": mean_int_ignore_none([x for x, t in zip(tests_list, tc_list) if t == 1]),
                    "Incompat:Compat (solved, pre-first-correct)": mean_ignore_none([x for x, t in zip(ic_pre_list, tc_list) if t == 1]),
                    "Incompat:Compat (unsolved, all)": mean_ignore_none([x for x, t in zip(ic_all_list, tc_list) if t == 0]),
                    "MED:DAX (solved, pre-first)": mean_ignore_none([x for x, t in zip(md_pre_list, tc_list) if t == 1]),
                    "MED:DAX (unsolved, all)": mean_ignore_none([x for x, t in zip(md_all_list, tc_list) if t == 0]),
                    "AvgThinkTok (solved, pre-first)": mean_ignore_none([x for x, t in zip(th_pre_list, tc_list) if t == 1]),
                    "AvgThinkTok (unsolved, all)": mean_ignore_none([x for x, t in zip(th_all_list, tc_list) if t == 0]),
                    "First Guess Rate": (sum(fg_list) / float(len(fg_list))) if fg_list else None,
                }
            else:
                ny_pre_list = [m["ny_ratio_solved_pre"] for m in rows]
                ny_all_list = [m["ny_ratio_unsolved_all"] for m in rows]
                rule_means = {
                    "Task Completion Rate": sum(tc_list) / float(len(tc_list)) if tc_list else None,
                    "#Tests Before First Correct (solved only)": mean_int_ignore_none([x for x, t in zip(tests_list, tc_list) if t == 1]),
                    "Incompat:Compat (solved, pre-first-correct)": mean_ignore_none([x for x, t in zip(ic_pre_list, tc_list) if t == 1]),
                    "Incompat:Compat (unsolved, all)": mean_ignore_none([x for x, t in zip(ic_all_list, tc_list) if t == 0]),
                    "NO:YES (solved, pre-first)": mean_ignore_none([x for x, t in zip(ny_pre_list, tc_list) if t == 1]),
                    "NO:YES (unsolved, all)": mean_ignore_none([x for x, t in zip(ny_all_list, tc_list) if t == 0]),
                    "AvgThinkTok (solved, pre-first)": mean_ignore_none([x for x, t in zip(th_pre_list, tc_list) if t == 1]),
                    "AvgThinkTok (unsolved, all)": mean_ignore_none([x for x, t in zip(th_all_list, tc_list) if t == 0]),
                    "First Guess Rate": (sum(fg_list) / float(len(fg_list))) if fg_list else None,
                }

            # Added per-rule fraction means
            rule_means["Avg Incompatibility Fraction (solved, pre-first)"] = None if not frac_solved_rule else sum(frac_solved_rule) / len(frac_solved_rule)
            rule_means["Avg Incompatibility Fraction (unsolved, all)"] = None if not frac_unsolved_rule else sum(frac_unsolved_rule) / len(frac_unsolved_rule)

            # NEW per-rule token sums
            rule_means["Sum Total Tokens Generated (solved, up-to-correct)"] = float(tok_solved_sum)
            rule_means["Sum Total Tokens Generated (unsolved, all)"] = float(tok_unsolved_sum)

            group_rule_means[rule] = rule_means

        per_group_rule_means[group_id] = group_rule_means

    totals = {
        "compat_solved_pre": total_compat_solved_pre,
        "incompat_solved_pre": total_incompat_solved_pre,
        "compat_unsolved_all": total_compat_unsolved_all,
        "incompat_unsolved_all": total_incompat_unsolved_all,
        "total_tok_solved": total_tok_solved,
        "total_tok_unsolved": total_tok_unsolved,
    }

    write_summary_txt(
        expdir,
        base_out,
        per_group_rule_means,
        variant_guess,
        totals=totals,
        avg_incompat_frac_solved=avg_incompat_frac_solved,
        avg_incompat_frac_unsolved=avg_incompat_frac_unsolved,
    )

if __name__ == "__main__":
    main()
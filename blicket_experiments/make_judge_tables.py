#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

TRIAL_RE = re.compile(r"trial[-_]?(\d+)", re.IGNORECASE)

OBJ_COUNTS = [4, 8]
RULES = ["disjunctive", "conjunctive", "xor"]
N_TRIALS_EXPECTED = 16


# ------------------------- IO -------------------------

def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Could not parse {path} line {line_no}: {e}")
    return rows


def find_trial_idx(path: Path) -> Optional[int]:
    m = TRIAL_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ------------------------- First-correct from _judge_guess.jsonl -------------------------

def compute_first_correct_tests(judge_file: Path) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Returns:
      (is_correct_anywhere, first_correct_turns, tests_before_first_correct)

    turns is the 'turns' field from your log.
    If first correct announce is at turns=18 => tests_before = turns/2 = 9.
    """
    rows = read_jsonl(judge_file)
    if not rows:
        return False, None, None

    best_turns: Optional[int] = None
    for r in rows:
        rc = r.get("rule_correct", None)
        if rc is True:
            t = r.get("turns", None)
            if t is None:
                continue
            try:
                t_int = int(t)
            except Exception:
                continue
            if best_turns is None or t_int < best_turns:
                best_turns = t_int

    if best_turns is None:
        return False, None, None

    tests_before = best_turns // 2
    return True, best_turns, tests_before


# ------------------------- I:C from _judge_compat.jsonl -------------------------

def compat_path_from_guess_path(guess_path: Path) -> Path:
    # action_log_trial-0_judge_guess.jsonl -> action_log_trial-0_judge_compat.jsonl
    return guess_path.with_name(guess_path.name.replace("_judge_guess.jsonl", "_judge_compat.jsonl"))


def read_compat_rows(compat_file: Path) -> List[dict]:
    """
    Compat files produced by our judge are JSONL with a first "meta" row.
    We return only per-pair rows (skip meta).
    """
    rows = read_jsonl(compat_file)
    if not rows:
        return []
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and "meta" in r and len(r.keys()) == 1:
            continue
        out.append(r)
    return out


def compute_ic_for_trial(
    guess_file: Path,
    solved: bool,
    first_correct_turns: Optional[int],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    I:C = (#incompatible tests)/(#compatible tests)
    - Solved: only consider tests BEFORE the first correct announce (test_turn < first_correct_turns).
    - Unsolved: consider the full trajectory (all pairs).

    We ignore judge rows with status != OK/OK_AFTER_RUNTIME_RETRY and non-bool 'compatible'.
    Returns:
      (ic_ratio_or_None, debug_info dict)
    """
    compat_file = compat_path_from_guess_path(guess_file)
    debug = {
        "compat_file_exists": compat_file.exists(),
        "n_pairs_total": 0,
        "n_pairs_used": 0,
        "n_compat": 0,
        "n_incompat": 0,
        "n_skipped_nonok": 0,
        "n_skipped_missing": 0,
        "n_skipped_after_first": 0,
    }

    if not compat_file.exists():
        return None, debug

    rows = read_compat_rows(compat_file)
    debug["n_pairs_total"] = len(rows)
    if not rows:
        return None, debug

    n_compat = 0
    n_incompat = 0

    for r in rows:
        status = r.get("status", "")
        comp = r.get("compatible", None)  # expected bool for OK rows
        test_turn = r.get("test_turn", None)

        if status not in ("OK", "OK_AFTER_RUNTIME_RETRY"):
            debug["n_skipped_nonok"] += 1
            continue
        if not isinstance(comp, bool):
            debug["n_skipped_missing"] += 1
            continue

        # windowing
        if solved:
            if first_correct_turns is None:
                debug["n_skipped_after_first"] += 1
                continue
            try:
                tt = int(test_turn)
            except Exception:
                debug["n_skipped_missing"] += 1
                continue
            if tt >= int(first_correct_turns):
                debug["n_skipped_after_first"] += 1
                continue

        # count
        if comp:
            n_compat += 1
        else:
            n_incompat += 1
        debug["n_pairs_used"] += 1

    debug["n_compat"] = n_compat
    debug["n_incompat"] = n_incompat

    if n_compat == 0:
        # no compatible tests in-window -> undefined/infinite; keep None so tables stay clean
        return None, debug

    return (n_incompat / n_compat), debug


def compute_compat_incompat_counts_for_trial(
    guess_file: Path,
    solved: bool,
    first_correct_turns: Optional[int],
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Return (n_compat_used, n_incompat_used, debug)
    using the same filtering rules as compute_ic_for_trial.
    - Solved: tests with test_turn < first_correct_turns
    - Unsolved: all tests
    Skips non-OK rows and rows where compatible isn't bool (incl compile errors).
    """
    compat_file = compat_path_from_guess_path(guess_file)
    debug = {
        "compat_file_exists": compat_file.exists(),
        "n_pairs_total": 0,
        "n_pairs_used": 0,
        "n_compat": 0,
        "n_incompat": 0,
        "n_skipped_nonok": 0,
        "n_skipped_missing": 0,
        "n_skipped_after_first": 0,
    }

    if not compat_file.exists():
        return 0, 0, debug

    rows = read_compat_rows(compat_file)
    debug["n_pairs_total"] = len(rows)
    if not rows:
        return 0, 0, debug

    n_compat = 0
    n_incompat = 0

    for r in rows:
        status = r.get("status", "")
        comp = r.get("compatible", None)
        test_turn = r.get("test_turn", None)

        if status not in ("OK", "OK_AFTER_RUNTIME_RETRY"):
            debug["n_skipped_nonok"] += 1
            continue
        if not isinstance(comp, bool):
            debug["n_skipped_missing"] += 1
            continue

        if solved:
            if first_correct_turns is None:
                debug["n_skipped_after_first"] += 1
                continue
            try:
                tt = int(test_turn)
            except Exception:
                debug["n_skipped_missing"] += 1
                continue
            if tt >= int(first_correct_turns):
                debug["n_skipped_after_first"] += 1
                continue

        if comp:
            n_compat += 1
        else:
            n_incompat += 1
        debug["n_pairs_used"] += 1

    debug["n_compat"] = n_compat
    debug["n_incompat"] = n_incompat
    return n_compat, n_incompat, debug


def collect_compat_errors_for_trial(guess_file: Path) -> List[Dict[str, Any]]:
    """
    Return a compact list of non-OK rows from _judge_compat.jsonl for printing.
    """
    compat_file = compat_path_from_guess_path(guess_file)
    if not compat_file.exists():
        return [{"status": "MISSING_FILE", "error": str(compat_file)}]

    rows = read_compat_rows(compat_file)
    errs: List[Dict[str, Any]] = []
    for r in rows:
        status = r.get("status", "")
        if status in ("OK", "OK_AFTER_RUNTIME_RETRY"):
            continue
        errs.append({
            "index": r.get("index", None),
            "announce_turn": r.get("announce_turn", None),
            "test_turn": r.get("test_turn", None),
            "status": status,
            "compile_error": r.get("compile_error", ""),
        })
    return errs


# ------------------------- Tables -------------------------

def format_table(
    trial_rows: List[Tuple[int, bool, Optional[int], Optional[float], Optional[float]]]
) -> Tuple[str, float, float, Optional[float], Optional[float]]:
    """
    trial_rows: list of (trial_idx, correct, tests_before_first_correct_or_None, ic_solved, ic_unsolved)

    Returns:
      (markdown_table, avg_tests_over_correct, accuracy, mean_ic_solved_over_solved, mean_ic_unsolved_over_unsolved)
    """
    correct_tests = [t for _, ok, t, _, _ in trial_rows if ok and t is not None]
    accuracy = sum(1 for _, ok, _, _, _ in trial_rows if ok) / max(1, len(trial_rows))
    avg_tests = float("nan") if len(correct_tests) == 0 else (sum(correct_tests) / len(correct_tests))

    # aggregate I:C across relevant trials
    ic_solved_vals = [ic for _, ok, _, ic, _ in trial_rows if ok and isinstance(ic, (int, float)) and not math.isnan(ic)]
    ic_unsolved_vals = [ic_u for _, ok, _, _, ic_u in trial_rows if (not ok) and isinstance(ic_u, (int, float)) and not math.isnan(ic_u)]

    mean_ic_solved = None if len(ic_solved_vals) == 0 else (sum(ic_solved_vals) / len(ic_solved_vals))
    mean_ic_unsolved = None if len(ic_unsolved_vals) == 0 else (sum(ic_unsolved_vals) / len(ic_unsolved_vals))

    def _fmt_ic(x: Optional[float]) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return f"{x:.3f}"

    lines = []
    lines.append("| Trial | Correct? | #Tests before first correct | I:C (solved, pre-first) | I:C (unsolved, all) |")
    lines.append("|---:|:---:|---:|---:|---:|")
    for tid, ok, tests, ic_s, ic_u in sorted(trial_rows, key=lambda x: x[0]):
        lines.append(
            f"| {tid} | {'✅' if ok else '❌'} | {'' if tests is None else tests} | "
            f"{_fmt_ic(ic_s) if ok else ''} | {_fmt_ic(ic_u) if (not ok) else ''} |"
        )

    avg_str = "" if math.isnan(avg_tests) else f"{avg_tests:.2f}"
    acc_str = f"{accuracy:.3f}"
    lines.append(f"|  | **Avg (correct only)** | **{avg_str}** |  |  |")
    lines.append(f"|  | **Accuracy** | **{acc_str}** |  |  |")
    if mean_ic_solved is not None:
        lines.append(f"|  |  |  | **Mean I:C (solved)** | **{_fmt_ic(mean_ic_solved)}** |")
    if mean_ic_unsolved is not None:
        lines.append(f"|  |  |  | **Mean I:C (unsolved)** | **{_fmt_ic(mean_ic_unsolved)}** |")

    return "\n".join(lines), avg_tests, accuracy, mean_ic_solved, mean_ic_unsolved


def collect_condition(folder: Path) -> List[Path]:
    return sorted(folder.rglob("*_judge_guess.jsonl"))


def ensure_trials_0_to_15(
    trial_map: Dict[int, Tuple[bool, Optional[int], Optional[float], Optional[float]]]
) -> List[Tuple[int, bool, Optional[int], Optional[float], Optional[float]]]:
    rows: List[Tuple[int, bool, Optional[int], Optional[float], Optional[float]]] = []
    for tid in range(N_TRIALS_EXPECTED):
        if tid in trial_map:
            ok, tests, ic_s, ic_u = trial_map[tid]
            rows.append((tid, ok, tests, ic_s, ic_u))
        else:
            rows.append((tid, False, None, None, None))
    return rows


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp_root",
        required=True,
        help="Path to exp_output_alt (contains model folders). Example: /scratch/.../exp_output_alt",
    )
    args = ap.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    if not exp_root.exists():
        raise SystemExit(f"exp_root does not exist: {exp_root}")

    results_dir = exp_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir() and p.name != "results"])

    # summary[condition_key][model_name] =
    #   (avg_tests, acc, mean_ic_solved, mean_ic_unsolved,
    #    sum_compat_solved, sum_incompat_solved, sum_compat_unsolved, sum_incompat_unsolved)
    summary: Dict[str, Dict[str, Tuple[float, float, Optional[float], Optional[float], int, int, int, int]]] = {}

    # also collect compat judge errors in the same “print only errors” style
    compat_errors_path = results_dir / "compat_judge_failures.md"
    compat_errors_lines: List[str] = []
    compat_errors_lines.append("# Compatibility judge failures")
    compat_errors_lines.append("")
    compat_errors_lines.append("Non-OK rows from `*_judge_compat.jsonl` (or missing compat files).")
    compat_errors_lines.append("")

    for model_dir in model_dirs:
        model_name = model_dir.name
        out_path = results_dir / f"{model_name}_judge_report.md"

        parts: List[str] = []
        parts.append(f"# Judge report for `{model_name}`")
        parts.append("")

        for obj in OBJ_COUNTS:
            for rule in RULES:
                cond_key = f"obj_{obj}_{rule}"
                summary.setdefault(cond_key, {})

                folder = model_dir / f"obj_{obj}" / f"{rule}_45tests"
                parts.append(f"## obj_{obj} / {rule}")
                parts.append("")

                if not folder.exists():
                    parts.append(f"**Missing folder:** `{folder}`")
                    parts.append("")
                    summary[cond_key][model_name] = (float("nan"), 0.0, None, None, 0, 0, 0, 0)
                    continue

                judge_files = collect_condition(folder)
                trial_map: Dict[int, Tuple[bool, Optional[int], Optional[float], Optional[float]]] = {}

                # NEW: totals for this (model, obj, rule)
                sum_compat_solved = 0
                sum_incompat_solved = 0
                sum_compat_unsolved = 0
                sum_incompat_unsolved = 0

                for jf in judge_files:
                    tid = find_trial_idx(jf)
                    if tid is None:
                        continue

                    ok, first_turns, tests_before = compute_first_correct_tests(jf)

                    # I:C solved / unsolved
                    ic_solved = None
                    ic_unsolved = None
                    if ok:
                        ic_solved, _dbg = compute_ic_for_trial(jf, solved=True, first_correct_turns=first_turns)
                        c_cnt, i_cnt, _dbg2 = compute_compat_incompat_counts_for_trial(
                            jf, solved=True, first_correct_turns=first_turns
                        )
                        sum_compat_solved += c_cnt
                        sum_incompat_solved += i_cnt
                    else:
                        ic_unsolved, _dbg = compute_ic_for_trial(jf, solved=False, first_correct_turns=None)
                        c_cnt, i_cnt, _dbg2 = compute_compat_incompat_counts_for_trial(
                            jf, solved=False, first_correct_turns=None
                        )
                        sum_compat_unsolved += c_cnt
                        sum_incompat_unsolved += i_cnt

                    trial_map[tid] = (ok, tests_before, ic_solved, ic_unsolved)

                    # collect compat judge errors (only if there are any)
                    errs = collect_compat_errors_for_trial(jf)
                    if errs:
                        compat_errors_lines.append(f"## {model_name} / obj_{obj} / {rule} / trial {tid}")
                        compat_errors_lines.append("")
                        compat_errors_lines.append("| index | announce_turn | test_turn | status | compile_error |")
                        compat_errors_lines.append("|---:|---:|---:|---|---|")
                        for e in errs:
                            ce = (e.get("compile_error", "") or "").replace("\n", " ").strip()
                            compat_errors_lines.append(
                                f"| {e.get('index','')} | {e.get('announce_turn','')} | {e.get('test_turn','')} | "
                                f"{e.get('status','')} | {ce} |"
                            )
                        compat_errors_lines.append("")

                trial_rows = ensure_trials_0_to_15(trial_map)
                table_md, avg_tests, acc, mean_ic_s, mean_ic_u = format_table(trial_rows)
                parts.append(table_md)
                parts.append("")

                # NEW: print totals (no averages)
                parts.append("**Compat/Incompat totals (no averaging; filtered to OK rows only):**")
                parts.append("")
                parts.append("| Split | Sum Compatible | Sum Incompatible |")
                parts.append("|---|---:|---:|")
                parts.append(f"| Solved (pre-first-correct) | {sum_compat_solved} | {sum_incompat_solved} |")
                parts.append(f"| Unsolved (all) | {sum_compat_unsolved} | {sum_incompat_unsolved} |")
                parts.append("")

                summary[cond_key][model_name] = (
                    avg_tests, acc, mean_ic_s, mean_ic_u,
                    sum_compat_solved, sum_incompat_solved, sum_compat_unsolved, sum_incompat_unsolved
                )

        out_path.write_text("\n".join(parts), encoding="utf-8")
        print(f"[OK] Wrote model report: {out_path}")

    # ---- write summary ----
    summary_path = results_dir / "summary_judge_report.md"
    sp: List[str] = []
    sp.append("# Summary judge report")
    sp.append("")
    sp.append(f"Models found: {len(model_dirs)}")
    sp.append("")

    model_names = [d.name for d in model_dirs]

    for obj in OBJ_COUNTS:
        for rule in RULES:
            cond_key = f"obj_{obj}_{rule}"
            sp.append(f"## obj_{obj} / {rule}")
            sp.append("")
            sp.append("| Model | Avg #Tests (correct only) | Accuracy | Mean I:C (solved, pre-first) | Mean I:C (unsolved, all) | "
                      "ΣC (solved pre-first) | ΣI (solved pre-first) | ΣC (unsolved all) | ΣI (unsolved all) |")
            sp.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

            for mn in model_names:
                avg_tests, acc, mic_s, mic_u, scs, sis, scu, siu = summary.get(cond_key, {}).get(
                    mn, (float("nan"), 0.0, None, None, 0, 0, 0, 0)
                )
                avg_str = "" if math.isnan(avg_tests) else f"{avg_tests:.2f}"

                def _fmt(x: Optional[float]) -> str:
                    if x is None:
                        return ""
                    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                        return ""
                    return f"{x:.3f}"

                sp.append(
                    f"| {mn} | {avg_str} | {acc:.3f} | {_fmt(mic_s)} | {_fmt(mic_u)} | "
                    f"{scs} | {sis} | {scu} | {siu} |"
                )

            sp.append("")

    summary_path.write_text("\n".join(sp), encoding="utf-8")
    print(f"[OK] Wrote summary report: {summary_path}")

    # ---- write compat judge failures (only errors) ----
    compat_errors_path.write_text("\n".join(compat_errors_lines), encoding="utf-8")
    print(f"[OK] Wrote compatibility failures report: {compat_errors_path}")


if __name__ == "__main__":
    main()
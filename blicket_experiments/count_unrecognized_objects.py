import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PHRASE = "not a recognized object"

RULE_STEPS_RE = re.compile(r"^(conjunctive|disjunctive)_(\d+)$", re.IGNORECASE)

def safe_load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # ignore malformed lines rather than dying
                continue
    return rows

def find_model_rule_steps(exp_output_root: Path, jsonl_path: Path):
    """
    Expect: exp_output/<model>/<rule_steps>/.../file.jsonl
    Returns (model, rule, steps) or ("UNKNOWN", "UNKNOWN", "UNKNOWN")
    """
    rel = jsonl_path.resolve().relative_to(exp_output_root.resolve())
    parts = rel.parts
    if len(parts) < 2:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"

    model = parts[0]
    rule_steps = parts[1]
    m = RULE_STEPS_RE.match(rule_steps)
    if m:
        rule = m.group(1).lower()
        steps = int(m.group(2))
    else:
        rule, steps = "UNKNOWN", "UNKNOWN"

    return model, rule, steps

def last_history_obs_per_trial(rows):
    """
    rows: list[dict] for (maybe multiple trials).
    Returns dict trial_idx -> last_history_obs (string)
    """
    by_trial = defaultdict(list)
    for r in rows:
        t = r.get("trial_idx", 0)
        by_trial[t].append(r)

    out = {}
    for t, tr in by_trial.items():
        last = tr[-1]
        out[t] = last.get("history_obs", "") or ""
    return out

def main(exp_output_dir: str):
    exp_root = Path(exp_output_dir)
    if not exp_root.exists():
        raise FileNotFoundError(exp_root)

    # experiment_key := (model, rule, steps)
    agg = defaultdict(lambda: {
        "n_trials": 0,
        "total_occurrences": 0,
        "trials_with_any": 0,
        "files": set(),
    })

    for jsonl_path in exp_root.rglob("*.jsonl"):
        rows = safe_load_jsonl(jsonl_path)
        if not rows:
            continue

        model, rule, steps = find_model_rule_steps(exp_root, jsonl_path)
        exp_key = (model, rule, steps)

        per_trial_last = last_history_obs_per_trial(rows)

        for trial_idx, hist in per_trial_last.items():
            c = hist.lower().count(PHRASE.lower())

            agg[exp_key]["n_trials"] += 1
            agg[exp_key]["total_occurrences"] += c
            if c > 0:
                agg[exp_key]["trials_with_any"] += 1
            agg[exp_key]["files"].add(str(jsonl_path))

    # Print grouped report: per model -> per experiment (rule, steps)
    by_model = defaultdict(list)
    for (model, rule, steps), stats in agg.items():
        n = stats["n_trials"]
        total = stats["total_occurrences"]
        by_model[model].append((rule, steps, n, total, stats["trials_with_any"], len(stats["files"])))

    # Sort nicely: steps desc then rule
    for model in sorted(by_model.keys()):
        print(f"\n=== Model: {model} ===")
        rows = sorted(by_model[model], key=lambda x: (x[1] if isinstance(x[1], int) else -1, x[0]), reverse=True)

        print("rule\tsteps\tn_trials\ttotal_occ\tavg_per_trial\ttrials_with_any\tn_files")
        for rule, steps, n_trials, total_occ, trials_any, n_files in rows:
            avg = total_occ / n_trials if n_trials else 0.0
            print(f"{rule}\t{steps}\t{n_trials}\t{total_occ}\t{avg:.3f}\t{trials_any}\t{n_files}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python report_unrecognized_per_experiment.py <exp_output_dir>")
        sys.exit(1)

    main(sys.argv[1])

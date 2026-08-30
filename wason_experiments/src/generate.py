# src/generate.py
import random, yaml, csv
from pathlib import Path
from collections import OrderedDict
from src.rule_compiler import compile_expr

SEED = 1337
RANGE = range(-99, 101)

def load_rules(path):
    data = yaml.safe_load(Path(path).read_text())
    rules = OrderedDict()
    for r in data["rules"]:
        rules[r["id"]] = {**r, "pred": compile_expr(r["expr"])}
    return rules

def load_splits(path):
    data = yaml.safe_load(Path(path).read_text())
    return data["splits"]

def resolve_groups(groups_section, train_groups=None):
    """
    Resolve a list of group specs (supports {like: T1} for IID),
    returning a list of dicts with:
      { 'name': <str>, 'rules': [id1,id2,id3,id4] }
    """
    out = []
    for g in groups_section:
        if "like" in g:
            if train_groups is None:
                raise ValueError("Found 'like' reference but no train_groups provided")
            ref = next(x for x in train_groups if x["name"] == g["like"])
            out.append({"name": ref["name"], "rules": list(ref["rules"])})
        else:
            if "name" not in g or "rules" not in g:
                raise ValueError("Each group must have 'name' and 'rules' (or use {like: ...})")
            if len(g["rules"]) != 4:
                raise ValueError(f"group {g['name']} must list exactly 4 rule IDs")
            out.append({"name": g["name"], "rules": list(g["rules"])})
    return out

def main():
    # Paths
    REPO = Path(__file__).resolve().parents[1]
    RULES_YAML  = REPO / "rules" / "rules.yaml"
    GROUPS_YAML = REPO / "groups" / "groups.yaml"
    OUT_CSV     = REPO / "data" / "instances_train.csv"

    rng = random.Random(SEED)
    Path(REPO / "data").mkdir(exist_ok=True)

    rules = load_rules(RULES_YAML)
    splits = load_splits(GROUPS_YAML)

    # Resolve group specs (train first, then allow iid to copy with {like: ...})
    train_groups = resolve_groups(splits["train"])
    iid_groups   = resolve_groups(splits.get("iid", []), train_groups)
    oodv_groups  = resolve_groups(splits["ood_val"])
    oodt_groups  = resolve_groups(splits["ood_test"])

    # Plan: (split_name, group_list, samples_per_group)
    plan = [
        ("train",   train_groups, 180),
        ("iid",     iid_groups,   1),
        ("ood_val", oodv_groups,  2),
    ]

    # Precompile predicates per group in the given rules order
    # Store feasible triples per group
    all_groups = []
    for split, groups, _k in plan:
        for g in groups:
            preds = [rules[rid]["pred"] for rid in g["rules"]]
            all_groups.append({
                "split": split,
                "name": g["name"],
                "rules": g["rules"],   # [rule1, rule2, rule3, rule4] in order
                "preds": preds,        # callables in the same order
                "feasible": [],
            })

    # Evaluate feasibility across the whole cube
    total_checked = 0
    for a in RANGE:
        print(f"a={a}")  # progress (200 lines)
        for b in RANGE:
            for c in RANGE:
                total_checked += 1
                for G in all_groups:
                    p0, p1, p2, p3 = G["preds"]
                    # short-circuit in provided order
                    if not p0(a,b,c): continue
                    if not p1(a,b,c): continue
                    if not p2(a,b,c): continue
                    if not p3(a,b,c): continue
                    G["feasible"].append((a,b,c))

    # Report feasible sizes
    print("\nFeasible set sizes:")
    for G in all_groups:
        print(f"{G['split']}/{G['name']}: |S|={len(G['feasible'])}")
    print(f"Total triples checked: {total_checked}")

    # Sample and write CSV
    rows = []
    for split, _groups, k in plan:
        for G in [g for g in all_groups if g["split"] == split]:
            S = G["feasible"]
            if len(S) < k:
                raise RuntimeError(f"{split}/{G['name']}: feasible set too small ({len(S)}) for k={k}")
            for (a,b,c) in rng.sample(S, k):
                r1, r2, r3, r4 = G["rules"]
                rows.append([split, G["name"], r1, r2, r3, r4, a, b, c])

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","group","rule1","rule2","rule3","rule4","a","b","c"])
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV} with {len(rows)} rows")

if __name__ == "__main__":
    main()

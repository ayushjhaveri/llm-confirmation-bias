# src/select_cases.py
"""
Map testcase ids (e.g., T1_1, O1_2) to rule predicate + one or more seed triples
selected by instance index from data/instances.csv.

Instance semantics:
- We gather ALL rows in instances.csv whose `group` == testcase group prefix (e.g., "T1" for "T1_1")
- They are kept in CSV order.
- Instance indices are 1-based over this filtered list.
- "--instances all" selects all such rows.
"""

import csv, yaml
from pathlib import Path
from typing import Tuple, Dict, Callable, List, Any
from src.rule_compiler import compile_expr

def load_rules(repo: Path) -> Dict[str, Dict]:
    data = yaml.safe_load((repo / "rules" / "rules.yaml").read_text())
    rules = {}
    for r in data["rules"]:
        rules[r["id"]] = {**r, "pred": compile_expr(r["expr"])}
    return rules

def load_instances_for_group(repo: Path, group: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(repo / "data" / "instances.csv", "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["group"] == group:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No instances found for group '{group}' in data/instances.csv")
    return rows  # CSV order preserved

def parse_instances_arg(arg: str, total: int) -> List[int]:
    """
    Returns a sorted list of 1-based indices.
    'all' -> [1..total]
    '1,3,5' -> [1,3,5]
    """
    if arg.lower() == "all":
        return list(range(1, total + 1))
    idxs = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i < 1 or i > total:
            raise ValueError(f"Instance index {i} out of range 1..{total}")
        idxs.append(i)
    # keep order as provided (don’t sort by default)
    return idxs

def resolve_testcase_instances(repo: Path, testcase: str, instances_arg: str):
    """
    Returns a list of dicts, one per requested instance:
      {
        'rule_id': <e.g., T1_1>,
        'rule_meta': {..., 'name', 'expr', 'family', 'pred': callable},
        'seed': (a,b,c),                 # ints from instances.csv
        'split': <split in CSV>,         # e.g., train/iid/ood_val/ood_test
        'instance_index': <1-based>,     # index within all rows for this group
      }
    """
    assert "_" in testcase, "Testcase must be like T1_1, O3_2, etc."
    group = testcase.split("_")[0]

    rules = load_rules(repo)
    if testcase not in rules:
        raise KeyError(f"Unknown rule id: {testcase}")
    rule_meta = rules[testcase]  # includes 'pred'

    rows = load_instances_for_group(repo, group)
    sel = parse_instances_arg(instances_arg, len(rows))

    out = []
    for idx in sel:
        row = rows[idx - 1]  # 1-based -> 0-based
        a, b, c = int(row["a"]), int(row["b"]), int(row["c"])
        out.append({
            "rule_id": testcase,
            "rule_meta": rule_meta,
            "seed": (a, b, c),
            "split": row["split"],
            "instance_index": idx,
        })
    return out

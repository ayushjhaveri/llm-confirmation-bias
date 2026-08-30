import yaml
from pathlib import Path
from typing import Dict, Any
from src.rule_compiler import compile_expr

def load_prior_data(repo: Path) -> Dict[str, Any]:
    data = yaml.safe_load((repo / "rules" / "prior_rules.yaml").read_text())

    rules = {}
    for r in data["rules"]:
        rules[r["id"]] = {**r, "pred": compile_expr(r["expr"])}

    instances = {}
    for inst in data["instances"]:
        instances[inst["id"]] = inst

    return {"rules": rules, "instances": instances}

def parse_instances_arg(arg: str):
    if arg.lower() == "all":
        return "all"
    return [x.strip() for x in arg.split(",") if x.strip()]

def parse_rules_arg(arg: str):
    if arg.lower() == "all":
        return "all"
    return [x.strip() for x in arg.split(",") if x.strip()]

def resolve_prior_cases(repo: Path, instance_arg: str, rule_arg: str):
    """
    Returns list of specs:
      {
        "instance_id": "P1",
        "rule_id": "P1_3",
        "instance_meta": {...},
        "rule_meta": {..., "pred": callable},
        "seed": (a,b,c),
        "candidate_rule_ids": [...],
        "candidate_rule_names": [...],
      }
    """
    data = load_prior_data(repo)
    rules = data["rules"]
    instances = data["instances"]

    inst_sel = parse_instances_arg(instance_arg)
    if inst_sel == "all":
        inst_ids = list(instances.keys())
    else:
        inst_ids = inst_sel

    out = []
    for inst_id in inst_ids:
        if inst_id not in instances:
            raise KeyError(f"Unknown instance id: {inst_id}")

        inst = instances[inst_id]
        seed = tuple(inst["seed"])
        candidate_rule_ids = list(inst["rule_ids"])

        rule_sel = parse_rules_arg(rule_arg)
        if rule_sel == "all":
            chosen_rule_ids = candidate_rule_ids
        else:
            chosen_rule_ids = rule_sel

        for rid in chosen_rule_ids:
            if rid not in candidate_rule_ids:
                raise ValueError(
                    f"Rule {rid} is not a candidate rule for instance {inst_id}. "
                    f"Valid rules: {candidate_rule_ids}"
                )
            if rid not in rules:
                raise KeyError(f"Unknown rule id: {rid}")

            candidate_rule_names = [rules[x]["name"] for x in candidate_rule_ids]

            out.append({
                "instance_id": inst_id,
                "rule_id": rid,
                "instance_meta": inst,
                "rule_meta": rules[rid],
                "seed": seed,
                "candidate_rule_ids": candidate_rule_ids,
                "candidate_rule_names": candidate_rule_names,
            })

    return out
# src/run_experiment.py
import argparse
from pathlib import Path
import json  # NEW

from .models.llama33_70b import Llama33_70B
from .models.qwq_32b import QwQ_32B
from .models.qwen3_8b import Qwen3_8B
from .models.qwen3_14b import Qwen3_14B
from .models.qwen3_32b import Qwen3_32B
from .models.qwen3_8b_no_think import Qwen3_8B_NoThink
from .models.qwen3_14b_no_think import Qwen3_14B_NoThink
from .models.qwen3_32b_no_think import Qwen3_32B_NoThink
from .models.gpt_4o import OpenAIChat
from .models.gemini_2_5_pro import Gemini25Pro
from .models.r1_distill_llama_70b import R1DistillLlama70B
from .experiments.baseline import BaselineExperiment
from .experiments.dual_goal import DualGoalExperiment
from .experiments.think_opposites import ThinkOppositesExperiment
from .select_cases import resolve_testcase_instances
from .io_utils import ensure_dir, timestamp, save_transcript_txt, save_model_params  # removed save_transcript_jsonl

EXPERIMENTS = {
    "baseline": BaselineExperiment,
    "dual-goal": DualGoalExperiment,
    "think-in-opposites": ThinkOppositesExperiment,
}

def build_model(name: str, model_path: str | None):
    n = name.lower()
    if n in ["llama-3.3-70b", "llama", "llama33_70b", "llama-3.3-70b-instruct"]:
        return Llama33_70B(model_path=model_path)
    if n in ["qwq-32b", "qwq32b", "qwq", "qwq_32b", "qwen-qwq-32b"]:
        return QwQ_32B(model_path=model_path or "/scratch/aj4332/models/qwq-32b")
    if n in ["qwen3-8b", "qwen3_8b", "qwen3-8b-instruct"]:
        return Qwen3_8B(model_path=model_path)
    if n in ["qwen3-14b", "qwen3_14b", "qwen3-14b-instruct"]:
        return Qwen3_14B(model_path=model_path)
    if n in ["qwen3-32b", "qwen3_32b", "qwen3-32b-instruct"]:
        return Qwen3_32B(model_path=model_path)
    if n.startswith("gpt-"):
        return OpenAIChat(model=name)
    if "gemini" in n:
        return Gemini25Pro(model=name)
    if n in ["r1-distill-llama-70b", "deepseek-r1-distill-llama-70b", "r1-llama-70b"]:
        return R1DistillLlama70B(model_path=model_path)
    if n in ["qwen3-8b-no-think"]:
        return Qwen3_8B_NoThink(model_path=model_path)
    if n in ["qwen3-14b-no-think"]:
        return Qwen3_14B_NoThink(model_path=model_path)
    if n in ["qwen3-32b-no-think"]:
        return Qwen3_32B_NoThink(model_path=model_path)
    raise ValueError(f"Unknown model: {name}")

def build_experiment(variant: str, rule_name: str, pred, seed):
    v = variant.lower()
    if v not in EXPERIMENTS:
        raise ValueError(f"Unknown variant: {variant} (choose from {list(EXPERIMENTS)})")
    return EXPERIMENTS[v](rule_name, pred, seed)

def parse_testcases(testcases: str, repo: Path):
    if testcases.lower() == "all":
        import yaml
        data = yaml.safe_load((repo / "rules" / "rules.yaml").read_text())
        return [r["id"] for r in data["rules"]]
    return [x.strip() for x in testcases.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name e.g. llama-3.3-70b or qwq-32b")
    ap.add_argument("--model-path", default=None, help="Local HF path (optional if env var set)")
    ap.add_argument(
        "--model-dir-name",
        default=None,
        help="Optional: override the model name used in output directory structure and meta['model']",
    )
    ap.add_argument("--variant", required=True, choices=["baseline","dual-goal","think-in-opposites"])
    ap.add_argument("--outdir", required=True, help="Output base directory")
    ap.add_argument("--testcases", required=True, help="Comma-separated rule ids (e.g., T1_1,O1_2) or 'all'")
    ap.add_argument("--instances", required=True, help="Comma-separated 1-based indices (e.g., 1,3) or 'all'")

    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]

    # Build model (decoding params are hardcoded inside the model wrappers)
    model = build_model(args.model, args.model_path)
    model_params_dict = model.describe_params()

    # Decide the name used for directory + meta
    model_dir_name = args.model_dir_name or model_params_dict["model_name"]

    # Resolve testcases
    cases = parse_testcases(args.testcases, repo)

    for tc in cases:
        instance_specs = resolve_testcase_instances(repo, tc, args.instances)
        for spec in instance_specs:
            rule_id   = spec["rule_id"]
            rule_meta = spec["rule_meta"]
            pred      = rule_meta["pred"]
            seed      = spec["seed"]
            split     = spec["split"]
            inst_idx  = spec["instance_index"]

            exp = build_experiment(args.variant, rule_meta["name"], pred, seed)

            # Paths: <outdir>/<split>/<model>/<variant>/<rule_id>/inst<idx>/
            base = (Path(args.outdir)
                    / split
                    / model_dir_name
                    / exp.NAME
                    / rule_id
                    / f"inst{inst_idx}")
            ensure_dir(base)

            stamp = timestamp()
            stem = f"{rule_id}_inst{inst_idx}_{stamp}"
            jsonl_path  = base / f"{stem}.jsonl"
            txt_path    = base / f"{stem}.txt"
            params_path = base / f"{stem}.model_params.json"

            meta = {
                "testcase_id": rule_id,
                "hidden_rule_name": rule_meta["name"],
                "hidden_rule_expr": rule_meta["expr"],
                "variant": exp.NAME,
                "model": model_dir_name,
                "split": split,
                "instance_index": inst_idx,
                "seed_example": {"a": seed[0], "b": seed[1], "c": seed[2]},
                "decoding": model_params_dict.get("sampling", model_params_dict),
            }

            # Initialize JSONL with meta line
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")

            def append_turn_line(obj: dict) -> None:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # Run (streaming JSONL appends per turn via callback)
            turns = exp.run(model, on_turn=append_turn_line)

            # Write remaining artifacts
            save_transcript_txt(turns, str(txt_path))
            save_model_params(model_params_dict, str(params_path))

            print(f"Wrote:\n  {jsonl_path}\n  {txt_path}\n  {params_path}")

if __name__ == "__main__":
    main()

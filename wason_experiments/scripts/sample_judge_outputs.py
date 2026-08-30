import json
import random
from pathlib import Path

RUNS_ROOT = Path("/scratch/aj4332/cb_env/cb_experiments/runs/ood_test")

MODELS = [
    "gemini-2.5-pro",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "qwq-32b",
    "qwen3-8b-without-thinking",
    "qwen3-14b-without-thinking",
    "qwen3-32b-without-thinking",
    "llama-3.3-70b-instruct",
    "gpt-4o",
]

VARIANTS = ["baseline","think-in-opposites","dual-goal"]

random.seed(1337)

ann_correct = []
ann_incorrect = []

compat_true = []
compat_false = []

def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

for model in MODELS:

    model_dir = RUNS_ROOT / model
    if not model_dir.exists():
        continue

    for variant in VARIANTS:

        variant_dir = model_dir / variant
        if not variant_dir.exists():
            continue

        for guess_file in variant_dir.rglob("*_judge_guesses.jsonl"):

            compat_file = guess_file.with_name(
                guess_file.name.replace("_judge_guesses","_judge_compatibility")
            )

            guesses = read_jsonl(guess_file)

            # -------------------
            # ANNOUNCEMENTS
            # -------------------
            for g in guesses:

                gt = g.get("hidden_rule_name","")
                announced = g.get("announced_rule","")
                outcome = g.get("outcome","")

                row = (
                    str(guess_file),
                    gt,
                    announced,
                    outcome
                )

                if outcome == "CORRECT":
                    ann_correct.append(row)
                else:
                    ann_incorrect.append(row)

            # -------------------
            # COMPATIBILITY
            # -------------------
            if compat_file.exists():

                compats = read_jsonl(compat_file)

                for c in compats:

                    if not c.get("compile_ok",False):
                        continue

                    prev_ann = c.get("announced_rule","")
                    triple = c.get("triple","")
                    comp = c.get("compatible",None)

                    row = (
                        str(compat_file),
                        prev_ann,
                        str(triple),
                        comp
                    )

                    if comp:
                        compat_true.append(row)
                    else:
                        compat_false.append(row)


# stratified samples
ann_sample = random.sample(ann_correct,100) + random.sample(ann_incorrect,100)
compat_sample = random.sample(compat_true,100) + random.sample(compat_false,100)


# write announcement table
with open("announcement_judge_audit.tsv","w") as f:

    f.write("Source\tCorrect announcement\tModel announcement\tJudge outcome\n")

    for r in ann_sample:
        f.write("\t".join(map(str,r))+"\n")


# write compatibility table
with open("compatibility_judge_audit.tsv","w") as f:

    f.write("Source\tPrevious Announcement\tTest triple\tJudge Model Output\n")

    for r in compat_sample:
        f.write("\t".join(map(str,r))+"\n")

print("Done.")
print("announcement_judge_audit.tsv")
print("compatibility_judge_audit.tsv")
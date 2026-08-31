# Wason Experiments

Code for running Wason-rule experiments, judging the resulting transcripts, and calculating aggregate metrics.

## Setup

Create an environment and install the dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Run all commands from this directory so that `data/`, `rules/`, and `prompts/` resolve correctly.

## Run an experiment

The main entry point is `src/run_experiment.py`. Outputs and transcripts should be written under `runs/`:

```bash
python -m src.run_experiment \
  --model qwen3-8b \
  --model-path /path/to/model \
  --variant baseline \
  --outdir runs \
  --testcases O1_1 \
  --instances 1
```

Use `--testcases all` or `--instances all` to run the full corresponding set. Available variants are `baseline`, `dual-goal`, and `think-in-opposites`.

## Judge runs

Judge the generated transcripts with:

```bash
python -m src.run_judge_guess \
  --expdir runs/ood_test/qwen3-8b/baseline \
  --model-path /path/to/llama-3.3-70b-instruct \
  --prompts-dir prompts \
  --judge-temp 0.0 \
  --judge-max-tokens 12

python -m src.run_judge_compatibility \
  --expdir runs/ood_test/qwen3-8b/baseline \
  --model-path /path/to/llama-3.3-70b-instruct \
  --prompt-file prompts/generate_hypothesis_function.txt
```

For both commands, `--expdir` and `--model-path` are required. The other
arguments shown above are optional and use the displayed values by default.
Both commands operate on experiment outputs under `runs/` and write judge
results alongside the transcripts.

## Calculate metrics

First calculate per-variant metrics, then aggregate them into `results/`:

```bash
python -m src.analyze_variant_metrics \
  --expdir runs/ood_test/qwen3-8b/baseline \
  --outdir results/ood_test/qwen3-8b/baseline

python scripts/aggregate_overall_metrics.py
```

Both arguments to `src.analyze_variant_metrics` are required. Run it once for
each model and variant. `aggregate_overall_metrics.py` takes no command-line
arguments; it reads per-variant summaries from `results/ood_test/` and writes
the combined table to `results/ood_test/overall_summary.txt`.

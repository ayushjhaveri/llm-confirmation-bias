# Math Experiments

This directory contains evaluations and saved results for AIME 2025 and
BeyondAIME. Run all commands from this directory.

## Requirements

The evaluator requires Python 3.10 or newer and the `datasets`, `tqdm`, and
`vllm` packages. Model inference uses a local Qwen3 checkpoint. The required
Qwen3 model wrappers are included under `src/models/`.

The `--model` argument describes the base architecture and must be either
`qwen3-8b` or `qwen3-32b`. Use `--model-path` to select the corresponding base
or fine-tuned checkpoint.

## AIME 2025

The following command evaluates a Qwen3 8B checkpoint on both AIME 2025 test
sets, allows up to five retries after the initial answer, and saves resumable
JSONL progress plus the final JSON summary under `results_aime2025/`:

```bash
python eval_aime2025.py \
  --model qwen3-8b \
  --model-path /path/to/qwen3-8b \
  --dataset-name opencompass/AIME2025 \
  --split test \
  --max-retries 5 \
  --shuffle \
  --seed 0 \
  --output results_aime2025/results_qwen3-8b.json
```

For a 32B checkpoint, change `--model` to `qwen3-32b`, update
`--model-path`, and choose a distinct output filename. Existing output files
are resumed by default; pass `--no-resume` to start over.

## BeyondAIME

BeyondAIME uses the same evaluator with a different dataset. The saved
evaluation was divided into four shards. Run the following command once for
each `--shard-id` from `0` through `3`:

```bash
python eval_aime2025.py \
  --benchmark-name BeyondAIME \
  --model qwen3-8b \
  --model-path /path/to/qwen3-8b \
  --dataset-name ByteDance-Seed/BeyondAIME \
  --split test \
  --max-retries 5 \
  --shuffle \
  --seed 0 \
  --num-shards 4 \
  --shard-id 0 \
  --output results_beyondaime/results_qwen3-8b_five-retries_shard0-of-4.json
```

Keep `--seed`, `--num-shards`, and the shuffle setting identical across all
shards. Change both `--shard-id` and the shard number in `--output` for each
run. As with AIME 2025, use `qwen3-32b` for 32B checkpoints.

## Saved results

- `results_aime2025/` contains the AIME 2025 progress files and summaries.
- `results_beyondaime/` contains the four-shard BeyondAIME progress files and
  summaries.

For each evaluation, the `.jsonl` file stores incremental per-problem results
and the `.json` file stores the summarized result. Keeping both files allows
an interrupted evaluation to resume.

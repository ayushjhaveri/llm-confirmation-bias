# LLM Confirmation-Bias Blicket Experiments

This repository contains our Blicket experiments for evaluating confirmation
bias in language models. We built on top of the code released with
[Towards Data-Efficient and Generalizable LLM Agents: A Case Study in the Blicket Environment](https://arxiv.org/abs/2505.09614).

## Running an alternating-oracle experiment

Run commands from the repository root. The following example runs 16 trials
for one model on the two-blicket, four-object, conjunctive condition and saves
the output in the same layout as the checked-in evaluation logs:

```bash
HYDRA_FULL_ERROR=1 python run_trials_blicket_oracle_alternating.py \
  model_name=qwen3-8b \
  agent.model=local_vllm:/path/to/qwen3-8b \
  agent.thinking_mode=true \
  num_trials=16 \
  max_actions_per_trial=45 \
  env_kwargs.num_objects=4 \
  env_kwargs.num_blickets=2 \
  env_kwargs.rule=conjunctive \
  'hydra.run.dir=runs_2blickets/${model_name}/obj_${env_kwargs.num_objects}/${env_kwargs.rule}_${max_actions_per_trial}tests'
```

Set `agent.model` to the local model path and use `model_name` for its output
folder name. To run the other evaluation conditions, vary:

- `env_kwargs.num_objects`: `4` or `8`
- `env_kwargs.num_blickets`: `2` or `3`
- `env_kwargs.rule`: `conjunctive`, `disjunctive`, or `xor`

For three-blicket runs, set `env_kwargs.num_blickets=3` and change the output
root in `hydra.run.dir` to `runs_3blickets`. Think-in-opposites runs use the
same layout with a `tio-` prefix in `model_name`.

Each condition directory contains the raw `action_log_trial-*.jsonl` files,
conversation logs, Hydra configuration, per-trial judge outputs, and
`results.jsonl`.

## Running the judges and calculating results

After generating runs, first judge whether each announced hypothesis is
correct. Set `--root` to either `runs_2blickets` or `runs_3blickets`:

```bash
python judge_blicket_announces.py \
  --root <run-root> \
  --model_path /path/to/llama-3.3-70b-instruct \
  --tp 2 \
  --max_model_len 8192 \
  --gpu_mem_util 0.98
```

Then judge whether each test is compatible with the model's current announced
hypothesis:

```bash
python judge_blicket_compatibility.py \
  --expdir <run-root> \
  --model-path /path/to/qwen2.5-coder-32b-instruct
```

The announcement judge writes `*_judge_guess.jsonl` files, and the
compatibility judge writes `*_judge_compat.jsonl` files beside the original
action logs.

Calculate aggregate run metrics as CSV:

```bash
python aggregate_blicket_results.py \
  --exp_root <run-root> \
  --out_csv <run-root>/results/aggregated_results.csv
```

Finally, generate the readable per-model reports, compatibility-failure
report, and combined summary table:

```bash
python make_judge_tables.py --exp_root <run-root>
```

Run this pipeline separately with `<run-root>` set to `runs_2blickets` and
`runs_3blickets`.

## Current evaluation logs and results

- `runs_2blickets/`: consolidated two-blicket evaluation logs
- `runs_3blickets/`: consolidated three-blicket evaluation logs
- `runs_2blickets/results/`: two-blicket per-model judge reports, compatibility
  failures, and `summary_judge_report.md`
- `runs_3blickets/results/`: three-blicket per-model judge reports,
  compatibility failures, and `summary_judge_report.md`

To regenerate only the Markdown reports after adding already-judged runs:

```bash
python make_judge_tables.py --exp_root runs_2blickets
python make_judge_tables.py --exp_root runs_3blickets
```

---

# Original Blicket repository README

## Text-based Blicket Environment

Lightweight environment and runners to study active exploration and Q&A in a text-only “blicket” task. Agents can interact with an environment, ask/answer questions, and logs are saved for analysis.

⚠️ This code base is functional but have not been thoroughly tested. Please raise any issues if you encounter them! ⚠️


### Installation
```bash
conda create -n blicket python=3.10
conda activate blicket
pip install -r requirements.txt
```

### Configure API access (choose one or more)
- OpenAI-compatible (OpenAI):
  - `export OPENAI_API_KEY="<your_openai_key>"`
- DeepSeek:
  - `export DEEPSEEK_API_KEY="<your_deepseek_key>"`
- Ollama (local):
  - Install Ollama and pull a chat model
  - The code will use `http://localhost:11434/v1` automatically when model starts with `ollama/`

Note: We estimate costs with static prices in `lm_api.py` for convenience; verify against provider pricing.

### Quick start: single run
Minimal run using default `random_agent` (no API needed):
```bash
python run_trials_blicket.py num_trials=4 env_kwargs.rule="conjunctive"
```

LM-driven run (requires API key):
```bash
HYDRA_FULL_ERROR=1 python run_trials_blicket.py \
	agent=prompts_llm \
	num_trials=1 max_actions_per_trial=4 \
	env_kwargs.rule="disjunctive" \
	env_kwargs.num_objects=4 env_kwargs.num_blickets=2 \
	env_kwargs.transition_noise=0.0 \
	agent.model="gpt-4o-mini-2024-07-18" \
	agent.temperature=0.0 \
	seed=20
```

### Sweep example
Hydra makes it easy to sweep over configs:
```bash
HYDRA_FULL_ERROR=1 python run_trials_blicket.py \
	agent=prompts_llm \
	use_threadpool=True tp_max_workers=32 \
	num_trials=32 max_actions_per_trial=32 \
	agent.react=False,True \
	agent.system_msg_path="./agent/prompts/system_human_conj.txt","./agent/prompts/system_human.txt","./agent/prompts/system_math_def.txt" \
	env_kwargs.rule="conjunctive","disjunctive" \
	env_kwargs.num_objects=3 env_kwargs.num_blickets=2 \
	env_kwargs.transition_noise=0.0 \
	agent.model="deepseek-chat" \
	agent.temperature=0.0 \
	seed=20 -m
```

All outputs (results and per-trial logs) are saved under `exp_output/<date>/<time>/`.

### Interactive play
```bash
python play_blicket.py --num_objects 4 --num_blickets 2 --rule disjunctive --noise 0.0
```

### Post-processing to DuckDB
Aggregate experiment outputs into DuckDB databases for analysis:
```bash
python process_hypothesis_exps.py \
  exp_output/*/*/results.jsonl \
  --output_dir processed_output \
  --max_workers 4
```
The script writes three databases under `processed_output/...` for results, action logs, and question logs.

### Tips
- `HYDRA_FULL_ERROR=1` shows full tracebacks when debugging.
- Set `env_kwargs.transition_noise=0.0` for deterministic transitions.
- Choose models via `agent.model` (OpenAI, DeepSeek) or `ollama/<model_name>` for local.

# Failing to Falsify: Evaluating and Mitigating Confirmation Bias in Language Models

Official code repository for:

**Failing to Falsify: Evaluating and Mitigating Confirmation Bias in Language Models**  
Ayush Rajesh Jhaveri, Anthony GX-Chen, Ilia Sucholutsky, Eunsol Choi 

---

## Overview

This repository contains code, data, and language model outputs for evaluating confirmation bias in large language models (LLMs) during interactive hypothesis exploration.

We introduce:
- A multi-turn rule-discovery framework inspired by Wason’s 2-4-6 task
- A process-level confirmation bias metric (Incompatible:Compatible ratio, I:C)
- Psychology-inspired debiasing interventions (Dual-Goal, Think-in-Opposites)
- Symbolic knowledge distillation for internalizing falsification-oriented reasoning
- Cross-domain evaluation on the Blicket Test
- Evaluation on the AIME 2025 and BeyondAIME mathematics benchmarks

## Environment setup

From the repository root, create a Python 3.10 virtual environment and install
the shared dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Repository structure

```text
llm-confirmation-bias/
├── README.md
├── requirements.txt
├── wason_experiments/
│   ├── README.md
│   ├── data/
│   ├── prompts/
│   ├── rules/
│   ├── runs/
│   ├── results/
│   ├── scripts/
│   └── src/
├── blicket_experiments/
│   ├── README.md
│   ├── agent/
│   ├── env/
│   ├── runs_2blickets/
│   ├── runs_3blickets/
│   └── run_trials_blicket_oracle_alternating.py
└── math_experiments/
    ├── README.md
    ├── eval_aime2025.py
    ├── results_aime2025/
    ├── results_beyondaime/
    └── src/
```

Each experiment directory has its own README with commands for running the
experiments and locating the corresponding outputs and results.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback


@dataclass
class Sample:
    id: str
    messages: List[Dict[str, str]]
    completion: str


class JsonlSFTDataset(Dataset):
    def __init__(self, path: Path):
        self.rows: List[Sample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.rows.append(Sample(
                    id=r["id"],
                    messages=r["messages"],
                    completion=r["completion"],
                ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Sample:
        return self.rows[i]


class Collator:
    def __init__(self, tokenizer, max_length: int, debug_first_n: int = 3):
        self.tok = tokenizer
        self.max_length = max_length
        self.debug_first_n = debug_first_n
        self._dbg_count = 0

    def _render_prompt(self, messages):
        prompt = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = prompt + "/think"  # keep as you currently do
        return prompt

    def _maybe_debug(self, ex_id: str, prompt_text: str, completion: str,
                     prompt_ids, full_ids, labels):
        
        self._dbg_count += 1

        sup = len(labels) - min(len(prompt_ids), len(labels))
        print("=" * 100, flush=True)
        print(f"[SFT DEBUG #{self._dbg_count}] id={ex_id}", flush=True)
        print(f"  max_length         = {self.max_length}", flush=True)
        print(f"  prompt_tokens      = {len(prompt_ids)}", flush=True)
        print(f"  full_tokens        = {len(full_ids)}", flush=True)
        print(f"  supervised_tokens  = {sup}", flush=True)
        if sup == 0:
            print("  !!! supervised_tokens == 0 (all labels masked) !!!", flush=True)

        # Decode a small slice to see boundaries
        try:
            # last 200 tokens of prompt
            ptail = self.tok.decode(prompt_ids[-200:], skip_special_tokens=False) if len(prompt_ids) else ""
            # first 200 chars of completion raw
            chead = (completion or "")[:200].replace("\n", "\\n")
            print("  prompt_tail(decoded, last 200 toks):", flush=True)
            print(ptail.replace("\n", "\\n")[:1200], flush=True)
            print("  completion_head(raw, first 200 chars):", flush=True)
            print(chead, flush=True)
        except Exception as e:
            print(f"  [debug decode failed] {e}", flush=True)

        print("=" * 100, flush=True)

    def __call__(self, batch):
        input_ids_list, attn_list, labels_list = [], [], []

        for ex in batch:
            prompt_text = self._render_prompt(ex.messages)
            full_text = prompt_text + (ex.completion or "")

            prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
            full = self.tok(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            input_ids = full["input_ids"]
            attn = full["attention_mask"]

            labels = input_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            for j in range(prompt_len):
                labels[j] = -100

            # supervised tokens = tokens after the prompt (in this truncated full_text)
            sup = len(labels) - prompt_len

            # Print per-example debug (if you insist on always printing)
            # self._maybe_debug(
            #     ex_id=ex.id,
            #     prompt_text=prompt_text,
            #     completion=ex.completion,
            #     prompt_ids=prompt_ids,
            #     full_ids=input_ids,
            #     labels=labels,
            # )

            # If this is the real failure mode, catch it HERE (per-example)
            if sup == 0:
                print("=" * 120, flush=True)
                print("[SFT BAD EXAMPLE] supervised_tokens == 0 (this example contributes no loss)", flush=True)
                print(f"  id            = {ex.id}", flush=True)
                print(f"  max_length    = {self.max_length}", flush=True)
                print(f"  prompt_tokens = {len(prompt_ids)}", flush=True)
                print(f"  full_tokens   = {len(input_ids)}", flush=True)
                print("  completion_head(raw, first 400 chars):", flush=True)
                print(((ex.completion or "")[:400]).replace("\n","\\n"), flush=True)
                print("  completion_tail(raw, last 400 chars):", flush=True)
                print(((ex.completion or "")[-400:]).replace("\n","\\n"), flush=True)
                print("=" * 120, flush=True)
                raise RuntimeError("Bad example: supervised_tokens == 0")

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attn_list.append(torch.tensor(attn, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tok.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attn_list, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class OODEvalCallback(TrainerCallback):
    def __init__(self, ood_dataset, prefix: str = "ood"):
        self.ood_dataset = ood_dataset
        self.prefix = prefix
        self.trainer = None
        self._running = False  # recursion guard

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._running:
            return control
        if self.trainer is None:
            print("[OOD EVAL] trainer is None (callback not wired).", flush=True)
            return control

        self._running = True
        try:
            print(f"[OOD EVAL] step={state.global_step}", flush=True)
            ood_metrics = self.trainer.evaluate(
                eval_dataset=self.ood_dataset,
                metric_key_prefix=self.prefix,
            )
            self.trainer.log(ood_metrics)
        finally:
            self._running = False

        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF model path (student base), e.g. /scratch/.../qwen3-8b")
    ap.add_argument("--train_jsonl", required=True, type=Path)
    ap.add_argument("--iid_jsonl", required=True, type=Path)
    ap.add_argument("--ood_jsonl", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)

    ap.add_argument("--max_length", type=int, default=32768)

    ap.add_argument("--batch_size", type=int, default=1)          # per-device
    ap.add_argument("--grad_accum", type=int, default=64)         # gives effective bs ~64
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)  # default 2*r
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--wandb_project", type=str, default="cb_sft")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import os
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.run_name is None:
        args.run_name = args.outdir.name

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",  # works well for many HF models; adjust if needed
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("num trainable tensors:", len(trainable), "total trainable params:", sum(p.numel() for p in trainable), flush=True)
    assert len(trainable) > 0 and sum(p.numel() for p in trainable) > 0

    train_ds = JsonlSFTDataset(args.train_jsonl)
    iid_ds   = JsonlSFTDataset(args.iid_jsonl)
    ood_ds   = JsonlSFTDataset(args.ood_jsonl)

    collator = Collator(tokenizer, max_length=args.max_length, debug_first_n=3)

    # Evaluate on iid then ood in separate runs (Trainer supports one eval_dataset at a time).
    # We'll do iid as eval_dataset and also compute ood at the end via a second evaluate() call.
    targs = TrainingArguments(
        output_dir=str(args.outdir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name=args.run_name,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    ood_cb = OODEvalCallback(ood_ds, prefix="ood")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=iid_ds,
        data_collator=collator,
    )

    ood_cb.trainer = trainer
    trainer.add_callback(ood_cb)

    trainer.train()

    # Final eval on IID and OOD
    iid_metrics = trainer.evaluate(eval_dataset=iid_ds)
    ood_metrics = trainer.evaluate(eval_dataset=ood_ds)

    with (args.outdir / "final_eval.json").open("w", encoding="utf-8") as f:
        json.dump({"iid": iid_metrics, "ood": ood_metrics}, f, indent=2)

    trainer.save_model(str(args.outdir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.outdir / "tokenizer"))

    print("Done. Wrote:", args.outdir)


if __name__ == "__main__":
    main()

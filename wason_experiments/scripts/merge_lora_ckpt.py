#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Base HF model dir, e.g. /scratch/.../models/qwen3-8b")
    ap.add_argument("--adapter_ckpt", required=True, help="Trainer checkpoint dir, e.g. .../checkpoint-1200")
    ap.add_argument("--out", required=True, help="Output dir for merged model")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter from the checkpoint folder
    model = PeftModel.from_pretrained(base, args.adapter_ckpt)
    # Merge LoRA explainably into the base weights
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("Saved merged model to:", args.out)

if __name__ == "__main__":
    main()

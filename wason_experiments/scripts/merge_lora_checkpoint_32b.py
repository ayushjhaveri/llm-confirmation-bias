#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_max_memory(s: str):
    """
    Example:
      'cuda:0=78GiB,cuda:1=78GiB,cpu=400GiB'
    Returns dict usable as transformers' max_memory.
    """
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Base HF model dir, e.g. /scratch/.../models/qwen3-32b")
    ap.add_argument("--adapter_ckpt", required=True, help="Checkpoint dir containing adapter_model.safetensors, e.g. .../checkpoint-1000")
    ap.add_argument("--out", required=True, help="Output dir for merged model")

    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="dtype for loading/merging; bfloat16 recommended if supported")
    ap.add_argument("--device_map", default="auto",
                    help='Transformers device_map, e.g. "auto" (multi-GPU) or "cpu"')
    ap.add_argument("--max_memory", default=None,
                    help='Optional max_memory, e.g. \'cuda:0=78GiB,cuda:1=78GiB,cpu=400GiB\'')
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True")
    args = ap.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    max_memory = parse_max_memory(args.max_memory) if args.max_memory else None

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    # Load LoRA adapter (your checkpoint dir has adapter_config.json + adapter_model.safetensors)
    model = PeftModel.from_pretrained(base, args.adapter_ckpt)

    # Merge LoRA into base weights
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("Saved merged model to:", args.out)

if __name__ == "__main__":
    main()

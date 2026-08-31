#!/usr/bin/env python3
# Judge "compatibility" of (Announce -> next Test) for Blicket oracle logs.
# Compatibility here = whether the announced hypothesis predicts ON for the tested set.
#
# Writes <action_log_trial-X>_judge_compat.jsonl next to each input log file.
#
# Example:
#   python run_judge_blicket_compat.py \
#     --expdir /scratch/aj4332/cb_env/blicket-text-llm/exp_output_alt_oracle \
#     --model-path /scratch/aj4332/models/qwen2.5-coder-32b-instruct \
#     --prompt-file prompts/generate_blicket_hypothesis_function.txt

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import signal
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

# ------------------------- vLLM wrapper (Qwen2.5-Coder-32B) -------------------------

from vllm import LLM, SamplingParams

DEFAULT_QWEN_STOPS = ["<|im_end|>", "<|endoftext|>", "```"]

class Qwen25Coder32B:
    """
    vLLM-based wrapper for local Qwen2.5-Coder-32B-Instruct.

    Notes
    - Tailored for SHORT, STRICT code-gen (Announce → Python function).
    - Slightly cooler temperature, modest rep-penalty, longer max_new_tokens.
    - Uses tokenizer.apply_chat_template on a list of chat turns or a single string.
    """
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.98,
        max_model_len: int = 16384,
        enforce_eager: bool = True,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        base = os.path.basename(model_path.rstrip("/"))
        self.model_name = base or "qwen2.5-coder-32b-instruct"

        self._defaults: Dict[str, Any] = {
            "temperature": 0.15,
            "top_p": 0.95,
            "top_k": 0,
            "max_new_tokens": 640,
            "repetition_penalty": 1.05,
            "stop": DEFAULT_QWEN_STOPS,
        }

        self.engine = LLM(
            model=self.model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = self.engine.get_tokenizer()
            except Exception:
                self._tokenizer = self.engine.llm_engine.tokenizer
        return self._tokenizer

    def _as_chat_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        tok = self._get_tokenizer()
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _to_sampling_params(self, override: Optional["GenParams"] = None) -> SamplingParams:
        d = dict(self._defaults)
        if override is not None:
            d["temperature"] = override.temperature if override.temperature is not None else d["temperature"]
            d["top_p"] = override.top_p if override.top_p is not None else d["top_p"]
            d["top_k"] = override.top_k if override.top_k is not None else d["top_k"]
            d["max_new_tokens"] = override.max_new_tokens if override.max_new_tokens is not None else d["max_new_tokens"]
            d["repetition_penalty"] = override.repetition_penalty if override.repetition_penalty is not None else d["repetition_penalty"]
            if getattr(override, "stop", None) is not None:
                d["stop"] = override.stop

        return SamplingParams(
            temperature=cast(float, d["temperature"]),
            top_p=cast(float, d["top_p"]),
            top_k=cast(int, d["top_k"]),
            max_tokens=cast(int, d["max_new_tokens"]),
            repetition_penalty=cast(float, d["repetition_penalty"]),
            stop=d["stop"],
        )

    def generate(self, prompt: Union[str, List[Dict[str, str]]], params: Optional["GenParams"] = None) -> str:
        sp = self._to_sampling_params(params)
        chat_formatted = self._as_chat_prompt(prompt)
        outs = self.engine.generate([chat_formatted], sp)
        text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
        return text.strip()

    def describe_params(self) -> Dict[str, Any]:
        d = self._defaults
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "backend": "vllm",
            "dtype": "bfloat16",
            "temperature": d["temperature"],
            "top_p": d["top_p"],
            "top_k": d["top_k"],
            "max_new_tokens": d["max_new_tokens"],
            "repetition_penalty": d["repetition_penalty"],
            "stop": d["stop"],
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": None,
            "max_model_len": None,
        }

@dataclass
class GenParams:
    temperature: float = 0.15
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 640
    repetition_penalty: float = 1.05
    stop: Optional[List[str]] = None


# ------------------------- discovery + IO -------------------------

TRIAL_RE = re.compile(r"^action_log_trial-(\d+)\.jsonl$")

def iter_action_logs(expdir: Path) -> Iterable[Path]:
    for p in expdir.rglob("action_log_trial-*.jsonl"):
        if not p.is_file():
            continue
        if p.name.endswith("_judge_compat.jsonl"):
            continue
        m = TRIAL_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx <= 15:
            yield p

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------- parsing turns -------------------------

ANN_RE = re.compile(r"(?mi)^\s*Announce:\s*relevant=\[(?P<blickets>[^\]]*)\]\s*;\s*rule=(?P<rule>.+?)\s*$")
TEST_RE = re.compile(r"(?mi)^\s*Test:\s*\[(?P<objs>[^\]]*)\]\s*$")
OBJ_RE = re.compile(r"\bobject\s+(\d+)\b", re.IGNORECASE)

def parse_announce_payload(action_str: str) -> Optional[Dict[str, Any]]:
    """
    action_str example:
      'Announce: relevant=[object 2, object 3]; rule=at least one relevant object is on the machine'
    Returns:
      {"hypothesis_text": "relevant=[object 2, object 3]; rule=...", "blicket_ids":[2,3], "rule_text":"..."}
    """
    if not action_str:
        return None
    s = action_str.strip()
    if s.lower().startswith("announce:"):
        s2 = s
    else:
        return None

    m = ANN_RE.match(s2)
    if not m:
        return None

    blickets_raw = (m.group("blickets") or "").strip()
    rule_text = (m.group("rule") or "").strip()

    blicket_ids = [int(x) for x in OBJ_RE.findall(blickets_raw)]
    hypothesis_text = f"relevant=[{', '.join([f'object {i}' for i in blicket_ids])}]; rule={rule_text}"
    # keep original too, in case you want
    return {
        "hypothesis_text": hypothesis_text,
        "blicket_ids": blicket_ids,
        "rule_text": rule_text,
        "raw_action": s,
    }

def parse_test_objects(action_str: str) -> Optional[List[int]]:
    """
    action_str example:
      'Test: [object 0, object 1]'
      'Test: []'
    Returns list of object indices.
    """
    if not action_str:
        return None
    s = action_str.strip()
    if not s.lower().startswith("test:"):
        return None
    m = TEST_RE.match(s)
    if not m:
        return None
    inner = (m.group("objs") or "").strip()
    ids = [int(x) for x in OBJ_RE.findall(inner)]
    return ids

def collect_pairs(doc: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract (announce -> next test) pairs using turn_type fields in your action_log.
    Uses the STRICT alternating assumption: announce then test then announce...
    """
    meta: Dict[str, Any] = {}
    pairs: List[Dict[str, Any]] = []

    if not doc:
        return meta, pairs

    # log-level meta
    first = doc[0]
    gs0 = first.get("game_state", {}) if isinstance(first, dict) else {}
    object_names = gs0.get("object_names", [])
    num_objects = len(object_names) if isinstance(object_names, list) else None
    trial_idx = first.get("trial_idx", None)

    meta = {
        "trial_idx": trial_idx,
        "num_objects": num_objects,
        "source_file": None,
    }

    # iterate turns; pair each announce with immediately following test
    for i in range(len(doc) - 1):
        cur = doc[i]
        nxt = doc[i + 1]
        if cur.get("turn_type") != "announce":
            continue
        if nxt.get("turn_type") != "test":
            continue

        ann = parse_announce_payload(cur.get("action", "") or "")
        tst = parse_test_objects(nxt.get("action", "") or "")
        if ann is None or tst is None:
            continue

        pairs.append({
            "announce_turn": cur.get("turns", i),
            "test_turn": nxt.get("turns", i + 1),
            "hypothesis_text": ann["hypothesis_text"],
            "announced_blicket_ids": ann["blicket_ids"],
            "announced_rule_text": ann["rule_text"],
            "tested_object_ids": tst,
            "num_objects": num_objects,
        })

    return meta, pairs


# ------------------------- prompt loading -------------------------

def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def render_prompt(tmpl: str, hypothesis_text: str, num_objects: int) -> str:
    return (
        tmpl.replace("{HYPOTHESIS}", hypothesis_text.strip())
            .replace("{NUM_OBJECTS}", str(int(num_objects)))
    )


# ------------------------- safe compile/exec (Wason-style) -------------------------

class Timeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise Timeout()

def strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t.replace("```", "")
    return t.replace("```", "").strip()

def safe_load_rule(func_text: str, compile_timeout_s: int = 2) -> types.FunctionType:
    """
    Expects:
      def rule(state: list) -> bool:
          ...
    Enforces AST constraints similar to your Wason setup.
    """
    code_raw = strip_fences(func_text)

    sig = re.search(
        r'^\s*def\s+rule\s*\(\s*state\s*:\s*list\s*\)\s*->\s*bool\s*:\s*',
        code_raw, re.MULTILINE
    )
    if not sig:
        raise ValueError("Missing signature: def rule(state: list) -> bool:")

    code = code_raw[sig.start():]
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from e

    forbidden = (
        ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
        ast.With, ast.Try, ast.Raise, ast.Lambda, ast.ClassDef
    )
    for n in ast.walk(tree):
        if isinstance(n, forbidden):
            raise ValueError(f"Forbidden construct: {type(n).__name__}")

    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fdefs or fdefs[0].name != "rule":
        raise ValueError("Top-level function 'rule' is not the first definition.")

    safe_globals = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "range": range,
            "len": len,
            "all": all,
            "any": any,
            "sum": sum,
            "int": int,
            "bool": bool,
            "isinstance": isinstance,
        }
    }
    safe_locals: Dict[str, Any] = {}

    old = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(compile_timeout_s)
        exec(compile(tree, "<blicket_rule>", "exec"), safe_globals, safe_locals)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    fn = safe_locals.get("rule") or safe_globals.get("rule")
    if not callable(fn):
        raise ValueError("Compiled object `rule` is not callable.")
    return cast(types.FunctionType, fn)

def run_with_timeout(fn, state: List[bool], run_timeout_s: int = 1) -> bool:
    old = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(run_timeout_s)
        return bool(fn(state))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ------------------------- retry helpers (Wason-style) -------------------------

def render_retry_messages(
    tmpl: str,
    hypothesis_text: str,
    num_objects: int,
    bad_output: str,
    python_error: str,
    attempt: int,
) -> List[Dict[str, str]]:
    first_user = render_prompt(tmpl, hypothesis_text, num_objects)
    safe_bad = (bad_output or "").strip()
    return [
        {"role": "user", "content": first_user},
        {"role": "assistant", "content": safe_bad},
        {
            "role": "user",
            "content": (
                f"The above function FAILED with this Python error:\n"
                f"```\n{python_error}\n```\n\n"
                f"This was attempt {attempt}. "
                "Regenerate a correct function ONLY.\n"
                f"Hypothesis:\n{hypothesis_text.strip()}\n"
                f"num_objects={int(num_objects)}"
            ),
        },
    ]

def call_model_once(model: Qwen25Coder32B, prompt_or_messages, *, temperature: float, max_new_tokens: int) -> str:
    params = GenParams(
        temperature=temperature,
        top_p=0.95,
        top_k=0,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        stop=DEFAULT_QWEN_STOPS,
    )
    return model.generate(prompt_or_messages, params)

def call_model_with_retries_compile(
    model: Qwen25Coder32B,
    prompt_template: str,
    hypothesis_text: str,
    num_objects: int,
    *,
    max_retries: int = 2,
) -> Tuple[Optional[types.FunctionType], str, List[str], bool, str]:
    """
    Returns: (fn_or_none, final_code, raw_attempts, compile_ok, compile_err)
    """
    raw_attempts: List[str] = []
    first_prompt = render_prompt(prompt_template, hypothesis_text, num_objects)

    raw0 = call_model_once(model, first_prompt, temperature=0.15, max_new_tokens=640)
    raw_attempts.append(raw0)
    try:
        fn = safe_load_rule(raw0)
        return fn, raw0, raw_attempts, True, ""
    except Exception as e0:
        last_err = str(e0)

    # Optional: bump tokens once if looks truncated
    if "EOF" in last_err or "unexpected EOF" in last_err:
        raw1 = call_model_once(model, first_prompt, temperature=0.15, max_new_tokens=896)
        raw_attempts.append(raw1)
        try:
            fn = safe_load_rule(raw1)
            return fn, raw1, raw_attempts, True, ""
        except Exception as e1:
            last_err = str(e1)

    prev_code = raw_attempts[-1]
    for attempt_idx in range(1, max_retries + 1):
        msgs = render_retry_messages(prompt_template, hypothesis_text, num_objects, prev_code, last_err, attempt_idx)
        raw_retry = call_model_once(model, msgs, temperature=0.12 if attempt_idx == 1 else 0.10, max_new_tokens=896)
        raw_attempts.append(raw_retry)
        prev_code = raw_retry
        try:
            fn = safe_load_rule(raw_retry)
            return fn, raw_retry, raw_attempts, True, ""
        except Exception as e_retry:
            last_err = str(e_retry)

    return None, raw_attempts[-1] if raw_attempts else "", raw_attempts, False, last_err


# ------------------------- core judging -------------------------------------

def state_from_test(num_objects: int, tested_ids: List[int]) -> List[bool]:
    st = [False] * int(num_objects)
    for i in tested_ids:
        if 0 <= int(i) < int(num_objects):
            st[int(i)] = True
    return st

def judge_file(
    action_log_path: Path,
    model: Qwen25Coder32B,
    prompt_template: str,
) -> Path:
    print(f"[compat] reading: {action_log_path}", file=sys.stderr, flush=True)
    doc = read_jsonl(action_log_path)
    meta, pairs = collect_pairs(doc)

    out_path = action_log_path.with_name(action_log_path.stem + "_judge_compat.jsonl")

    # meta header row (Wason-style)
    rows: List[Dict[str, Any]] = []
    rows.append({
        "meta": {
            **meta,
            "judger_model": model.describe_params(),
            "judger_task": "announce_to_test_compatibility",
            "prompt_file": "<from-arg>",
            "source_log": str(action_log_path),
            "num_pairs": len(pairs),
        }
    })

    cache: Dict[str, types.FunctionType] = {}

    for k, pair in enumerate(pairs, start=1):
        hyp = pair["hypothesis_text"]
        num_objects = int(pair["num_objects"] or 0)
        tested_ids = list(pair["tested_object_ids"])
        st = state_from_test(num_objects, tested_ids)

        fn = cache.get(hyp)
        compile_ok = True
        compile_err = ""
        raw_code = None
        raw_attempts: List[str] = []

        if fn is None:
            fn, raw_code, raw_attempts, compile_ok, compile_err = call_model_with_retries_compile(
                model,
                prompt_template,
                hyp,
                num_objects,
                max_retries=2,
            )
            if compile_ok and fn is not None:
                cache[hyp] = fn

        if not compile_ok or fn is None:
            rows.append({
                "index": k,
                "announce_turn": pair["announce_turn"],
                "test_turn": pair["test_turn"],
                "hypothesis_text": hyp,
                "tested_object_ids": tested_ids,
                "state": st,
                "compile_ok": False,
                "compile_error": compile_err,
                "pred_machine_on": None,
                "compatible": None,
                "status": "COMPILE_ERROR",
                "raw_fn": raw_code,
                "raw_attempts": raw_attempts,
            })
            continue

        try:
            pred_on = run_with_timeout(fn, st, run_timeout_s=1)
            # compatibility == predicted ON for the tested set (confirmatory test under hypothesis)
            rows.append({
                "index": k,
                "announce_turn": pair["announce_turn"],
                "test_turn": pair["test_turn"],
                "hypothesis_text": hyp,
                "tested_object_ids": tested_ids,
                "state": st,
                "compile_ok": True,
                "compile_error": "",
                "pred_machine_on": bool(pred_on),
                "compatible": bool(pred_on),
                "status": "OK",
            })
        except Timeout:
            rows.append({
                "index": k,
                "announce_turn": pair["announce_turn"],
                "test_turn": pair["test_turn"],
                "hypothesis_text": hyp,
                "tested_object_ids": tested_ids,
                "state": st,
                "compile_ok": True,
                "compile_error": "",
                "pred_machine_on": "TIMEOUT",
                "compatible": "TIMEOUT",
                "status": "TIMEOUT",
            })
        except Exception as e_runtime:
            runtime_err = str(e_runtime)

            # runtime retries (like Wason)
            fixed = False
            last_code = raw_code or ""
            attempts_rt = list(raw_attempts)

            for attempt_idx in (1, 2):
                msgs = render_retry_messages(prompt_template, hyp, num_objects, last_code, runtime_err, attempt_idx)
                raw_retry = call_model_once(model, msgs, temperature=0.12 if attempt_idx == 1 else 0.10, max_new_tokens=896)
                attempts_rt.append(raw_retry)
                last_code = raw_retry
                try:
                    fn2 = safe_load_rule(raw_retry)
                    pred_on = run_with_timeout(fn2, st, run_timeout_s=1)
                    rows.append({
                        "index": k,
                        "announce_turn": pair["announce_turn"],
                        "test_turn": pair["test_turn"],
                        "hypothesis_text": hyp,
                        "tested_object_ids": tested_ids,
                        "state": st,
                        "compile_ok": True,
                        "compile_error": "",
                        "pred_machine_on": bool(pred_on),
                        "compatible": bool(pred_on),
                        "status": "OK_AFTER_RUNTIME_RETRY",
                        "raw_attempts": attempts_rt,
                    })
                    fixed = True
                    break
                except Exception as e2:
                    runtime_err = str(e2)
                    continue

            if not fixed:
                rows.append({
                    "index": k,
                    "announce_turn": pair["announce_turn"],
                    "test_turn": pair["test_turn"],
                    "hypothesis_text": hyp,
                    "tested_object_ids": tested_ids,
                    "state": st,
                    "compile_ok": True,
                    "compile_error": f"runtime_error: {runtime_err}",
                    "pred_machine_on": "ERROR",
                    "compatible": "ERROR",
                    "status": "ERROR",
                    "raw_attempts": attempts_rt,
                })

    write_jsonl(out_path, rows)
    print(f"[compat] wrote: {out_path} (pairs={len(pairs)})", file=sys.stderr, flush=True)
    return out_path


# ------------------------- CLI + prompt default -------------------------------------

DEFAULT_PROMPT_TEXT = """Your task is to write a Python function that determines whether a given device would be ON,
according to a natural-language hypothesis about relevant objects and a rule.

The input is a list named state of length num_objects.
- state[i] is True iff "object i" is on the device.
- state[i] is False iff "object i" is NOT on the device.

The function should implement the ON-condition described by the hypothesis.
Return True if the hypothesis predicts the device is ON for the given state, otherwise False.

You may assume objects are named "object 0" ... "object (num_objects-1)".
Empty sets are allowed.

---
FORMAT & SAFETY CONSTRAINTS (MANDATORY):

- Output ONLY the function definition. No prose, no explanations, no code fences.
- Signature must be exactly:
  def rule(state: list) -> bool:
- You may define small helper functions NESTED INSIDE rule().
- Do NOT use: import, from, global, nonlocal, with, try/except, raise, class, lambda,
  eval, exec, file or network I/O, or context managers.
- Use only pure Python booleans, indexing into state, comparisons, if/else, loops.
- Allowed builtins: abs, min, max, range, len, all, any, sum, int, bool, isinstance.
- You may only access:
  * the parameter state,
  * local variables / helper functions defined inside rule,
  * the allowed builtins listed above.
- Before you output the function, silently check you did not use disallowed syntax or names.
  Then output ONLY the final function.

---
Hypothesis:
{HYPOTHESIS}

num_objects={NUM_OBJECTS}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True, help="Root experiment directory (recursively scanned)")
    ap.add_argument("--model-path", required=True, help="Local path to qwen2.5-coder-32b-instruct")
    ap.add_argument("--prompt-file", default="", help="Optional prompt file. If omitted, uses an embedded default.")
    args = ap.parse_args()

    expdir = Path(args.expdir).expanduser().resolve()
    if not expdir.exists():
        print(f"[error] expdir does not exist: {expdir}", file=sys.stderr)
        sys.exit(2)

    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.exists():
            print(f"[error] Missing prompt file: {prompt_path}", file=sys.stderr)
            sys.exit(2)
        prompt_template = load_prompt_template(prompt_path)
    else:
        prompt_template = DEFAULT_PROMPT_TEXT

    model = Qwen25Coder32B(args.model_path)

    any_found = False
    for p in iter_action_logs(expdir):
        any_found = True
        judge_file(p, model, prompt_template)

    if not any_found:
        print(f"[compat] No action_log_trial-*.jsonl found under: {expdir}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
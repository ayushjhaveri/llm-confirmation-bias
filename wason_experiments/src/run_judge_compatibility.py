#!/usr/bin/env python3
# Judge compatibility of (Announce -> next Check) using a local vLLM model.
# Writes <transcript>_judge_compatibility.jsonl next to each input .jsonl.
#
# Example:
# python -m src.run_judge_compatibility \
#   --expdir runs/train/llama-3.3-70b-instruct/baseline \
#   --model-path /scratch/aj4332/models/llama-3.3-70b-instruct \
#   --prompt-file prompts/generate_hypothesis_function.txt
#
# Also supports:
#   --model-path /scratch/aj4332/models/qwen2.5-coder-32b-instruct

from __future__ import annotations
import argparse, json, re, sys, signal, ast, types, os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

# Keep using your existing GenParams shape from the Llama wrapper
from .models.llama33_70b import Llama33_70B, GenParams
from .models.qwen25_coder_32b import Qwen25Coder32B  # new

# ------------------------- file discovery ----------------------------

def iter_transcript_jsonl(expdir: Path) -> Iterable[Path]:
    for p in expdir.rglob("*.jsonl"):
        name = p.name
        if name.endswith("_judge_guesses.jsonl"):
            continue
        if name.endswith("_judge_compatibility.jsonl"):
            continue
        if name.endswith(".model_params.json"):
            continue
        yield p

def read_jsonl(path: Path) -> List[Dict]:
    out = []
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

def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------------- turn parsing ------------------------------

ANN_DAX_RE = re.compile(r"(?mi)^\s*Announce:\s*DAX\s*rule\s*-\s*(.+)\s*$")
ANNOUNCE_RE = re.compile(r"(?mi)^\s*Announce:\s*(.+)\s*$")
CHECK_RE    = re.compile(r"(?mi)^\s*Check:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$")

def extract_meta(doc: List[Dict]) -> Dict:
    if not doc: return {}
    return doc[0].get("meta", {}) if isinstance(doc[0], dict) else {}

def _normalize_announce_text(s: str) -> str:
    t = (s or "").strip()
    if t.lower().startswith("announce:"):
        t = t.split(":", 1)[1].strip()
    if t.lower().startswith("dax rule -"):
        t = t.split("-", 1)[1].strip()
    return t

def iter_model_announces(doc: List[Dict]) -> Iterable[Tuple[int,str]]:
    for i, rec in enumerate(doc):
        if not isinstance(rec, dict):
            continue
        if rec.get("role") != "model":
            continue
        content = (rec.get("content") or "").strip()

        m_dax = ANN_DAX_RE.search(content)
        if m_dax:
            text = m_dax.group(1).strip().splitlines()[0]
            yield (rec.get("step", i), _normalize_announce_text(text))
            continue

        m = ANNOUNCE_RE.search(content)
        if m:
            text = m.group(1).strip().splitlines()[0]
            yield (rec.get("step", i), _normalize_announce_text(text))

def find_next_model_check(doc: List[Dict], start_idx: int) -> Optional[Tuple[int,Tuple[int,int,int]]]:
    for j in range(start_idx + 1, len(doc)):
        rec = doc[j]
        if not isinstance(rec, dict):
            continue
        if rec.get("role") != "model":
            continue
        content = (rec.get("content") or "").strip()
        m = CHECK_RE.search(content)
        if m:
            a,b,c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (rec.get("step", j), (a,b,c))
    return None

def collect_pairs(doc: List[Dict]) -> List[Dict]:
    pairs = []
    for idx, (i, ann_txt) in enumerate(iter_model_announces(doc)):
        nxt = find_next_model_check(doc, i if isinstance(i,int) else idx)
        if nxt is None:
            continue
        j, triple = nxt
        pairs.append({
            "announce_step": i,
            "announced_rule": ann_txt,
            "check_step": j,
            "triple": list(triple),
        })
    return pairs

# ------------------------- prompt loading --------------------------------

def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def render_prompt(tmpl: str, hypothesis: str) -> str:
    return tmpl.replace("{HYPOTHESIS}", hypothesis.strip())

# ------------------------- safe compile/exec -------------------------------

class Timeout(Exception): pass

def _alarm_handler(signum, frame):  # POSIX only
    raise Timeout()

def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t.replace("```", "")
    return t.replace("```","").strip()

def safe_load_rule(func_text: str, compile_timeout_s: int = 2) -> types.FunctionType:
    code_raw = strip_fences(func_text)

    sig = re.search(
        r'^\s*def\s+rule\s*\(\s*a\s*:\s*int\s*,\s*b\s*:\s*int\s*,\s*c\s*:\s*int\s*\)\s*->\s*bool\s*:\s*',
        code_raw, re.MULTILINE
    )
    if not sig:
        raise ValueError("Missing signature: def rule(a: int, b: int, c: int) -> bool:")

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
        "enumerate": enumerate,
        "filter": filter,
        "int": int,
        "isinstance": isinstance,
        "map": map,
        "next": next,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "tuple": tuple,
    }
}
    safe_locals: Dict = {}

    old = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(compile_timeout_s)
        exec(compile(tree, "<model_rule>", "exec"), safe_globals, safe_locals)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    fn = safe_locals.get("rule") or safe_globals.get("rule")
    if not callable(fn):
        raise ValueError("Compiled object `rule` is not callable.")
    return fn

def run_with_timeout(fn, triple: Tuple[int,int,int], run_timeout_s: int = 1) -> bool:
    old = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(run_timeout_s)
        return bool(fn(*triple))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ------------------------- retry helpers ----------------------------------

def render_first_attempt_prompt(tmpl: str, hypothesis: str) -> str:
    base = tmpl.replace("{HYPOTHESIS}", hypothesis.strip())
    return base

def render_retry_messages(
    tmpl: str,
    hypothesis: str,
    bad_output: str,
    python_error: str,
    attempt: int
) -> List[Dict[str, str]]:
    """
    Retry message that now includes the actual compile/runtime error.
    bad_output: previous code from the LLM (may be None)
    python_error: exact Python error string
    """
    first_user = render_first_attempt_prompt(tmpl, hypothesis)
    safe_bad = (bad_output or "").strip()  # <-- handle None safely
    return [
        # original prompt
        {"role": "user", "content": first_user},

        # show model its own previous code (or empty if unavailable)
        {"role": "assistant", "content": safe_bad},

        # show the exact Python error + instruct it to fix
        {
            "role": "user",
            "content": (
                f"The above function FAILED with this Python error:\n"
                f"```\n{python_error}\n```\n\n"
                f"This was attempt {attempt}. "
                "Regenerate a correct function ONLY.\n"
                f"Rule:\n{hypothesis.strip()}"
            )
        },
    ]


def call_llama_function_messages(
    llama_like_model,  # Llama33_70B or Qwen25Coder32B
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_new_tokens: int,
    stop: Optional[List[str]] = None
) -> str:
    params = GenParams(
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        stop=stop or ["<|eot_id|>", "<|end_of_text|>", "</s>", "```"],
    )
    out = llama_like_model.generate(messages, params)
    return out["text"] if isinstance(out, dict) and "text" in out else str(out)

def call_llama_function_once(
    llama_like_model,
    prompt_or_messages,
    *,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    stop: Optional[List[str]] = None
) -> str:
    params = GenParams(
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        stop=stop or ["<|eot_id|>", "<|end_of_text|>", "</s>", "```"],
    )
    out = llama_like_model.generate(prompt_or_messages, params)
    return out["text"] if isinstance(out, dict) and "text" in out else str(out)

def call_llama_function_with_retry(
    llama_like_model,
    prompt_template: str,
    hypothesis: str,
    *,
    max_retries: int = 2
) -> Tuple[str, List[str]]:
    attempts: List[str] = []
    first_prompt = render_first_attempt_prompt(prompt_template, hypothesis)
    raw = call_llama_function_once(
        llama_like_model,
        first_prompt,
        temperature=0.2,
        max_new_tokens=512,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "```"],
    )
    attempts.append(raw)
    return raw, attempts

def retry_llama_function_after_failure(
    llama_like_model,
    prompt_template: str,
    hypothesis: str,
    previous_raw: str,
    python_error: str, 
    attempt_idx: int
) -> str:
    messages = render_retry_messages(prompt_template, hypothesis, previous_raw, python_error, attempt_idx)
    raw = call_llama_function_messages(
        llama_like_model,
        messages,
        temperature=0.15 if attempt_idx == 1 else 0.1,
        max_new_tokens=768 if attempt_idx == 1 else 896,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "```"],
    )
    return raw

def retry_llama_after_runtime_failure(
    model,
    prompt_template: str,
    hypothesis: str,
    previous_code: str,
    runtime_error: str,
    attempt_idx: int,
):
    """
    Retry after runtime error using the same message structure as compile errors.
    """
    messages = render_retry_messages(
        prompt_template,
        hypothesis,
        previous_code,
        runtime_error,
        attempt_idx,
    )
    raw = call_llama_function_messages(
        model,
        messages,
        temperature=0.15 if attempt_idx == 1 else 0.1,
        max_new_tokens=768 if attempt_idx == 1 else 896,
        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "```"],
    )
    return raw

# ------------------------- core judging -------------------------------------

def _debug_log_llm_output(index: int, phase: str, attempt_idx: int, ann_norm: str, text: str) -> None:
    """
    Helper to print LLM outputs with index and rule so you can match them.
    phase ∈ {"initial", "eof_retry", "chat_retry"}.
    """
    try:
        sys.stderr.write(
            f"[compat][llm-output] index={index} phase={phase} attempt={attempt_idx} "
            f"rule={ann_norm!r} len={len(text)}\n{text}\n\n"
        )
        sys.stderr.flush()
    except Exception:
        pass

def judge_file(
    transcript_path: Path,
    model,  # Llama33_70B or Qwen25Coder32B
    prompt_template: str,
    judger_model_name: str,
) -> Path:
    doc = read_jsonl(transcript_path)
    meta = extract_meta(doc)
    pairs = collect_pairs(doc)

    if len(pairs) != 45:
        print(f"[warn] {transcript_path.name}: found {len(pairs)} pairs (expected 45). Proceeding.", file=sys.stderr)

    out_path = transcript_path.with_name(transcript_path.stem + "_judge_compatibility.jsonl")

    rows = []
    rows.append({
        "meta": {
            **meta,
            "judger_model": judger_model_name,
            "judger_task": "announce_to_function_compatibility",
            "prompt_file": "prompts/generate_hypothesis_function.txt",
            "source_transcript": str(transcript_path.name),
            "num_pairs": len(pairs),
        }
    })

    cache: Dict[str, types.FunctionType] = {}
    for k, pair in enumerate(pairs, start=1):
        ann = pair["announced_rule"].rstrip(".").strip()
        ann_norm = _normalize_announce_text(ann)
        triple = tuple(pair["triple"])
        ann_step = pair["announce_step"]
        chk_step = pair["check_step"]

        fn = cache.get(ann_norm)
        compile_ok, compile_err = True, ""
        raw_code = None
        raw_attempts: List[str] = []

        if fn is None:
            # ----- Initial attempt -----
            raw0, attempts = call_llama_function_with_retry(model, prompt_template, ann_norm, max_retries=2)
            raw_attempts.extend(attempts)

            # Log *all* attempts from initial helper (currently only one)
            for a_idx, txt in enumerate(attempts):
                _debug_log_llm_output(k, phase="initial", attempt_idx=a_idx, ann_norm=ann_norm, text=txt)

            raw_code = raw0
            try:
                fn = safe_load_rule(raw_code)
                cache[ann_norm] = fn
            except Exception as e0:
                last_err = str(e0)

                # ---- Retry 0: truncation heuristic ----
                if "EOF" in last_err:
                    try_raw = call_llama_function_once(
                        model,
                        render_first_attempt_prompt(prompt_template, ann_norm),
                        temperature=0.2,
                        max_new_tokens=896,
                        stop=["<|eot_id|>", "<|end_of_text|>", "</s>", "```"],
                    )
                    raw_attempts.append(try_raw)
                    try:
                        fn = safe_load_rule(try_raw)
                        cache[ann_norm] = fn
                        raw_code = try_raw
                        last_err = None
                    except Exception as e_trunc:
                        last_err = str(e_trunc)

                # ---- Retry 1–2: full error-in-feedback compile retries ----
                if fn is None:
                    for attempt_idx in (1, 2):
                        raw_retry = retry_llama_function_after_failure(
                            model,
                            prompt_template,
                            ann_norm,
                            raw_attempts[-1],
                            last_err,
                            attempt_idx,
                        )
                        raw_attempts.append(raw_retry)
                        try:
                            fn = safe_load_rule(raw_retry)
                            cache[ann_norm] = fn
                            raw_code = raw_retry
                            last_err = None
                            break
                        except Exception as e_retry:
                            last_err = str(e_retry)
                            continue

                # Final failure
                if fn is None:
                    compile_ok = False
                    compile_err = last_err or "unknown compile failure"
                    raw_code = raw_attempts[-1] if raw_attempts else None

                    # print(
                    #     f"[compat][compile-fail] index={k} rule={ann_norm!r} error={compile_err}\n{raw_code}\n",
                    #     file=sys.stderr,
                    #     flush=True,
                    # )

        if not compile_ok or fn is None:
            rows.append({
                "index": k,
                "announce_step": ann_step,
                "check_step": chk_step,
                "announced_rule": ann_norm,
                "triple": list(triple),
                "compile_ok": False,
                "compile_error": compile_err,
                "compatible": None,
                "status": "COMPILE_ERROR",
                "raw_fn": raw_code,
                "raw_attempts": raw_attempts,
            })
            continue

        try:
            ok = run_with_timeout(fn, triple, run_timeout_s=1)
            rows.append({
                "index": k,
                "announce_step": ann_step,
                "check_step": chk_step,
                "announced_rule": ann_norm,
                "triple": list(triple),
                "compile_ok": True,
                "compile_error": "",
                "compatible": bool(ok),
                "status": "OK",
            })
        except Timeout:
            rows.append({
                "index": k,
                "announce_step": ann_step,
                "check_step": chk_step,
                "announced_rule": ann_norm,
                "triple": list(triple),
                "compile_ok": True,
                "compile_error": "",
                "compatible": "TIMEOUT",
                "status": "TIMEOUT",
            })
        except Exception as e_runtime:
            runtime_err = str(e_runtime)

            # Try 1–2 LLM retries to fix runtime errors
            fixed = False
            for attempt_idx in (1, 2):
                raw_retry = retry_llama_after_runtime_failure(
                    model,
                    prompt_template,
                    ann_norm,
                    raw_code,
                    runtime_err,
                    attempt_idx,
                )
                raw_attempts.append(raw_retry)

                try:
                    fn2 = safe_load_rule(raw_retry)
                    ok = run_with_timeout(fn2, triple, run_timeout_s=1)
                    # success!
                    rows.append({
                        "index": k,
                        "announce_step": ann_step,
                        "check_step": chk_step,
                        "announced_rule": ann_norm,
                        "triple": list(triple),
                        "compile_ok": True,
                        "compile_error": "",
                        "compatible": bool(ok),
                        "status": "OK_AFTER_RUNTIME_RETRY",
                        "raw_attempts": raw_attempts,
                    })
                    fixed = True
                    break
                except Exception as e2:
                    runtime_err = str(e2)
                    continue

            # If still broken:
            if not fixed:
                rows.append({
                    "index": k,
                    "announce_step": ann_step,
                    "check_step": chk_step,
                    "announced_rule": ann_norm,
                    "triple": list(triple),
                    "compile_ok": True,
                    "compile_error": f"runtime_error: {runtime_err}",
                    "compatible": "ERROR",
                    "status": "ERROR",
                    "raw_attempts": raw_attempts,
                })


    write_jsonl(out_path, rows)
    print(f"[compat] {transcript_path} -> {out_path} ({len(pairs)} pairs)", flush=True)
    return out_path

# ------------------------- CLI ---------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True, help="Experiment directory (recursively scanned)")
    ap.add_argument("--model-path", required=True, help="Path to local model dir (llama-3.3-70b-instruct or qwen2.5-coder-32b-instruct)")
    ap.add_argument("--prompt-file", default="prompts/generate_hypothesis_function.txt",
                    help="Prompt file for converting Announce → function")
    args = ap.parse_args()

    expdir = Path(args.expdir).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    if not prompt_path.exists():
        print(f"[error] Missing prompt file: {prompt_path}", file=sys.stderr)
        sys.exit(2)

    tmpl = load_prompt_template(prompt_path)

    base = os.path.basename(args.model_path.rstrip("/")).lower()
    if "qwen2.5-coder-32b-instruct" in base or ("qwen2.5" in base and "coder" in base):
        model = Qwen25Coder32B(args.model_path)
    else:
        model = Llama33_70B(args.model_path)

    judger_model_name = base or "local-model"

    any_found = False
    for transcript in iter_transcript_jsonl(expdir):
        any_found = True
        judge_file(transcript, model, tmpl, judger_model_name)

    if not any_found:
        print(f"[compat] No transcript jsonl files found under: {expdir}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()

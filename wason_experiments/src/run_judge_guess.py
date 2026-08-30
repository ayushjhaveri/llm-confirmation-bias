# src/run_judge_guess.py
"""
Judge announced rules in a whole experiment directory using Llama 3.3 70B.

- Reuses your existing prompt templates (prompts/*.txt) and guidance (core/judge_guidance.py).
- Uses the same model wrapper as run_experiments.py:
      from .models.llama33_70b import Llama33_70B, GenParams
- Writes one output jsonl per transcript, named:
      <basename_without_.jsonl>_judge_guesses.jsonl
  in the SAME folder as the transcript.

Example:
  python -m src.run_judge_guess \
      --expdir runs/baseline \
      --model-path /scratch/aj4332/models/llama-3.3-70b-instruct \
      [--prompts-dir prompts] [--judge-temp 0.0] [--judge-max-tokens 12]
"""

from __future__ import annotations
import argparse, json, re, sys, dataclasses
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Use the same llama wrapper as run_experiments.py
from .models.llama33_70b import Llama33_70B, GenParams

# Reuse your judge guidance (already tested)
from .core.judge_guidance import guidance_for_rule


# ------------------------- helpers: file discovery ----------------------------

# def iter_transcript_jsonl(expdir: Path) -> Iterable[Path]:
#     """
#     Yield every transcript .jsonl file we should judge.
#     Heuristics:
#       - First line is a meta line: {"meta": {...}}
#       - Skip files that already look like judge outputs ("_judge_guesses.jsonl").
#       - Skip model_params files.
#     """
#     for p in expdir.rglob("*.jsonl"):
        # name = p.name
        # if name.endswith("_judge_guesses.jsonl"):
        #     continue
        # if name.endswith("_judge_compatibility.jsonl"):
        #     continue
        # if name.endswith(".model_params.json"):
        #     continue
        # try:
        #     with p.open("r", encoding="utf-8") as f:
        #         first = f.readline()
        #     if first.strip().startswith('{"meta"'):
        #         yield p
        # except Exception:
            # continue

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


# ------------------------- helpers: parsing turns -----------------------------

_ANNOUNCE_RE = re.compile(r"(?mi)^Announce:\s*(.+)\s*$")
# Dual-goal two-line announce
_ANN_DAX_RE = re.compile(r"(?mi)^Announce:\s*DAX\s*rule\s*-\s*(.+)$")

def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # be tolerant; skip broken lines
                continue
    return out

def _extract_meta(doc: List[Dict]) -> Dict:
    if not doc:
        return {}
    first = doc[0]
    return first.get("meta", {}) if "meta" in first else {}

def _variant_from_meta(meta: Dict) -> str:
    # expected: "baseline" | "dual-goal" | "think-in-opposites"
    return str(meta.get("variant", "")).strip().lower()

def _hidden_rule_name(meta: Dict) -> str:
    return str(meta.get("hidden_rule_name", "")).strip()

def _iter_model_announces(doc: List[Dict], variant: str) -> Iterable[Tuple[int, str]]:
    """
    Yield (step_index, announce_text) for each announce turn produced by the model.
    baseline/think: single-line "Announce: ...".
    dual-goal: take ONLY the DAX rule line (judge prompt is DAX-focused).
    """
    steps = 0
    for rec in doc[1:]:  # skip meta
        steps += 1
        role = rec.get("role", "")
        content = rec.get("content", "") or ""
        if role != "model":
            continue

        if variant == "dual-goal":
            m = _ANN_DAX_RE.search(content)
            if m:
                dax_txt = m.group(1).strip().splitlines()[0]
                yield (steps, dax_txt)
            else:
                m_any = _ANNOUNCE_RE.search(content)
                if m_any:
                    yield (steps, m_any.group(1).strip().splitlines()[0])
        else:
            m = _ANNOUNCE_RE.search(content)
            if m:
                yield (steps, m.group(1).strip().splitlines()[0])


# ------------------------- prompts & rendering -------------------------------

def _load_prompt_text(variant: str, prompts_dir: Path) -> str:
    """
    Load the appropriate judge prompt template from your existing `prompts/` files.
    - baseline / think-in-opposites -> judge_equivalence_wason.txt
    - dual-goal -> judge_equivalence_daxmed.txt
    These files are already tested in your repo.
    """
    if variant == "dual-goal":
        fname = "judge_equivalence_daxmed.txt"
    else:
        fname = "judge_equivalence_wason.txt"
    p = prompts_dir / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing prompt template: {p}")
    return p.read_text(encoding="utf-8")

def _render_prompt(tmpl: str, announced_rule: str, ground_truth_rule: str, rule_guidance: str) -> str:
    """
    Your tested templates expect at least {announced_rule} and {ground_truth_rule}.
    If they include {rule_guidance}, we fill it; otherwise the replace is a no-op.
    """
    s = tmpl
    s = s.replace("{announced_rule}", announced_rule)
    s = s.replace("{ground_truth_rule}", ground_truth_rule)
    s = s.replace("{rule_guidance}", rule_guidance)
    return s


# ------------------------- LLM call & parsing --------------------------------

_CODE_FENCE = re.compile(r"^```[^\n]*\n|\n```$", re.MULTILINE)
_YES_NO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    # Remove all fenced blocks markers but keep content
    # First, simple leading fence
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t.replace("```", "")
    # Remove any trailing ``` and incidental fences
    t = t.replace("```", "")
    return t.strip()

def _parse_yes_no(text: str) -> Optional[str]:
    t = _strip_code_fences(text)
    matches = _YES_NO.findall(t)
    if not matches:
        return None
    # Take the *last* occurrence — the model's final commitment
    return matches[-1].upper()

def _call_model_yes_no(llama: Llama33_70B, prompt: str, temperature: float, max_tokens: int) -> Tuple[Optional[str], str]:
    """
    Call Llama with GenParams and parse a YES/NO if present.
    Returns (parsed, raw_text) where parsed in {"YES","NO",None}.
    """
    # Decoding: simple, robust defaults that mirror your runs
    params = GenParams(
        temperature=float(temperature),
        top_p=0.95,
        top_k=50,
        max_new_tokens=int(max_tokens),
        repetition_penalty=1.0,
    )
    raw = llama.generate(prompt, params)
    if isinstance(raw, dict) and "text" in raw:
        raw_text = str(raw["text"])
    else:
        raw_text = str(raw)

    parsed = _parse_yes_no(raw_text)
    return parsed, raw_text


# ----------------------------- main judging loop ------------------------------

def judge_file(
    transcript_path: Path,
    llama: Llama33_70B,
    judge_temp: float,
    judge_max_tokens: int,
    prompts_dir: Path,
) -> Path:
    """
    For a single transcript .jsonl, run judge on each announcement and write:
      <basename_without_.jsonl>_judge_guesses.jsonl
    Returns the output path.
    """
    doc = _read_jsonl(transcript_path)
    meta = _extract_meta(doc)
    variant = _variant_from_meta(meta)
    gt_rule_name = _hidden_rule_name(meta)

    tmpl = _load_prompt_text(variant, prompts_dir)
    guidance = guidance_for_rule(gt_rule_name)

    out_path = transcript_path.with_name(transcript_path.stem + "_judge_guesses.jsonl")
    count = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for step_idx, announced in _iter_model_announces(doc, variant):
            prompt = _render_prompt(tmpl, announced, gt_rule_name, guidance)
            parsed, raw_text = _call_model_yes_no(
                llama=llama,
                prompt=prompt,
                temperature=judge_temp,
                max_tokens=judge_max_tokens,
            )
            outcome = (
                "CORRECT" if parsed == "YES"
                else "INCORRECT" if parsed == "NO"
                else "UNPARSEABLE"
            )

            fout.write(json.dumps({
                "step": step_idx,
                "variant": variant,
                "hidden_rule_name": gt_rule_name,
                "announced_rule": announced,
                "judge_prompt": prompt,
                "judge_raw": raw_text,
                "judge_parsed": parsed,        # "YES"/"NO"/None
                "outcome": outcome,
            }) + "\n")
            count += 1

    print(f"[judge] {transcript_path} -> {out_path} ({count} announcements)", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True, help="Experiment directory (e.g., runs/baseline)")
    ap.add_argument("--model-path", required=True, help="Path to llama-3.3-70b-instruct weights/config")
    ap.add_argument("--prompts-dir", default="prompts", help="Directory with judge prompt templates")
    ap.add_argument("--judge-temp", type=float, default=0.0)
    ap.add_argument("--judge-max-tokens", type=int, default=12)
    args = ap.parse_args()

    expdir = Path(args.expdir).expanduser().resolve()
    prompts_dir = Path(args.prompts_dir).expanduser().resolve()

    # Same model class used in run_experiments.py
    llama = Llama33_70B(args.model_path)

    any_found = False
    for transcript in iter_transcript_jsonl(expdir):
        any_found = True
        judge_file(
            transcript_path=transcript,
            llama=llama,
            judge_temp=args.judge_temp,
            judge_max_tokens=args.judge_max_tokens,
            prompts_dir=prompts_dir,
        )

    if not any_found:
        print(f"[judge] No transcript jsonl files found under: {expdir}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

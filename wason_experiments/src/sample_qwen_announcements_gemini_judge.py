from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .experiments.parse_utils import extract_visible


_ANNOUNCE_RE = re.compile(r"(?mi)^Announce:\s*(.+)\s*$")
_ANN_DAX_RE = re.compile(r"(?mi)^Announce:\s*DAX\s*rule\s*-\s*(.+)$")
_ANN_MED_RE = re.compile(r"(?mi)^Announce:\s*MED\s*rule\s*-\s*(.+)$")
_THINK_PAIR = re.compile(r"(?is)<\s*think\b[^>]*>.*?</\s*think\s*>")
_THINK_CLOSE = re.compile(r"</\s*think\s*>", re.IGNORECASE)
_YES_NO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {e}") from e
    return rows


def iter_transcripts(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.jsonl")):
        name = p.name
        if name.endswith("_judge_guesses.jsonl"):
            continue
        if name.endswith("_judge_compatibility.jsonl"):
            continue
        if name.endswith("_resampled_announcements_gemini_judge.jsonl"):
            continue
        yield p


def clean_after_think(text: str) -> str:
    s = str(text or "")
    if _THINK_CLOSE.search(s):
        s = _THINK_CLOSE.split(s)[-1]
    s = _THINK_PAIR.sub("", s)
    return s.strip()


def parse_announcement(text: str, variant: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (full_announcement, compare_rule, parse_error).

    For dual-goal, compare_rule is only the DAX rule. For the other variants,
    compare_rule is the one announced rule.
    """
    visible = clean_after_think(text)
    variant = variant.lower()

    if variant == "dual-goal":
        m_dax = _ANN_DAX_RE.search(visible)
        m_med = _ANN_MED_RE.search(visible)
        if m_dax:
            dax = m_dax.group(1).strip().splitlines()[0]
            full = visible.strip()
            if m_med:
                med = m_med.group(1).strip().splitlines()[0]
                full = f"Announce: DAX rule - {dax}\nAnnounce: MED rule - {med}"
            return full, dax, None
        m_any = _ANNOUNCE_RE.search(visible)
        if m_any:
            announced = m_any.group(1).strip().splitlines()[0]
            return f"Announce: {announced}", announced, None
        return None, None, "could not parse dual-goal DAX announcement"

    parsed = extract_visible(visible, strategy="auto_think")
    if parsed is not None and parsed[0] == "announce":
        announced = str(parsed[1]).strip().splitlines()[0]
        return f"Announce: {announced}", announced, None

    m_any = _ANNOUNCE_RE.search(visible)
    if m_any:
        announced = m_any.group(1).strip().splitlines()[0]
        return f"Announce: {announced}", announced, None
    return None, None, "could not parse single-rule announcement"


def transcript_messages_before(rows: List[Dict[str, Any]], rec_index: int) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for rec in rows[1:rec_index]:
        role = rec.get("role")
        content = rec.get("content")
        if not isinstance(content, str):
            continue
        if role == "environment":
            messages.append({"role": "user", "content": content})
        elif role == "model":
            messages.append({"role": "assistant", "content": content})
    return messages


def iter_original_announcements(
    rows: List[Dict[str, Any]],
    max_announcements: Optional[int],
    selected_steps: Optional[Set[int]] = None,
) -> Iterable[Tuple[int, Dict[str, Any], List[Dict[str, str]], str, str]]:
    meta = rows[0].get("meta", {}) if rows and "meta" in rows[0] else {}
    variant = str(meta.get("variant", "")).lower()
    count = 0

    for rec_index, rec in enumerate(rows):
        if rec_index == 0 or rec.get("role") != "model":
            continue
        step = int(rec.get("step", -1))
        if selected_steps is not None and step not in selected_steps:
            continue
        full, compare_rule, err = parse_announcement(str(rec.get("content", "")), variant)
        if err or full is None or compare_rule is None:
            continue
        count += 1
        if max_announcements is not None and count > max_announcements:
            break
        yield rec_index, rec, transcript_messages_before(rows, rec_index), full, compare_rule


def render_judge_prompt(
    variant: str,
    original_rule: str,
    sampled_rule: str,
) -> str:
    if variant == "dual-goal":
        return (
            "You are a strict evaluator of SHORT RULE DESCRIPTIONS for number triples.\n\n"
            "Context:\n"
            "- This is the DAX-MED experiment. DAX is the positive class.\n"
            "- Compare ONLY the DAX rule meaning. Ignore any MED rule text.\n\n"
            "Task: Decide if the SAMPLED DAX RULE has the SAME MEANING as the "
            "ORIGINAL DAX RULE.\n\n"
            "Rules:\n"
            "- Ignore superficial phrasing/synonyms if the meaning is identical.\n"
            "- If scopes differ (broader or narrower) or it is a different family, answer NO.\n\n"
            "Respond with ONLY one token: YES or NO.\n\n"
            f'ORIGINAL DAX RULE:\n"{original_rule}"\n\n'
            f'SAMPLED DAX RULE:\n"{sampled_rule}"\n'
        )

    return (
        "You are a strict evaluator of SHORT RULE DESCRIPTIONS for number triples.\n\n"
        "Task: Decide if the SAMPLED ANNOUNCED RULE has the SAME MEANING as the "
        "ORIGINAL ANNOUNCED RULE.\n\n"
        "Rules:\n"
        "- Ignore superficial phrasing/synonyms if the meaning is identical.\n"
        "- If scopes differ (broader or narrower) or it is a different family, answer NO.\n\n"
        "Respond with ONLY one token: YES or NO.\n\n"
        f'ORIGINAL ANNOUNCED RULE:\n"{original_rule}"\n\n'
        f'SAMPLED ANNOUNCED RULE:\n"{sampled_rule}"\n'
    )


def parse_yes_no(text: str) -> Optional[str]:
    matches = _YES_NO.findall(str(text or ""))
    return matches[-1].upper() if matches else None


def load_completed(out_path: Path) -> Set[Tuple[str, int, int]]:
    completed: Set[Tuple[str, int, int]] = set()
    if not out_path.exists():
        return completed
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "sample":
                continue
            completed.add((
                str(row.get("source_transcript", "")),
                int(row.get("announce_step", -1)),
                int(row.get("sample_index", -1)),
            ))
    return completed


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def judge_sample(
    gemini: Any,
    variant: str,
    original_rule: str,
    sampled_rule: str,
    judge_max_tokens: int,
) -> Tuple[Optional[str], str, str]:
    from .models.gemini_2_5_pro import GenParams as GeminiGenParams

    prompt = render_judge_prompt(variant, original_rule, sampled_rule)
    params = GeminiGenParams(
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_new_tokens=judge_max_tokens,
        repetition_penalty=None,
        stop=["```"],
    )
    raw = gemini.generate(prompt, params)
    return parse_yes_no(raw), str(raw), prompt


def announcement_retry_hint(variant: str) -> str:
    if variant == "dual-goal":
        return (
            "Turn - Announce\n"
            "Remember: Output EXACTLY TWO LINES:\n"
            "Announce: DAX rule - <one short sentence>\n"
            "Announce: MED rule - <one short sentence>\n"
            "No extra text."
        )
    return (
        "Turn - Announce\n"
        "Remember: Please do not test a triple! This is an ANNOUNCE turn. Output EXACTLY ONE LINE:\n"
        "Announce: <one short sentence naming the rule>\n"
        "No extra text."
    )


def sample_announcement_with_retry(
    qwen: Any,
    messages: List[Dict[str, str]],
    qwen_params: Any,
    variant: str,
    max_retries: int = 3,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], List[Dict[str, Any]]]:
    retry_trace: List[Dict[str, Any]] = []
    hint = announcement_retry_hint(variant)

    for attempt in range(1, max_retries + 2):
        msgs_try = messages if attempt == 1 else messages + [{"role": "user", "content": hint}]
        raw = qwen.generate(msgs_try, qwen_params)
        sampled_full, sampled_rule, parse_err = parse_announcement(raw, variant)
        retry_trace.append({
            "attempt": attempt,
            "used_retry_hint": attempt > 1,
            "parsed": sampled_rule is not None,
            "parse_error": parse_err,
            "raw": raw,
        })
        if sampled_rule is not None:
            return raw, sampled_full, sampled_rule, None, retry_trace

    last = retry_trace[-1]
    return str(last["raw"]), None, None, str(last["parse_error"]), retry_trace


def preflight_gemini(gemini: Any, judge_max_tokens: int) -> None:
    from .models.gemini_2_5_pro import GenParams as GeminiGenParams

    preflight_max_tokens = max(int(judge_max_tokens), 1024)
    params = GeminiGenParams(
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_new_tokens=preflight_max_tokens,
        repetition_penalty=None,
        stop=["```"],
    )
    raw = gemini.generate(
        "Respond with ONLY one token: YES or NO.\n\nAre these rules equivalent?\n"
        'Rule A: "all numbers are even"\n'
        'Rule B: "each number is even"',
        params,
    )
    parsed = parse_yes_no(raw)
    if parsed not in {"YES", "NO"}:
        raise RuntimeError(f"Gemini preflight returned an unparseable response: {raw!r}")


def build_qwen_model(name: str, model_path: str) -> Any:
    n = name.lower()
    if n in ["qwen3-8b", "qwen3_8b", "qwen3-8b-instruct"]:
        from .models.qwen3_8b import Qwen3_8B

        return Qwen3_8B(model_path=model_path)
    raise ValueError(f"Unsupported sampling model for this script: {name}")


def collect_announcement_candidates(input_root: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for transcript in iter_transcripts(input_root):
        rows = read_jsonl(transcript)
        meta = rows[0].get("meta", {}) if rows and "meta" in rows[0] else {}
        for _, rec, _, original_full, original_rule in iter_original_announcements(
            rows,
            max_announcements=None,
            selected_steps=None,
        ):
            candidates.append({
                "transcript": transcript,
                "step": int(rec.get("step", -1)),
                "variant": meta.get("variant"),
                "testcase_id": meta.get("testcase_id"),
                "instance_index": meta.get("instance_index"),
                "original_announcement": original_full,
                "original_compare_rule": original_rule,
            })
    return candidates


def select_random_announcement_steps(
    input_root: Path,
    total_announcements: int,
    random_seed: int,
    output_root: Path,
) -> Dict[Path, Set[int]]:
    candidates = collect_announcement_candidates(input_root)
    if total_announcements > len(candidates):
        raise ValueError(
            f"Requested {total_announcements} announcements, but only found {len(candidates)}"
        )

    rng = random.Random(random_seed)
    selected = rng.sample(candidates, total_announcements)
    selected.sort(key=lambda x: (str(x["transcript"]), int(x["step"])))

    by_transcript: Dict[Path, Set[int]] = {}
    for item in selected:
        by_transcript.setdefault(item["transcript"], set()).add(int(item["step"]))

    manifest_path = output_root / "selection_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "type": "meta",
            "input_root": str(input_root),
            "total_candidates": len(candidates),
            "selected_announcements": total_announcements,
            "random_seed": random_seed,
        }, ensure_ascii=False) + "\n")
        for i, item in enumerate(selected, start=1):
            row = dict(item)
            row["type"] = "selected_announcement"
            row["selection_index"] = i
            row["transcript"] = str(row["transcript"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[selection] wrote {manifest_path} ({len(selected)} announcements)", flush=True)
    return by_transcript


def process_transcript(
    transcript: Path,
    input_root: Path,
    output_root: Path,
    qwen: Any,
    gemini: Any,
    samples_per_announcement: int,
    max_announcements_per_file: Optional[int],
    selected_announcement_steps: Optional[Set[int]],
    qwen_max_tokens: int,
    judge_max_tokens: int,
) -> Path:
    rows = read_jsonl(transcript)
    meta = rows[0].get("meta", {}) if rows and "meta" in rows[0] else {}
    variant = str(meta.get("variant", "")).lower()
    rel = transcript.relative_to(input_root)
    out_path = (output_root / rel).with_name(transcript.stem + "_resampled_announcements_gemini_judge.jsonl")
    completed = load_completed(out_path)

    if not out_path.exists():
        append_jsonl(out_path, {
            "type": "meta",
            "source_transcript": str(transcript),
            "source_meta": meta,
            "samples_per_announcement": samples_per_announcement,
            "selected_announcement_steps": sorted(selected_announcement_steps) if selected_announcement_steps is not None else None,
            "qwen_decoding": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_new_tokens": qwen_max_tokens,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.0,
                "min_p": 0.0,
            },
            "judge_model": gemini.name(),
            "judge_decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": judge_max_tokens,
                "stop": ["```"],
            },
        })

    from .models.qwen3_8b import GenParamsCompat as QwenGenParams

    for _, rec, messages, original_full, original_rule in iter_original_announcements(
        rows,
        max_announcements=max_announcements_per_file,
        selected_steps=selected_announcement_steps,
    ):
        step = int(rec.get("step", -1))
        same_count = 0
        judged_count = 0

        for sample_index in range(1, samples_per_announcement + 1):
            key = (str(transcript), step, sample_index)
            if key in completed:
                continue

            qwen_params = QwenGenParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_new_tokens=qwen_max_tokens,
                repetition_penalty=1.0,
                presence_penalty=1.0,
                min_p=0.0,
            )
            raw, sampled_full, sampled_rule, parse_err, sample_retry_trace = sample_announcement_with_retry(
                qwen=qwen,
                messages=messages,
                qwen_params=qwen_params,
                variant=variant,
            )

            judge_parsed: Optional[str] = None
            judge_raw = ""
            judge_prompt = ""
            judge_error = None
            if sampled_rule is not None:
                try:
                    judge_parsed, judge_raw, judge_prompt = judge_sample(
                        gemini=gemini,
                        variant=variant,
                        original_rule=original_rule,
                        sampled_rule=sampled_rule,
                        judge_max_tokens=judge_max_tokens,
                    )
                except Exception as e:
                    judge_error = repr(e)
                if judge_parsed is not None:
                    judged_count += 1
                    same_count += int(judge_parsed == "YES")

            append_jsonl(out_path, {
                "type": "sample",
                "source_transcript": str(transcript),
                "variant": variant,
                "testcase_id": meta.get("testcase_id"),
                "hidden_rule_name": meta.get("hidden_rule_name"),
                "instance_index": meta.get("instance_index"),
                "announce_step": step,
                "sample_index": sample_index,
                "history_messages": messages,
                "original_announcement": original_full,
                "original_compare_rule": original_rule,
                "sample_raw": raw,
                "sample_announcement": sampled_full,
                "sample_compare_rule": sampled_rule,
                "sample_parse_error": parse_err,
                "sample_retry_trace": sample_retry_trace,
                "judge_prompt": judge_prompt,
                "judge_raw": judge_raw,
                "judge_parsed": judge_parsed,
                "judge_error": judge_error,
                "same_meaning": None if judge_parsed is None else judge_parsed == "YES",
            })
            if judge_error is not None:
                raise RuntimeError(f"Gemini judge failed for {transcript} step {step} sample {sample_index}: {judge_error}")

        append_jsonl(out_path, {
            "type": "announcement_summary",
            "source_transcript": str(transcript),
            "announce_step": step,
            "original_announcement": original_full,
            "original_compare_rule": original_rule,
            "samples_requested": samples_per_announcement,
            "newly_judged_samples": judged_count,
            "newly_same_meaning": same_count,
        })

    print(f"[done] {transcript} -> {out_path}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="runs/ood_test/qwen3-8b")
    ap.add_argument("--output-root", default="runs/ood_test/qwen3-8b_resampled_gemini_judge")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gemini-model", default="@vertexai-gemini-ec5413/gemini-2.5-pro")
    ap.add_argument("--samples-per-announcement", type=int, default=10)
    ap.add_argument("--total-announcements", type=int, default=None, help="Randomly sample this many original announce turns across input-root")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--max-announcements-per-file", type=int, default=None)
    ap.add_argument("--announcement-steps", default=None, help="Optional comma-separated transcript step numbers to sample")
    ap.add_argument("--qwen-max-tokens", type=int, default=16384)
    ap.add_argument("--judge-max-tokens", type=int, default=1024)
    ap.add_argument("--transcript", default=None, help="Optional single transcript path for array jobs/debugging")
    ap.add_argument("--skip-gemini-preflight", action="store_true")
    args = ap.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    from .models.gemini_2_5_pro import Gemini25Pro

    gemini = Gemini25Pro(model=args.gemini_model)
    if not args.skip_gemini_preflight:
        try:
            preflight_gemini(gemini, args.judge_max_tokens)
        except Exception as e:
            raise RuntimeError(
                "Gemini preflight failed before loading Qwen. Check PORTKEY_API_KEY/"
                "PORTKEY_API_TOKEN, PORTKEY_BASE_URL, and --gemini-model access. "
                f"Configured Gemini model: {args.gemini_model}. Original error: {e!r}"
            ) from e

    selected_by_transcript: Optional[Dict[Path, Set[int]]] = None
    if args.total_announcements is not None:
        if args.transcript:
            raise ValueError("--total-announcements samples across --input-root, so do not combine it with --transcript")
        if args.announcement_steps:
            raise ValueError("--total-announcements selects steps randomly, so do not combine it with --announcement-steps")
        selected_by_transcript = select_random_announcement_steps(
            input_root=input_root,
            total_announcements=args.total_announcements,
            random_seed=args.random_seed,
            output_root=output_root,
        )
        transcripts = sorted(selected_by_transcript)
    elif args.transcript:
        transcripts = [Path(args.transcript).expanduser().resolve()]
    else:
        transcripts = list(iter_transcripts(input_root))

    if not transcripts:
        print(f"No transcript JSONL files found under {input_root}", file=sys.stderr)
        sys.exit(2)

    selected_steps = None
    if args.announcement_steps:
        selected_steps = {int(x.strip()) for x in args.announcement_steps.split(",") if x.strip()}

    qwen = build_qwen_model(args.model, args.model_path)

    for transcript in transcripts:
        selected_for_transcript = (
            selected_by_transcript.get(transcript)
            if selected_by_transcript is not None
            else selected_steps
        )
        process_transcript(
            transcript=transcript,
            input_root=input_root,
            output_root=output_root,
            qwen=qwen,
            gemini=gemini,
            samples_per_announcement=args.samples_per_announcement,
            max_announcements_per_file=args.max_announcements_per_file,
            selected_announcement_steps=selected_for_transcript,
            qwen_max_tokens=args.qwen_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
        )


if __name__ == "__main__":
    main()

# eval_aime2025.py
import argparse
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from models.qwen3_8b import Qwen3_8B
from models.qwen3_32b import Qwen3_32B


BOXED_RE = re.compile(r"\\boxed\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}")
SOLUTION_INSTRUCTION = (
    "Give concise final working with only the key steps, then write your final answer after '####'."
)


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = ", ".join(str(x) for x in text)

    text = strip_thinking(str(text))
    text = text.strip().strip("$")
    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace(",", "")
    return text.lower()


def parse_numeric_answer(text: Any) -> Fraction | None:
    if text is None:
        return None

    text = strip_thinking(str(text)).strip().strip("$")
    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = text.replace(",", "")

    frac_matches = re.findall(r"\\frac\s*{?\s*(-?\d+)\s*}?\s*{?\s*(-?\d+)\s*}?", text)
    if frac_matches:
        numerator, denominator = frac_matches[-1]
        if int(denominator) != 0:
            return Fraction(int(numerator), int(denominator))

    plain_frac_matches = re.findall(r"(?<![\w.])-?\d+\s*/\s*-?\d+(?![\w.])", text)
    if plain_frac_matches:
        try:
            return Fraction(plain_frac_matches[-1].replace(" ", ""))
        except ZeroDivisionError:
            return None

    number_matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)
    if number_matches:
        return Fraction(number_matches[-1])

    return None


def extract_answer(text: str) -> str | None:
    text = strip_thinking(text)

    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    hash_answers = re.findall(r"^####\s*(.+)$", text, flags=re.MULTILINE)
    if hash_answers:
        return hash_answers[-1].strip()

    final_patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*(.+)",
        r"(?:therefore|thus),?\s*(.+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def get_field(item: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def get_problem(item: dict) -> str:
    value = get_field(item, ("problem", "question", "prompt", "input"))
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return str(value[0].get("content", ""))
    return str(value or "")


def get_gold_answers(item: dict) -> list[str]:
    for key in ("answer", "target", "final_answer", "ground_truth", "answers"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]
        if isinstance(value, dict):
            for nested_key in ("ground_truth", "answer", "target"):
                nested = value.get(nested_key)
                if nested is not None and str(nested).strip():
                    return [str(nested)]
        if str(value).strip():
            return [str(value)]

    reward_model = item.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return [str(reward_model["ground_truth"])]

    return []


def is_correct_prediction(pred: str | None, gold_answers: list[str]) -> bool:
    if pred is None or not gold_answers:
        return False

    pred_norm = normalize_answer(pred)
    gold_norms = [normalize_answer(gold) for gold in gold_answers]
    if pred_norm in gold_norms:
        return True

    pred_value = parse_numeric_answer(pred)
    if pred_value is None:
        return False

    return any(parse_numeric_answer(gold) == pred_value for gold in gold_answers)


def build_prompt(item: dict, benchmark_name: str = "AIME 2025") -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Solve the following problem. "
                f"{SOLUTION_INSTRUCTION}\n\n"
                f"Problem:\n{get_problem(item)}"
            ),
        }
    ]


def build_retry_message(within_100_words: bool = False) -> dict:
    word_limit = " Respond within 100 words." if within_100_words else ""
    return {
        "role": "user",
        "content": (
            "Your answer was incorrect. Please try again. "
            "Review your previous working, fix the mistake, and keep the corrected working concise. "
            f"{SOLUTION_INSTRUCTION}{word_limit}"
        ),
    }


def load_model(model_name: str, model_path: str):
    if model_name == "qwen3-8b":
        return Qwen3_8B(model_path=model_path)
    if model_name == "qwen3-32b":
        return Qwen3_32B(model_path=model_path)
    raise ValueError(f"Unknown model: {model_name}. Choose from: qwen3-8b, qwen3-32b")


def load_aime2025(dataset_name: str, dataset_config: str | None, split: str):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    if dataset_name == "opencompass/AIME2025":
        return concatenate_datasets(
            [
                load_dataset(dataset_name, "AIME2025-I", split=split),
                load_dataset(dataset_name, "AIME2025-II", split=split),
            ]
        )
    return load_dataset(dataset_name, split=split)


def build_eval_indices(
    dataset_size: int,
    num_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[int]:
    indices = list(range(dataset_size))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    if num_samples is not None:
        indices = indices[:num_samples]
    return indices


def load_completed_results(jsonl_path: Path | None, output_path: Path | None) -> dict[int, dict]:
    completed = {}

    if output_path and output_path.exists():
        with output_path.open() as f:
            data = json.load(f)
        for result in data.get("results", []):
            if "idx" in result:
                completed[int(result["idx"])] = result

    if jsonl_path and jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                if "idx" in result:
                    completed[int(result["idx"])] = result

    return completed


def summarize_results(results: list[dict], max_retries: int, target_total: int) -> dict:
    completed = len(results)
    correct_by_retry = {}
    accuracy_by_retry = {}

    for retry_level in range(max_retries + 1):
        correct = 0
        for result in results:
            turns = result.get("turns") or []
            gold_answers = result.get("gold") or []
            solved = any(
                int(turn.get("attempt", 0)) <= retry_level
                and is_correct_prediction(turn.get("pred"), gold_answers)
                for turn in turns
            )
            if not turns:
                solved = retry_level == 0 and is_correct_prediction(result.get("pred"), gold_answers)
            correct += int(solved)

        key = str(retry_level)
        correct_by_retry[key] = correct
        accuracy_by_retry[key] = correct / completed if completed else 0.0

    return {
        "accuracy": accuracy_by_retry[str(max_retries)] if completed else 0.0,
        "accuracy_by_retry": accuracy_by_retry,
        "correct_by_retry": correct_by_retry,
        "correct": correct_by_retry[str(max_retries)] if completed else 0,
        "completed": completed,
        "total": completed,
        "target_total": target_total,
        "max_retries": max_retries,
        "results": sorted(results, key=lambda result: int(result["idx"])),
    }


def write_summary(output_path: Path | None, results: list[dict], max_retries: int, target_total: int):
    if not output_path:
        return
    summary = summarize_results(results, max_retries=max_retries, target_total=target_total)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)


def evaluate_one_item(
    model,
    item: dict,
    idx: int,
    dataset_idx: int,
    max_retries: int,
    retry_within_100_words: bool = False,
    benchmark_name: str = "AIME 2025",
) -> dict:
    item_dict = dict(item)
    gold_answers = get_gold_answers(item_dict)
    messages = build_prompt(item_dict, benchmark_name=benchmark_name)
    turns = []

    for attempt in range(max_retries + 1):
        raw_response = model.generate(messages)
        response = strip_thinking(raw_response)
        pred = extract_answer(response)
        is_correct = is_correct_prediction(pred, gold_answers)

        turns.append(
            {
                "attempt": attempt,
                "user": messages[-1]["content"] if messages and messages[-1]["role"] == "user" else None,
                "response": response,
                "pred": pred,
                "correct": is_correct,
            }
        )

        if is_correct or attempt == max_retries:
            break

        messages.append({"role": "assistant", "content": response})
        messages.append(build_retry_message(within_100_words=retry_within_100_words))

    first_correct_attempt = next(
        (turn["attempt"] for turn in turns if turn.get("correct", False)),
        None,
    )
    final_turn = turns[-1]

    return {
        "idx": idx,
        "dataset_idx": dataset_idx,
        "id": item_dict.get("id"),
        "problem": get_problem(item_dict),
        "gold": gold_answers,
        "pred": final_turn.get("pred"),
        "correct": first_correct_attempt is not None,
        "first_correct_attempt": first_correct_attempt,
        "num_attempts": len(turns),
        "response": final_turn.get("response"),
        "turns": turns,
    }


def evaluate(
    model,
    dataset,
    num_samples: int | None = None,
    output_file: str | None = None,
    max_retries: int = 5,
    shuffle: bool = False,
    seed: int = 0,
    resume: bool = True,
    first_resumed_retry_within_100_words: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
    benchmark_name: str = "AIME 2025",
    eval_start: int | None = None,
    eval_end: int | None = None,
):
    eval_indices = build_eval_indices(len(dataset), num_samples=num_samples, shuffle=shuffle, seed=seed)
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    eval_indices = eval_indices[shard_id::num_shards]
    if eval_start is not None or eval_end is not None:
        start = 0 if eval_start is None else eval_start
        end = len(eval_indices) if eval_end is None else eval_end
        if start < 0 or end < start or end > len(eval_indices):
            raise ValueError(
                f"Invalid evaluation slice [{start}:{end}] for shard size "
                f"{len(eval_indices)}"
            )
        eval_indices = eval_indices[start:end]

    output_path = Path(output_file) if output_file else None
    jsonl_path = output_path.with_suffix(".jsonl") if output_path else None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_indices = set(range(len(eval_indices)))
    completed_by_idx = load_completed_results(jsonl_path, output_path) if resume else {}
    completed_by_idx = {
        idx: result for idx, result in completed_by_idx.items() if idx in valid_indices
    }
    results = [completed_by_idx[idx] for idx in sorted(completed_by_idx)]

    if completed_by_idx:
        print(f"Resuming from saved results: {len(completed_by_idx)}/{len(eval_indices)} already complete")

    first_resumed_idx = None
    if completed_by_idx and first_resumed_retry_within_100_words:
        first_resumed_idx = next(
            (idx for idx in range(len(eval_indices)) if idx not in completed_by_idx),
            None,
        )
        if first_resumed_idx is not None:
            print(f"Using the 100-word retry prompt for first resumed item idx={first_resumed_idx}")

    jsonl_f = jsonl_path.open("a" if resume else "w") if jsonl_path else None
    if jsonl_path:
        print(f"Streaming per-sample results to {jsonl_path}")

    try:
        for idx, dataset_idx in tqdm(list(enumerate(eval_indices)), desc="Evaluating"):
            if idx in completed_by_idx:
                continue

            result = evaluate_one_item(
                model,
                dataset[dataset_idx],
                idx,
                dataset_idx,
                max_retries,
                retry_within_100_words=idx == first_resumed_idx,
                benchmark_name=benchmark_name,
            )
            results.append(result)

            if jsonl_f:
                jsonl_f.write(json.dumps(result) + "\n")
                jsonl_f.flush()

            write_summary(output_path, results, max_retries=max_retries, target_total=len(eval_indices))

            completed = len(results)
            if completed % 5 == 0 or completed == len(eval_indices):
                summary = summarize_results(results, max_retries=max_retries, target_total=len(eval_indices))
                retry_bits = ", ".join(
                    f"r{k}={summary['accuracy_by_retry'][str(k)]:.2%}"
                    for k in range(max_retries + 1)
                )
                print(f"[{completed}/{len(eval_indices)}] Running accuracy by retry: {retry_bits}")
    finally:
        if jsonl_f:
            jsonl_f.close()

    summary = summarize_results(results, max_retries=max_retries, target_total=len(eval_indices))
    final_accuracy = summary["accuracy_by_retry"][str(max_retries)]
    print(f"\nFinal accuracy with {max_retries} retries: {summary['correct']}/{summary['completed']} = {final_accuracy:.2%}")
    print("Accuracy by retry:")
    for retry_level in range(max_retries + 1):
        key = str(retry_level)
        print(
            f"  retry {retry_level}: "
            f"{summary['correct_by_retry'][key]}/{summary['completed']} = "
            f"{summary['accuracy_by_retry'][key]:.2%}"
        )

    if output_path:
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Final results saved to {output_path}")

    return final_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3 model on AIME 2025")
    parser.add_argument("--model", choices=["qwen3-8b", "qwen3-32b"], required=True)
    parser.add_argument("--model-path", required=True, help="Path to the downloaded model")
    parser.add_argument("--dataset-name", default="opencompass/AIME2025")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=None, help="Subset size")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--benchmark-name", default="AIME 2025")
    parser.add_argument(
        "--eval-start",
        type=int,
        default=None,
        help="Inclusive position within the shuffled shard",
    )
    parser.add_argument(
        "--eval-end",
        type=int,
        default=None,
        help="Exclusive position within the shuffled shard",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--first-resumed-retry-within-100-words",
        action="store_true",
        help="Add 'within 100 words' to retries for only the first incomplete resumed item",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model} from {args.model_path}")
    model = load_model(args.model, args.model_path)
    print("Model params:", model.describe_params())

    dataset_desc = args.dataset_name
    if args.dataset_config:
        dataset_desc += f" / {args.dataset_config}"
    print(f"Loading AIME 2025 ({dataset_desc}, split={args.split})...")
    dataset = load_aime2025(args.dataset_name, args.dataset_config, args.split)

    evaluate(
        model,
        dataset,
        num_samples=args.num_samples,
        output_file=args.output,
        max_retries=args.max_retries,
        shuffle=args.shuffle,
        seed=args.seed,
        resume=not args.no_resume,
        first_resumed_retry_within_100_words=args.first_resumed_retry_within_100_words,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        benchmark_name=args.benchmark_name,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
    )


if __name__ == "__main__":
    main()

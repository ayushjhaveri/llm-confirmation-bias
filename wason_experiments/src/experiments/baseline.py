# src/experiments/baseline.py
import dataclasses
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Tuple, Optional, Any
from .parse_utils import extract_visible
from datetime import datetime

@dataclass
class Turn:
    role: str
    content: str
    raw: Optional[str] = None

class BaselineExperiment:
    NAME = "baseline"

    def __init__(self, hidden_rule_name: str, hidden_rule_pred, seed_example: Tuple[int, int, int]):
        self.rule_name = hidden_rule_name
        self.rule_pred = hidden_rule_pred
        self.seed = seed_example
        self.max_pairs = 45
        self.supports_think = False
        self.debug_retries = False

    def _initial_prompt(self) -> str:
        a, b, c = self.seed
        return (
            "You will be given three numbers which conform to a simple rule that I have in mind.\n"
            "This rule is concerned with the common attributes or relations between any three numbers,\n"
            "and not with their absolute magnitude – for example, it is not a rule like \"all numbers above (or below) 50.\"\n\n"
            "Your goal is to discover this hidden rule by proposing sets of three numbers.\n"
            "You will then be told if the triple conforms with the hidden rule or not.\n\n"
            "At each turn, you may perform one of two actions:\n\n"
            "- Test: propose a new set of three integers, denoted [a,b,c]. You will then be told whether that triple conforms with the hidden rule (\"YES\" or \"NO\").\n"
            "- Announce: based on all previous tets, state what you believe the hidden rule is. You will not receive any feedback for announcements.\n\n"
            "Each instruction you receive will specify what to do.\n\n"
            "Format (must follow exactly):\n"
            "- If the instruction is Turn – Announce, output exactly one line:\n"
            "  Announce: <one short sentence naming the rule>\n\n"
            "- If the instruction is Turn – Test, output exactly one line:\n"
            "  Check: [a,b,c]\n\n"
            f"A triple that conforms with the hidden rule is: [{a}, {b}, {c}].\n"
            "Let's begin.\n"
            "Turn – Announce."
        )

    @staticmethod
    def _next_instruction(turn_idx: int) -> str:
        if turn_idx == 1:
            return "Turn - Test"
        return "Turn - " + ("Announce" if turn_idx % 2 == 0 else "Test")

    def _parse_action(self, model_text: str):
        strategy = "auto_think" if self.supports_think else "first"
        res = extract_visible(model_text, strategy=strategy)
        if res is None:
            return ("bad", model_text)
        kind, payload = res
        return (kind, payload)

    def _format_parsed(self, kind: str, payload: Any) -> str:
        if kind == "test":
            a, b, c = payload
            return f"Check: [{a},{b},{c}]"
        elif kind == "announce":
            return f"Announce: {payload}"
        else:
            return str(payload)

    def _summarize(self, s: str, max_chars: int = 800) -> str:
        s = s if isinstance(s, str) else str(s)
        if len(s) <= max_chars:
            return s
        return "...[TRUNCATED]...\n" + s[-max_chars:]

    def _print_try(self, tag: str, expected: str, prompt_repr: str, raw: str) -> None:
        if not self.debug_retries:
            return
        print("=" * 80, flush=True)
        print(f"{tag} (expected: {expected.upper()})", flush=True)
        print("- PROMPT >>>", flush=True)
        print(self._summarize(prompt_repr), flush=True)
        print("- REPLY  <<<", flush=True)
        print(self._summarize(raw, max_chars=800), flush=True)
        print("=" * 80, flush=True)

    def _retry_hint(self, expected: str) -> str:
        if expected == "announce":
            return (
                "Turn - Announce\n"
                "Remember: Please do not test a triple! This is an ANNOUNCE turn. Output EXACTLY ONE LINE:\n"
                "Announce: <one short sentence naming the rule>\n"
                "No extra text."
            )
        else:
            return (
                "Turn - Test\n"
                "Remember: Please do not announce the rule! This is a TEST turn. Output EXACTLY ONE LINE:\n"
                "Check: [a,b,c]\n"
                "where a,b,c MUST be INTEGERS ONLY. No Fractions allowed. Check: [-1,2,1/2] is invalid as no fractions allowed.\n"
                "- MUST end with ']' (closing bracket)\n"
                "No rules announce, no extra text, test with INTEGERS only"
            )

    def _messages_repr(self, msgs: List[dict]) -> str:
        parts = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _is_qwq(self, model) -> bool:
        name = getattr(model, "model_name", "")
        return isinstance(name, str) and ("qwq" in name.lower())

    def _generate_with_retry(self, model, messages: List[dict], expected: str, max_retries: int = 10):
        retry_trace = []

        def attempt(tag: str, msgs: List[dict], hint: Optional[str] = None, attempt_num: int = 1):
            msgs_try = msgs + ([{"role": "user", "content": hint}] if hint else [])
            raw = model.generate(msgs_try)
            kind, payload = self._parse_action(raw)

            retry_trace.append({
                "attempt": attempt_num,
                "tag": tag,
                "expected": expected,
                "retry_prompt": hint,
                "used_retry_hint": hint is not None,
                "raw": raw,
                "parsed_kind": kind,
                "parsed_payload": payload if isinstance(payload, (str, int, float, list, dict)) else str(payload),
            })

            self._print_try(tag, expected, self._messages_repr(msgs_try), raw)
            return kind, payload, raw

        kind, payload, raw = attempt("TRY #1", messages, attempt_num=1)
        if kind == expected:
            return kind, payload, raw, self._format_parsed(kind, payload), retry_trace

        hint_line = self._retry_hint(expected)
        for i in range(1, max_retries + 1):
            kind, payload, raw = attempt(
                f"RETRY #{i}",
                messages,
                hint=hint_line,
                attempt_num=i + 1,
            )
            if kind == expected:
                return kind, payload, raw, self._format_parsed(kind, payload), retry_trace

        raise ValueError(
            f"Expected a single-line {expected.upper()} after retries, but did not get it.\n"
            f"Last raw output was:\n{str(raw)}"
        )

    def run(self, model, on_turn=None) -> List[Turn]:
        turns: List[Turn] = []
        step = 0  # running step counter

        if hasattr(model, "supports_think"):
            self.supports_think = bool(model.supports_think)

        sp = self._initial_prompt()
        messages: List[dict] = [{"role": "user", "content": sp}]
        turns.append(Turn("environment", sp))
        step += 1
        if on_turn:
            on_turn({"step": step, "role": "environment", "content": sp})

        # Turn 1 – Announce
        kind1, payload1, raw1, parsed1, retry_trace = self._generate_with_retry(
            model, messages, expected="announce"
        )
        if on_turn and len(retry_trace) > 1:
            on_turn({
                "step": step + 1,
                "role": "retry_trace",
                "expected": "announce",
                "retries": retry_trace,
            })
        turns.append(Turn("model", parsed1, raw=raw1))
        messages.append({"role": "assistant", "content": parsed1})
        step += 1
        if on_turn:
            on_turn({"step": step, "role": "model", "content": parsed1, "raw": raw1})

        instr = self._next_instruction(1)
        turns.append(Turn("environment", instr))
        messages.append({"role": "user", "content": instr})
        step += 1
        if on_turn:
            on_turn({"step": step, "role": "environment", "content": instr})

        current_turn = 2
        for _ in range(self.max_pairs * 2):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, " : ", current_turn)

            expected = "test" if current_turn % 2 == 0 else "announce"
            kind, payload, raw, parsed, retry_trace = self._generate_with_retry(
                model, messages, expected=expected
            )

            if on_turn and len(retry_trace) > 1:
                on_turn({
                    "step": step + 1,
                    "role": "retry_trace",
                    "expected": expected,
                    "retries": retry_trace,
                })

            if kind == "test":
                a, b, c = payload
                turns.append(Turn("model", parsed, raw=raw))
                messages.append({"role": "assistant", "content": parsed})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "model", "content": parsed, "raw": raw})

                label = "YES" if self.rule_pred(a, b, c) else "NO"
                env_line = f"{label}\n{self._next_instruction(current_turn)}"
                turns.append(Turn("environment", env_line))
                messages.append({"role": "user", "content": env_line})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "environment", "content": env_line})

            elif kind == "announce":
                turns.append(Turn("model", parsed, raw=raw))
                messages.append({"role": "assistant", "content": parsed})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "model", "content": parsed, "raw": raw})

                env_line = self._next_instruction(current_turn)
                turns.append(Turn("environment", env_line))
                messages.append({"role": "user", "content": env_line})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "environment", "content": env_line})

            else:
                turns.append(Turn("model", str(raw), raw=raw))
                env_line = "Please follow the FORMAT."
                turns.append(Turn("environment", env_line))
                messages.append({"role": "user", "content": env_line})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "model", "content": str(raw), "raw": raw})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "environment", "content": env_line})

            current_turn += 1
            if current_turn > 1 + self.max_pairs * 2:
                break

        return turns
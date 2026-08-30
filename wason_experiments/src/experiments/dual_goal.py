import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from datetime import datetime

from .parse_utils import extract_visible


@dataclass
class Turn:
    role: str
    content: str
    raw: Optional[str] = None


class DualGoalExperiment:
    NAME = "dual-goal"

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
            "Triples that fit this rule are called DAX, and triples that do not fit this rule are called MED.\n\n"
            "These rules are concerned with the common attributes or relations between any three numbers,\n"
            "and not with their absolute magnitude - for example, they are not rules like \"all numbers above (or below) 50.\"\n\n"
            "Your goal is to discover both rules: the rule for DAX triples and the rule for MED triples.\n"
            "You will then be told if the triple conforms with the DAX rule or the MED rule.\n\n"
            "At each turn, you may perform one of two actions:\n\n"
            "- Test: propose a new set of three integers, denoted [a,b,c]. You will then be told whether that triple conforms with the DAX or MED rule (\"DAX\" or \"MED\").\n"
            "- Announce: based on all previous tests, state what you believe the two hidden rules are. You will not receive any feedback for announcements.\n\n"
            "Each instruction you receive will specify what to do.\n\n"
            "Format (must follow exactly):\n\n"
            "- If the instruction is Turn – Announce, output exactly two lines:\n"
            "  Announce: DAX rule - <one short sentence>\n"
            "  Announce: MED rule - <one short sentence>\n\n"
            "- If the instruction is Turn – Test, output exactly one line:\n"
            "  Check: [a,b,c]\n\n"
            f"A DAX triple is: [{a}, {b}, {c}].\n"
            "Let's begin.\n"
            "Turn - Announce."
        )

    @staticmethod
    def _next_instruction(turn_idx: int) -> str:
        if turn_idx == 1:
            return "Turn - Test"
        return "Turn - " + ("Announce" if turn_idx % 2 == 0 else "Test")

    _ANN_DAX_RE = re.compile(r"(?mi)^Announce:\s*DAX\s*rule\s*-\s*(.+)$")
    _ANN_MED_RE = re.compile(r"(?mi)^Announce:\s*MED\s*rule\s*-\s*(.+)$")
    _CHECK_BR = re.compile(r"(?mi)^Check:\s*\[([^\]]+)\]\s*$")
    _CHECK_NOBR = re.compile(r"(?mi)^Check:\s*([^\[\]\n]+?)\s*$")
    _CHECK_NEXT = re.compile(r"(?mi)^Check:\s*\n([^\[\]\n]+?)\s*$")

    @staticmethod
    def _norm_triple(inner: str) -> Optional[Tuple[int, int, int]]:
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 3:
            return None
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None

    def _parse_dual_announce(self, text: str) -> Optional[Tuple[str, str]]:
        m_dax = self._ANN_DAX_RE.search(text)
        m_med = self._ANN_MED_RE.search(text)
        if not (m_dax and m_med):
            return None
        dax = m_dax.group(1).strip().splitlines()[0]
        med = m_med.group(1).strip().splitlines()[0]
        return dax, med

    def _parse_test(self, text: str) -> Optional[Tuple[int, int, int]]:
        m = self._CHECK_BR.search(text)
        if m:
            tup = self._norm_triple(m.group(1).strip())
            if tup is not None:
                return tup
        m = self._CHECK_NOBR.search(text)
        if m:
            tup = self._norm_triple(m.group(1).strip())
            if tup is not None:
                return tup
        m = self._CHECK_NEXT.search(text)
        if m:
            tup = self._norm_triple(m.group(1).strip())
            if tup is not None:
                return tup
        return None

    def _parse_action(self, model_text: str):
        strategy = "auto_think" if self.supports_think else "first"
        vis = extract_visible(model_text, strategy=strategy)
        if vis is not None and vis[0] == "test":
            return ("test", vis[1])

        am = self._parse_dual_announce(model_text)
        if am is not None:
            return ("announce", am)

        tup = self._parse_test(model_text)
        if tup is not None:
            return ("test", tup)

        return ("bad", model_text)

    def _format_parsed(self, kind: str, payload: Any) -> str:
        if kind == "test":
            a, b, c = payload
            return f"Check: [{a},{b},{c}]"
        elif kind == "announce":
            dax, med = payload
            return (
                f"Announce: DAX rule - {dax}\n"
                f"Announce: MED rule - {med}"
            )
        else:
            return str(payload)

    def _summarize(self, s: str, max_chars: int = 800) -> str:
        s = s if isinstance(s, str) else str(s)
        if len(s) <= max_chars:
            return s
        return "...[TRUNCATED]...\n" + s[-max_chars:]

    def _messages_repr(self, msgs: List[dict]) -> str:
        parts = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

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
                "Remember: Output EXACTLY TWO LINES:\n"
                "Announce: DAX rule - <one short sentence>\n"
                "Announce: MED rule - <one short sentence>\n"
                "No extra text."
            )
        else:
            return (
                "Turn - Test\n"
                "Remember: Please do not announce the rules! This is a TEST turn. Output EXACTLY ONE LINE:\n"
                "Check: [a,b,c]\n"
                "where a,b,c MUST be INTEGERS ONLY.\n"
                "No rules announce, no extra text."
            )

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
            f"Expected a {expected.upper()} after retries, but did not get it.\n"
            f"Last raw output was:\n{str(raw)}"
        )

    def run(self, model, on_turn=None) -> List[Turn]:
        turns: List[Turn] = []
        step = 0

        if hasattr(model, "supports_think"):
            self.supports_think = bool(model.supports_think)

        sp = self._initial_prompt()
        messages: List[dict] = [{"role": "user", "content": sp}]
        turns.append(Turn("environment", sp))
        step += 1
        if on_turn:
            on_turn({"step": step, "role": "environment", "content": sp})

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

                if (a, b, c) == tuple(self.seed):
                    env_line = "Please test a different triple (not the given example)."
                    turns.append(Turn("environment", env_line))
                    messages.append({"role": "user", "content": env_line})
                    step += 1
                    if on_turn:
                        on_turn({"step": step, "role": "environment", "content": env_line})
                else:
                    label = "DAX" if self.rule_pred(a, b, c) else "MED"
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
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "model", "content": str(raw), "raw": raw})

                env_line = "Please follow the FORMAT."
                turns.append(Turn("environment", env_line))
                messages.append({"role": "user", "content": env_line})
                step += 1
                if on_turn:
                    on_turn({"step": step, "role": "environment", "content": env_line})

            current_turn += 1
            if current_turn > 1 + self.max_pairs * 2:
                break

        return turns
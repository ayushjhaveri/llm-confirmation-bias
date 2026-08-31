from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import hydra
import lm_api
import sys

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_thinking(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = text
    if re.search(r"(?i)<think>", t) and re.search(r"(?i)</think>", t):
        t = THINK_BLOCK_RE.sub("", t)
    elif re.search(r"(?i)</think>", t):
        t = re.sub(r"(?is)^.*?</think>\s*", "", t, count=1)
    t = re.sub(r"(?i)</think>\s*", "", t)
    t = re.sub(r"(?i)<think>\s*", "", t)
    t = re.sub(r"(?i)\s*/think\s*", " ", t)
    return t.strip()

# ------------------------
# Parsing / validation
# ------------------------

ANNOUNCE_LINE_RE = re.compile(r"(?mi)^\s*Announce:\s*(.+?)\s*$")
TEST_LINE_RE = re.compile(r"(?mi)^\s*Test:\s*(.+?)\s*$")

# Test payload must look like: [object 0, object 2] or []
TEST_LIST_RE = re.compile(r"^\[\s*(?:(object\s*\d+\.?)(?:\s*,\s*(object\s*\d+\.?))*)?\s*\]\s*\.?\s*$", re.IGNORECASE)
OBJ_RE = re.compile(r"(object\s*(\d+))\.?", re.IGNORECASE)

def _extract_single_line_payload(text: str, kind: str) -> Optional[str]:
    """
    Requires the assistant output (after stripping thinking) to be EXACTLY ONE LINE.
    Then extracts the payload part after "Announce:" or "Test:".
    """
    if not text:
        return None
    t = strip_thinking(text) or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    line = lines[0]
    if kind == "announce":
        m = ANNOUNCE_LINE_RE.match(line)
        return m.group(1).strip() if m else None
    else:
        m = TEST_LINE_RE.match(line)
        return m.group(1).strip() if m else None

def _parse_test_objects_line(line: str, num_objects: int) -> Optional[List[str]]:
    """
    Accepts full line like:
      Test: [object 0, object 2]
    Returns list of object names (deduped, in order), or None if invalid.
    """
    t = strip_thinking(line) or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    m = TEST_LINE_RE.match(lines[0])
    if not m:
        return None
    payload = m.group(1).strip()

    m2 = TEST_LIST_RE.match(payload)
    if not m2:
        return None

    found = []
    seen = set()
    for full, idx in OBJ_RE.findall(payload):
        i = int(idx)
        if not (0 <= i < num_objects):
            return None
        name = f"object {i}"
        if name not in seen:
            seen.add(name)
            found.append(name)
    return found

def _retry_hint(expect: str, num_objects: int) -> str:
    if expect == "announce":
        return (
            "Turn - Announce\n"
            "Remember: Please do not test objects! This is an ANNOUNCE turn. Output in the format:\n"
            "Announce: relevant=[object 0, object 1, object 2]; rule=<one short description of the rule>\n"
            f"Where each object index is an integer in [0, {num_objects-1}] and there can be any number and combination of relevant objects.\n"
        )
    else:
        return (
            "Turn - Test\n"
            f"This is a retry attempt because you announced instead of testing or tested with the wrong format! Remember: Please do NOT announce the rule! And please ensure the object indices are between object 0 and object {num_objects-1}! And do not say N, say object N! This is a TEST turn. Output in the format:\n"
            "Test: [object 0, object 1, object 2]\n"
            f"Where each object index is an integer in [0, {num_objects-1}] and any number and combination of objects can be tested.\n"
        )

@dataclass
class TurnResult:
    expect: str                   # "announce" or "test"
    output_line: str              # the 1-line assistant output (kept as-is)
    parsed_test_objects: Optional[List[str]] = None


class AlternatingBlicketOracleAgent:
    """
    Alternating Announce/Test agent with format retries.
    Retries are NOT committed to self.messages.
    """
    def __init__(
        self,
        horizon_tests: int,
        model: str,
        temperature: float,
        initial_msg_path: str,
        max_format_retries: int = 10,
        thinking_mode: bool = True,
    ):
        self.model = model
        self.temperature = float(temperature)
        self.max_format_retries = 10
        self.horizon_tests = int(horizon_tests)
        self.thinking_mode = bool(thinking_mode)

        # load initial prompt template (contains {{INITIAL_CONFIGURATION}})
        if initial_msg_path.startswith("/"):
            p = initial_msg_path
        else:
            p = os.path.join(hydra.utils.get_original_cwd(), initial_msg_path)
        with open(p, "r", encoding="utf-8") as f:
            self.initial_prompt_template = f.read()

        self._client = lm_api.get_client(model, thinking_mode=self.thinking_mode)
        self.messages: List[Dict[str, str]] = []
        self.total_cost = 0.0

        self.expect = "announce"  # start
        self.turn_idx = 0

    def init_episode(self, initial_configuration: str):
        content = self.initial_prompt_template.replace(
            "{{INITIAL_CONFIGURATION}}",
            (initial_configuration or "").strip(),
        )
        self.messages = [{"role": "user", "content": content}]
        self.expect = "announce"
        self.turn_idx = 0

    def next_turn(self):
        self.expect = "test" if self.expect == "announce" else "announce"
        self.turn_idx += 1

    class FormatRetryError(RuntimeError):
        pass


    def _eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)


    def _call_with_retries(
        self,
        base_messages: List[Dict[str, str]],
        expect: str,
        num_objects: int,
    ) -> Tuple[str, str, Any, float, bool, Optional[str]]:
        api_error = False
        last_response = ""
        last_usage = None
        last_cost = 0.0
        last_bad = None

        # Keep a full log of failures for stderr
        failures: List[str] = []

        for attempt in range(self.max_format_retries + 1):
            tmp = list(base_messages)
            if attempt > 0:
                tmp.append({"role": "user", "content": _retry_hint(expect, num_objects)})

            try:
                resp, cost = lm_api.query_llm(
                    self._client,
                    self.model,
                    messages=tmp,
                    chat_kwargs={"temperature": self.temperature},
                )
                last_cost = float(cost or 0.0)
                last_usage = getattr(resp, "usage", None)
                last_response = (resp.choices[0].message.content or "")
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:
                api_error = True
                msg = f"[attempt {attempt}/{self.max_format_retries}] API EXCEPTION ({expect}): {e}"
                failures.append(msg)
                self._eprint(msg)
                last_bad = msg
                continue

            stripped = strip_thinking(last_response) or ""
            lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
            first_line = lines[0] if lines else ""

            if expect == "announce":
                payload = _extract_single_line_payload(last_response, "announce")
                if payload is not None:
                    return first_line, last_response, last_usage, last_cost, api_error, last_bad
                else:
                    msg = (
                        f"[attempt {attempt}/{self.max_format_retries}] FORMAT FAIL (announce):\n"
                        f"--- stripped ---\n{stripped}\n"
                        f"--- raw ---\n{last_response}\n"
                    )
                    failures.append(msg)
                    self._eprint(msg)
                    last_bad = stripped
            else:
                parsed = _parse_test_objects_line(first_line, num_objects=num_objects)
                if parsed is not None:
                    return first_line, last_response, last_usage, last_cost, api_error, last_bad
                else:
                    msg = (
                        f"[attempt {attempt}/{self.max_format_retries}] FORMAT FAIL (test):\n"
                        f"Expected: Test: [object i, ...] with i in [0, {num_objects-1}]\n"
                        f"--- first_line ---\n{first_line}\n"
                        f"--- stripped ---\n{stripped}\n"
                        f"--- raw ---\n{last_response}\n"
                    )
                    failures.append(msg)
                    self._eprint(msg)
                    last_bad = stripped

        # Exhausted retries: raise (no fallback)
        raise FormatRetryError(
            f"Exceeded max_format_retries={self.max_format_retries} for expect={expect}. "
            f"See stderr for failed outputs."
        )

    # def _call_with_retries(self, base_messages: List[Dict[str, str]], expect: str, num_objects: int):
    #     api_error = False
    #     last_response = ""
    #     last_usage = None
    #     last_cost = 0.0
    #     last_bad = None

    #     for attempt in range(self.max_format_retries + 1):
    #         tmp = list(base_messages)
    #         if attempt > 0:
    #             # IMPORTANT: this retry hint is NOT committed to self.messages
    #             tmp.append({"role": "user", "content": _retry_hint(expect, num_objects)})

    #         try:
    #             resp, cost = lm_api.query_llm(
    #                 self._client,
    #                 self.model,
    #                 messages=tmp,
    #                 chat_kwargs={"temperature": self.temperature},
    #             )
    #             last_cost = float(cost or 0.0)
    #             last_usage = getattr(resp, "usage", None)
    #             last_response = (resp.choices[0].message.content or "")
    #         except (KeyboardInterrupt, EOFError):
    #             raise
    #         except Exception as e:
    #             api_error = True
    #             last_bad = f"EXCEPTION: {e}"
    #             continue

    #         stripped = strip_thinking(last_response) or ""
    #         lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    #         first_line = lines[0] if lines else ""

    #         if expect == "announce":
    #             # Must be exactly one line, starting with Announce:
    #             payload = _extract_single_line_payload(last_response, "announce")
    #             if payload is not None:
    #                 return first_line, last_response, last_usage, last_cost, api_error, last_bad
    #         else:
    #             parsed = _parse_test_objects_line(first_line, num_objects=num_objects)
    #             if parsed is not None:
    #                 return first_line, last_response, last_usage, last_cost, api_error, last_bad

    #         last_bad = stripped

    #     # fallback valid line
    #     if expect == "announce":
    #         return "Announce: relevant=[]; rule=unknown", "", last_usage, last_cost, True, last_bad
    #     else:
    #         return "Test: []", "", last_usage, last_cost, True, last_bad

    def act(
        self,
        num_objects: int,
        obs_after_test: Optional[str],
    ) -> Tuple[str, Dict]:
        """
        Returns:
          output_line: one-line "Announce: ..." or "Test: [...]"
          act_info: includes response_message/stripped etc.
        """
        # If last turn was a test, inject feedback + next instruction as a USER message
        if obs_after_test is not None:
            self.messages.append({"role": "user", "content": obs_after_test.strip()})

        # Turn instruction
        self.messages.append({"role": "user", "content": f"Turn - {self.expect.capitalize()}."})

        output_line, raw, usage, cost, api_error, last_bad = self._call_with_retries(
            base_messages=self.messages,
            expect=self.expect,
            num_objects=num_objects,
        )
        self.total_cost += float(cost or 0.0)

        # Commit ONLY the final assistant answer (NOT retry hints)
        self.messages.append({"role": "assistant", "content": raw if raw else output_line})

        parsed_test_objects = None
        if self.expect == "test":
            parsed_test_objects = _parse_test_objects_line(output_line, num_objects=num_objects) or []

        act_info = {
            "model": self.model,
            "expect": self.expect,
            "response_message": raw if raw else output_line,
            "response_message_stripped": strip_thinking(raw if raw else output_line),
            "usage": usage,
            "api_error": api_error,
            "last_bad_attempt": last_bad,
            "parsed_test_objects": parsed_test_objects,
        }
        return output_line, act_info

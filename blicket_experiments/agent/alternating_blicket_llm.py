from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import hydra

import lm_api
from agent.agents import Agent

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

ACTION_LINE_RE = re.compile(r"(?mi)^\s*Action:\s*(.+?)\s*$")
ANNOUNCE_LINE_RE = re.compile(r"(?mi)^\s*Announce:\s*(.+?)\s*$")

ACTION_CMD_RE = re.compile(r"^(put object (?P<i>\d+) on machine|take object (?P<j>\d+) off machine)$")

def _extract_single_line_payload(text: str, kind: str) -> Optional[str]:
    if not text:
        return None
    t = strip_thinking(text) or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    line = lines[0]
    if kind == "action":
        m = ACTION_LINE_RE.match(line)
        return m.group(1).strip().rstrip(".") if m else None
    else:
        m = ANNOUNCE_LINE_RE.match(line)
        return m.group(1).strip() if m else None

def _validate_action(cmd: str, num_objects: int) -> bool:
    if not cmd:
        return False
    m = ACTION_CMD_RE.match(cmd.strip())
    if not m:
        return False
    idx = int(m.group("i") or m.group("j"))
    return 0 <= idx < num_objects

def _retry_hint(expected: str, num_objects: int) -> str:
    if expected == "announce":
        return (
            "Turn - Announce\n"
            "Your previous response was INVALID.\n"
            "Output EXACTLY ONE LINE, no extra text:\n"
            "Announce: blickets=[object 0, object 1]; rule=<one short description of the rule>\n"
        )
    return (
        "Turn - Test\n"
        "Your previous response was INVALID.\n"
        "Output EXACTLY ONE LINE, no extra text:\n"
        "Action: put object N on machine\n"
        "OR\n"
        "Action: take object N off machine\n"
        f"Where N is an integer in [0, {num_objects-1}].\n"
    )

class AlternatingBlicketOracleAgent:
    def __init__(
        self,
        horizon_tests: int,
        model: str,
        temperature: float,
        initial_msg_path: str,
        max_format_retries: int = 3,
    ):
        self.model = model
        self.temperature = float(temperature)
        self.max_format_retries = int(max_format_retries)
        self.horizon_tests = int(horizon_tests)

        # load initial prompt template (your oracle prompt with {{INITIAL_CONFIGURATION}})
        if initial_msg_path.startswith("/"):
            p = initial_msg_path
        else:
            p = os.path.join(hydra.utils.get_original_cwd(), initial_msg_path)
        with open(p, "r", encoding="utf-8") as f:
            self.initial_prompt_template = f.read()

        self._client = lm_api.get_client(model)
        self.messages: List[Dict[str, str]] = []
        self.total_cost = 0.0

        self.expect = "announce"  # start
        self.turn_idx = 0

    def init_episode(self, initial_configuration_block: str):
        content = self.initial_prompt_template.replace(
            "{{INITIAL_CONFIGURATION}}",
            initial_configuration_block.strip(),
        )
        self.messages = [{"role": "user", "content": content}]
        self.expect = "announce"
        self.turn_idx = 0

    def next_turn(self):
        self.expect = "test" if self.expect == "announce" else "announce"
        self.turn_idx += 1

    def _retry_hint(self, expect: str) -> str:
        if expect == "announce":
            return (
                "Turn - Announce\n"
                "Your previous response was INVALID.\n"
                "Output EXACTLY ONE LINE, no extra text:\n"
                "Announce: blickets=[object 0, object 1]; rule=<one short description of the rule>\n"
                "There can be any number of blickets.\n"
            )
        return (
            "Turn - Test\n"
            "Your previous response was INVALID.\n"
            "Output EXACTLY ONE LINE, no extra text:\n"
            "Test: [object 0, object 2]\n"
            "You may include any number of objects.\n"
        )

    def _call_with_retries(self, base_messages: List[Dict[str, str]], expect: str, object_names: List[str]):
        api_error = False
        last_response = ""
        last_usage = None
        last_cost = 0.0
        last_bad = None

        for attempt in range(self.max_format_retries + 1):
            tmp = list(base_messages)
            if attempt > 0:
                tmp.append({"role": "user", "content": self._retry_hint(expect)})

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
                last_bad = f"EXCEPTION: {e}"
                continue

            stripped = strip_thinking(last_response) or ""
            lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
            first_line = lines[0] if lines else ""

            if expect == "announce":
                if parse_announce(first_line) is not None:
                    return first_line, last_response, last_usage, last_cost, api_error, last_bad
            else:
                parsed = parse_test_objects(first_line, object_names)
                if parsed is not None:
                    return first_line, last_response, last_usage, last_cost, api_error, last_bad

            last_bad = stripped

        # fallback valid line
        if expect == "announce":
            return "Announce: blickets=[]; rule=unknown", "", last_usage, last_cost, True, last_bad
        else:
            return "Test: []", "", last_usage, last_cost, True, last_bad

    def act(self, object_names: List[str]):
        # user instruction for whose turn it is
        self.messages.append({"role": "user", "content": f"Turn - {self.expect.capitalize()}."})

        output_line, raw, usage, cost, api_error, last_bad = self._call_with_retries(
            base_messages=self.messages,
            expect=self.expect,
            object_names=object_names,
        )
        self.total_cost += float(cost or 0.0)

        # commit ONLY final assistant answer
        self.messages.append({"role": "assistant", "content": raw if raw else output_line})

        parsed_test_objects = None
        if self.expect == "test":
            parsed_test_objects = parse_test_objects(output_line, object_names) or []

        act_info = {
            "model": self.model,
            "expect": self.expect,
            "response_message": raw if raw else output_line,
            "response_message_stripped": strip_thinking(raw if raw else output_line),
            "usage": usage,
            "api_error": api_error,
            "last_bad_attempt": last_bad,
        }
        return TurnResult(expect=self.expect, output_line=output_line, parsed_test_objects=parsed_test_objects), act_info

    def append_feedback(self, feedback_block: str):
        # feedback is a USER message (like before)
        self.messages.append({"role": "user", "content": feedback_block.strip()})


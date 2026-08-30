# src/judger.py
import re
from typing import Optional, Dict, Any

# --- Prompt rendering helpers -------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"{\s*(announced_rule|ground_truth_rule)\s*}", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"(?is)<think>.*?</think>")
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)

def _clean_rule_text(s: str) -> str:
    """For rule strings: drop any leaked think blocks and normalize whitespace."""
    if not s:
        return ""
    s = s.split("</think>")[-1]  # tolerate leaked closing tag
    s = _THINK_BLOCK.sub("", s)  # remove any full <think>...</think> spans
    return re.sub(r"\s+", " ", s).strip()

def _render_prompt(tmpl: str, announced: str, truth: str) -> str:
    """
    Render even if placeholders have spaces, and sanitize rule strings.
    Falls back to a safe default if tmpl is empty.
    """
    text = (tmpl or
            "Decide if the two rules are equivalent in meaning. "
            "Answer with ONLY 'YES' or 'NO'.\n\n"
            "Announced: {announced_rule}\nGround truth: {ground_truth_rule}")
    a = _clean_rule_text(announced)
    t = _clean_rule_text(truth)

    def repl(m: re.Match) -> str:
        return a if m.group(1).lower() == "announced_rule" else t

    return _PLACEHOLDER_RE.sub(repl, text).strip()

# --- Judge output parsing helpers --------------------------------------------

def _strip_to_after_think(text: str) -> str:
    """
    For model outputs: remove any <think>...</think> blocks and, if a </think> appears,
    keep only the content after the LAST closing tag.
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)  # remove any full <think>...</think> spans
    parts = _THINK_CLOSE.split(text)
    cleaned = parts[-1] if parts else text
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()

def _parse_yes_no(text: str) -> Optional[bool]:
    """
    Returns True for YES, False for NO, or None if not parsable.
    Matches YES/NO anywhere (after think-stripping), case-insensitive, word-boundary.
    """
    if not text:
        return None
    m = re.search(r"\b(YES|NO)\b", text.upper())
    if not m:
        return None
    return m.group(1) == "YES"

# --- Public API ---------------------------------------------------------------

def judge_rule_equivalence(
    llm,                         # e.g., Llama33_70B instance with .generate(prompt, gen_params)
    announced_rule: str,
    ground_truth_rule: str,
    prompt_template_text: str,   # contents of prompts/judge_equivalence*.txt
    gen_params: Any,             # GenParams (or your decoding params object)
    extra_strict_retry: bool = True,
) -> Dict:
    """
    Ask an LLM to decide if two short rule descriptions mean the same thing.
    Uses your model's standard .generate(prompt, gen_params) API.

    Returns a dict: {"ok": bool, "raw": str, "parsed": Optional[bool]}
        - ok: True if parsed to YES/NO
        - raw: raw assistant output (post-retry if used)
        - parsed: True for YES, False for NO, None if unparseable
    """
    user_prompt = _render_prompt(prompt_template_text, announced_rule, ground_truth_rule)

    # First attempt
    raw1 = llm.generate(user_prompt, gen_params)
    text1 = _strip_to_after_think(raw1)
    parsed1 = _parse_yes_no(text1)

    if parsed1 is not None:
        return {"ok": True, "raw": text1, "parsed": parsed1}

    # Retry with a stricter reminder (same decoding params)
    if extra_strict_retry:
        strict_prompt = user_prompt + "\n\nSTRICT FORMAT REMINDER: Output ONLY 'YES' or 'NO'."
        raw2 = llm.generate(strict_prompt, gen_params)
        text2 = _strip_to_after_think(raw2)
        parsed2 = _parse_yes_no(text2)
        return {"ok": parsed2 is not None, "raw": text2, "parsed": parsed2}

    return {"ok": False, "raw": text1, "parsed": None}

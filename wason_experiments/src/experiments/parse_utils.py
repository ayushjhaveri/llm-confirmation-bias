# src/experiments/parse_utils.py
import re
from typing import Literal, Tuple, Union, Optional

# Tolerate hidden <think> blocks and multi-line chatter
_THINK_CLOSE = re.compile(r"</\s*think\s*>", re.IGNORECASE)
_THINK_PAIR  = re.compile(r"(?is)<\s*think\b[^>]*>.*?</\s*think\s*>")

# Single-line anchors (multiline mode)
_ANNOUNCE_RE = re.compile(r"(?mi)^Announce:\s*(.+)$")

# Check formats:
#   Check: [a,b,c]
#   Check: a,b,c
#   Check:
#     a,b,c
_CHECK_BR    = re.compile(r"(?mi)^Check:\s*\[([^\]]+)\]\s*$")
_CHECK_NOBR  = re.compile(r"(?mi)^Check:\s*([^\[\]\n]+?)\s*$")
_CHECK_NEXT  = re.compile(r"(?mi)^Check:\s*\n([^\[\]\n]+?)\s*$")

_BLOCK_START = re.compile(r"(?mi)^(Announce:|Check:)\b")

def _normalize_check_inner(inner: str) -> Optional[Tuple[int, int, int]]:
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 3:
        return None
    try:
        a, b, c = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    return (a, b, c)

def _pick_block(region: str, mode: Literal["first", "last"]) -> Union[Tuple[str, Tuple[int, int, int]], Tuple[str, str], None]:
    """
    Scan a text region and return the earliest or last well-formed block:
      ('test', (a,b,c))  or  ('announce', 'text')
    If none found, return None.
    """
    candidates = []

    # Announce
    for m in _ANNOUNCE_RE.finditer(region):
        line = m.group(1).strip().splitlines()[0]
        candidates.append(("announce", m.start(), m.end(), line))

    # Check: [a,b,c]
    for m in _CHECK_BR.finditer(region):
        inner = m.group(1).strip()
        tup = _normalize_check_inner(inner)
        if tup is not None:
            candidates.append(("test", m.start(), m.end(), tup))

    # Check: a,b,c  (no brackets)
    for m in _CHECK_NOBR.finditer(region):
        inner = m.group(1).strip()
        tup = _normalize_check_inner(inner)
        if tup is not None:
            candidates.append(("test", m.start(), m.end(), tup))

    # Check:\n a,b,c  (numbers on next line)
    for m in _CHECK_NEXT.finditer(region):
        inner = m.group(1).strip()
        tup = _normalize_check_inner(inner)
        if tup is not None:
            candidates.append(("test", m.start(), m.end(), tup))

    if not candidates:
        return None

    pick = min(candidates, key=lambda x: x[1]) if mode == "first" else max(candidates, key=lambda x: x[1])
    kind, _, _, payload = pick
    if kind == "announce":
        return ("announce", payload)
    else:
        return ("test", payload)

def extract_first_visible(model_text: str) -> Union[Tuple[str, Tuple[int, int, int]], Tuple[str, str], None]:
    """
    If there's any </think>, keep ONLY the text after the LAST </think>,
    strip any embedded <think>...</think>, then return the EARLIEST valid block there.
    Else, scan the whole text and return the LAST valid block.
    Returns:
      ('test', (a,b,c))  or  ('announce','...')  or  None
    """
    if not model_text:
        return None
    s = str(model_text)

    if _THINK_CLOSE.search(s):
        post = s.split("</think>")[-1]
        post = _THINK_PAIR.sub("", post)
        return _pick_block(post, mode="first")
    else:
        return _pick_block(s, mode="last")

def extract_visible(
    model_text: str,
    strategy: Literal["auto_think", "first", "last"] = "auto_think",
) -> Union[Tuple[str, Tuple[int, int, int]], Tuple[str, str], None]:
    """
    Returns one block:
      ('test', (a,b,c))  or  ('announce','...')  or  None

    strategy:
      - "auto_think": if </think> exists -> scan AFTER last </think> and pick FIRST;
                      else pick LAST.
      - "first":      always pick the FIRST valid block.
      - "last":       always pick the LAST valid block.
    """
    if not model_text:
        return None
    s = str(model_text)

    if strategy == "first":
        return _pick_block(s, "first")
    if strategy == "last":
        return _pick_block(s, "last")

    # auto_think
    if _THINK_CLOSE.search(s):
        post = s.split("</think>")[-1]
        post = _THINK_PAIR.sub("", post)
        return _pick_block(post, "first")
    else:
        return _pick_block(s, "last")

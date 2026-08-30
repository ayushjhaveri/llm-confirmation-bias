# src/io_utils.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def save_transcript_txt(turns, path_txt: str):
    """
    Write a human-readable transcript. If a Turn has .raw, it is not shown here—
    only the parsed .content is displayed (what the environment actually “saw”).
    """
    ensure_dir(os.path.dirname(path_txt))
    lines = []
    for i, t in enumerate(turns, 1):
        role = getattr(t, "role", "unknown")
        content = getattr(t, "content", "")
        lines.append(f"[{i:03d}] {role.upper()}:\n{content}\n")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_transcript_jsonl(turns: List[Any], meta: Dict[str, Any], path_jsonl: str):
    """
    Write a machine-readable log. First line is {"meta": ...}.
    Each subsequent line is one event with role, parsed content, and (if present) raw.
    """
    ensure_dir(os.path.dirname(path_jsonl))
    with open(path_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        step = 0
        for t in turns:
            step += 1
            rec = {
                "step": step,
                "role": getattr(t, "role", None),
                "content": getattr(t, "content", None),
            }
            raw = getattr(t, "raw", None)
            if raw is not None:
                rec["raw"] = raw
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Optional append-style helpers (useful for “live” logging if you choose)
def append_jsonl(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_txt(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

def save_model_params(params: Dict[str, Any], path_json: str):
    ensure_dir(os.path.dirname(path_json))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

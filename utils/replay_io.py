import json
from typing import Dict, Iterable, Any, List

def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Write iterable of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file into a list of dicts."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

import os, re, json
from typing import Dict, List, Any, Tuple


# ==============================================================================
# 4. 데이터 로드 및 후처리 함수
# ==============================================================================

def load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def load_transcripts(path: str) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    out = {}
    for k, v in (data or {}).items():
        if isinstance(v, str): out[k] = {"text": v, "entities": []}
        elif isinstance(v, dict): out[k] = {"text": (v.get("text") or "").strip(), "entities": v.get("entities", []) or []}
        else: out[k] = {"text": "", "entities": []}
    return out

"중복 제거(무슨 중복 제거?) 유틸리티"
def dedup_clean(words: List[str]) -> List[str]:
    seen, out = set(), []
    for w in words or []:
        w = (w or "").strip()
        if not w or len(w) < 2: continue
        if w in seen: continue
        seen.add(w)
        out.append(w)
    return out


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

"""원본 문자열의 인덱스를 보존하기 위한 유틸리티"""
def build_norm_with_map(raw: str) -> Tuple[str, List[int]]:
    if not raw: return "", []
    raw_l = raw.lower()
    norm_chars, idx_map = [], []
    for i, ch in enumerate(raw_l):
        if re.match(r"[0-9a-z\u3131-\u318e\uac00-\ud7a3]", ch):
            norm_chars.append(ch)
            idx_map.append(i)
    return "".join(norm_chars), idx_map
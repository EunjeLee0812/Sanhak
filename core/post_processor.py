from typing import Tuple, List, Dict, Any, Optional
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
from config.settings import RULE_TOL, RULE_GATE, RULE_WRATIO_TH # 상수 import
from utils.data_loader import build_norm_with_map # 유틸 import

def best_substring_span_raw(entity: str, hyp_raw: str, normalizer, tolerance: int = 2) -> Tuple[Optional[int], Optional[int], float]:
    """
    [주의] normalizer 객체를 인자로 받도록 수정하여 의존성 주입
    """
    e = normalizer.normalize(entity)
    if not e: return None, None, 1.0
    h_norm, h_map = build_norm_with_map(hyp_raw)
    if not h_norm: return None, None, 1.0

    L = len(e)
    if len(h_norm) <= L:
        span_cer = Levenshtein.distance(e, h_norm) / L
        s_raw = h_map[0] if h_map else 0
        e_raw = (h_map[-1] + 1) if h_map else len(hyp_raw)
        return s_raw, e_raw, span_cer

    best = 1.0
    best_s_norm = 0
    best_e_norm = min(len(h_norm), L)
    for wlen in range(max(1, L - tolerance), min(len(h_norm), L + tolerance) + 1):
        for i in range(0, len(h_norm) - wlen + 1):
            sub = h_norm[i:i + wlen]
            span_cer = Levenshtein.distance(e, sub) / L
            if span_cer < best:
                best = span_cer
                best_s_norm = i
                best_e_norm = i + wlen
                if best == 0.0: break
        if best == 0.0: break
    
    s_raw = h_map[best_s_norm]
    e_raw_idx = best_e_norm - 1
    e_raw = (h_map[e_raw_idx] + 1) if e_raw_idx < len(h_map) else len(hyp_raw)
    return s_raw, e_raw, best

def postprocess_with_hotwords(hyp_raw: str, hotwords: List[str], normalizer, gate: float = RULE_GATE, tol: int = RULE_TOL, wratio_th: int = RULE_WRATIO_TH) -> Tuple[str, List[Dict[str, Any]]]:
    if not hyp_raw or not hotwords: return hyp_raw, []
    sorted_hotwords = sorted(hotwords, key=len, reverse=True)
    hyp_pp = hyp_raw
    replog = []
    
    for hw in sorted_hotwords:
        # normalizer 객체 전달
        s, e, cer = best_substring_span_raw(hw, hyp_pp, normalizer, tolerance=tol)
        
        if s is None or e is None or cer > gate: continue
        surface = hyp_pp[s:e]
        if normalizer.normalize(surface) == normalizer.normalize(hw): continue
        
        similarity_score = fuzz.WRatio(surface, hw)
        if similarity_score < wratio_th: continue
        
        hyp_pp = hyp_pp[:s] + hw + hyp_pp[e:]
        replog.append({"type": "replace", "from": surface, "to": hw, "span_cer": float(cer), "wratio": float(similarity_score)})
        
    return hyp_pp, replog
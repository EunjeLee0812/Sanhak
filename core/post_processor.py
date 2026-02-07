from typing import Tuple, List, Dict, Any, Optional
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
from config.settings import RULE_TOL, RULE_GATE, RULE_WRATIO_TH # 상수 import
from utils.normalizer import build_norm_with_map # 유틸 import

import math

def best_substring_span_raw(entity: str, hyp_raw: str, normalizer, tolerance: int = 2, min_ratio: float = 0.7, min_abs_len: int = 2) -> Tuple[Optional[int], Optional[int], float]:
    
    #수정: 정답과 가장 유사한 substring이 너무 짧은 경우 방지; min_ratio: entity 길이 대비 최소 비율, min_abs: 절대 최소 길이
    """
    [주의] normalizer 객체를 인자로 받도록 수정하여 의존성 주입, span : 단어
    """
    ent = normalizer.normalize(entity)
    if not ent: return None, None, 1.0
    hyp_norm, hyp_map = build_norm_with_map(hyp_raw)
    if not hyp_norm: return None, None, 1.0

    L = len(ent)
    if len(hyp_norm) <= L:
        span_cer = Levenshtein.distance(ent, hyp_norm) / L
        start_idx_raw = hyp_map[0] if hyp_map else 0
        end_idx_raw = (hyp_map[-1] + 1) if hyp_map else len(hyp_raw)
        return start_idx_raw, end_idx_raw, span_cer

    best = 1.0
    best_s_norm = 0
    best_e_norm = min(len(hyp_norm), L)
    for wlen in range(max(1, L - tolerance), min(len(hyp_norm), L + tolerance) + 1):
        for i in range(0, len(hyp_norm) - wlen + 1):
            sub = hyp_norm[i:i + wlen]
            span_cer = Levenshtein.distance(ent, sub) / L
            if span_cer < best:
                best = span_cer
                best_s_norm = i
                best_e_norm = i + wlen
                if best == 0.0: break
        if best == 0.0: break
    
    start_idx_raw = hyp_map[best_s_norm]
    end_idx_raw = best_e_norm - 1
    end_idx_raw = (hyp_map[end_idx_raw] + 1) if end_idx_raw < len(hyp_map) else len(hyp_raw)
    return start_idx_raw, end_idx_raw, best

def postprocess_with_hotwords(hyp_raw: str, hotwords: List[str], normalizer, gate: float = RULE_GATE, tol: int = RULE_TOL, wratio_th: int = RULE_WRATIO_TH, min_ratio: float = 0.7, min_abs_len: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
    
    #수정: 정답과 가장 유사한 substring이 너무 짧은 경우 방지; min_ratio:  hw 길이 대비 surface 최소 비율, min_abs: 절대 최소 길이
    
    if not hyp_raw or not hotwords: return hyp_raw, []
    sorted_hotwords = sorted(hotwords, key=len, reverse=True)
    hyp_pp = normalizer.normalize(hyp_raw)
    replog = []
    
    for hw in sorted_hotwords:
        # normalizer 객체 전달
        start_idx, end_idx, cer = best_substring_span_raw(hw, hyp_pp, normalizer, tolerance=tol)
        
        if start_idx is None or end_idx is None or cer > gate: continue
        surface = hyp_pp[start_idx:end_idx]
        
        # 수정: normalize 기준 통일(공백 제거)
        surf_n = normalizer.normalize(surface, remove_space=True)
        hw_n   = normalizer.normalize(hw, remove_space=True)

        # 추가: 조각 치환 방지(너무 짧으면 skip)
        if len(surf_n) < min_abs_len: continue
        if len(surf_n) < int(math.ceil(len(hw_n) * min_ratio)): continue

        if surf_n == hw_n: continue
        
        # 수정: 유사도도 normalize 기준으로 계산
        similarity_score = fuzz.WRatio(surf_n, hw_n)
        if similarity_score < wratio_th: continue
        
        hyp_pp = hyp_pp[:start_idx] + hw + hyp_pp[end_idx:]
        replog.append({"type": "replace", "from": surface, "to": hw, "span_cer": round(float(cer),4), "wratio": round(float(similarity_score),4)})
        
    return hyp_pp, replog
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

# 추가: 너무 짧은 substring 후보(wlen=1 등) 차단
    min_wlen = max(min_abs_len, L - tolerance, int(math.ceil(L * min_ratio)))
    max_wlen = min(len(hyp_norm), L + tolerance)

    # 최소로 봐야 하는 길이(min_wlen)가 최대로 볼 수 있는 길이(max_wlen)보다 커서 검색할 후보 자체가 0개인 상황
    if min_wlen > max_wlen:
        # fallback: 앞에서 L 길이만 비교 (에러 케이스에서 코드가 죽지 않도록하는 안전장치)
        # 의도: 이 함수는 실패했지만, ‘hyp 앞부분을 기준으로 봤을 때 entity와 굉장히 다르다’는 정보 반환
        sub = hyp_norm[:L]
        span_cer = Levenshtein.distance(ent, sub) / L
        start_idx_raw = hyp_map[0] if hyp_map else 0
        end_idx_raw = (hyp_map[min(L - 1, len(hyp_map) - 1)] + 1) if hyp_map else len(hyp_raw)
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
    hyp_final = normalizer.normalize(hyp_raw)
    replog = []
    
    for hw in sorted_hotwords:
        # normalizer 객체 전달
        start_idx, end_idx, cer = best_substring_span_raw(hw, hyp_final, normalizer, tolerance=tol)
        
        if start_idx is None or end_idx is None or cer > gate: continue
        surface = hyp_final[start_idx:end_idx]
        
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
        
        hyp_final = hyp_final[:start_idx] + hw + hyp_final[end_idx:]
        replog.append({"type": "replace", "from": surface, "to": hw, "span_cer": round(float(cer),4), "wratio": round(float(similarity_score),4)})
        
    return hyp_final, replog
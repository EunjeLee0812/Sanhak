from typing import List, Optional, Tuple
from rapidfuzz.distance import Levenshtein
from mecab import MeCab

from utils.normalizer import TextNormalizer
from config.settings import *

import math

_NORMALIZER = None
_MECAB = None

def _get_normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = TextNormalizer()
    return _NORMALIZER

def _get_mecab():
    global _MECAB
    if _MECAB is None:
        _MECAB = MeCab()
    return _MECAB


# CER, WER 계산 함수
def calculate_cer(ref: str, hyp: str, normalizer) -> Tuple[float, float, int]:
    r = normalizer.normalize(ref, remove_space=True)
    h = normalizer.normalize(hyp, remove_space=True)
    # if not r:
    #     return (0.0 if not h else 1.0), (0.0 if not h else float(len(h))), 0
    return Levenshtein.distance(r, h) / len(r), Levenshtein.distance(r,h), len(r)

def calculate_wer(ref: str, hyp: str,  normalizer=None, mecab=None) -> Tuple[float, float, int,List,List]:
    normalizer = normalizer or _get_normalizer()
    mecab = mecab or _get_mecab()
    r_text = normalizer.normalize(ref, remove_space=False)
    h_text = normalizer.normalize(hyp, remove_space=False)
    r_morphs, h_morphs = mecab.morphs(r_text), mecab.morphs(h_text)
    # if not r_morphs:
    #     return 0.0 if not h_morphs else 1.0
    return Levenshtein.distance(r_morphs, h_morphs) / len(r_morphs), Levenshtein.distance(r_morphs, h_morphs), len(r_morphs), r_morphs, h_morphs

#인식된 고유명사를 정답 고유명사와 비교해 인식 결과를 교정하는 데 도움을 주는 함수
#수정: 고유명사 매칭에서 1글자(너무 짧은 조각)는 후보에서 제외하기
def best_proper_noun_match(entity: str, hyp: str, normalizer, tol: int = RULE_TOL, min_ratio: float = 0.7, min_abs_len: int = 2) -> Tuple[float, str]:
    # min ratio: entity 길이 대비 최소 일치 비율 (조각 매칭 방지), min_abs_len: 절대 최소 substring 길이
    
    normalizer = normalizer or _get_normalizer()
    e = normalizer.normalize(entity, remove_space=True)
    h = normalizer.normalize(hyp, remove_space=True)
    if not e: return 0.0, ""
    if not h: return 1.0, ""

    L = len(e)
    if len(h) <= L: return Levenshtein.distance(e, h) / L, h

    # 수정: 너무 짧은 substring(예: '웅', '왓')이 매칭 후보로 들어오는 것을 차단
    min_wlen = max(min_abs_len, L - tol, int(math.ceil(L * min_ratio)))
    max_wlen = min(len(h), L + tol)

    # 추가: 비교 가능한 substring 후보가 아예 없는 경우 (hyp가 너무 짧은 경우), 에러 방지를 위한 fallback
    if min_wlen > max_wlen:
        sub = h[:L] # hyp 앞부분만 사용 (길이가 짧으면 있는 만큼)
        return Levenshtein.distance(e, sub) / L, sub


    best_score, best_sub = 1.0, ""
    # 수정된 범위: min_wlen 이상만 탐색 for 조각 매칭 방지
    for wlen in range(min_wlen, max_wlen + 1):
        for i in range(0, len(h) - wlen + 1):
            sub = h[i:i + wlen]
            score = Levenshtein.distance(e, sub) / L
            if score < best_score:
                best_score, best_sub = score, sub
                if best_score == 0.0: return 0.0, best_sub
    return best_score, best_sub

#인식된 고유명사의 recall, avg_pn_cer, matched_texts(교정된 인식 고유명사) 값 반환
def evaluate_proper_nouns(
    entities: List[str],
    hyp_final: str,
    normalizer,
    match_th: float = PN_MATCH_TH,
    hard_th: float = HARD_MISS_TH
) -> Tuple[Optional[float], Optional[float], List[str], List[str]]: #수정: 엔티티가 없는 경우 pn_recall = 0이 됨 -> None 반환해서 집계에서 제외
    """
    return: (pn_recall, avg_pn_cer, hyp_pn(로그용), hard_missed(학습용))
    """
    if not entities:
        return None, None, [], []

    cers: List[float] = []
    hyp_pn: List[str] = []
    hard_missed: List[str] = []

    for entity in entities:
        # ✅ normalizer 반드시 전달
        cer, matched_sub = best_proper_noun_match(entity, hyp_final, normalizer)

        cers.append(cer)
        
        # 수정: 고유명사 결과 정리: 1글자 조각 제거
        cleaned = (matched_sub or "").replace(" ", "").strip()
        hyp_pn.append(cleaned if len(cleaned) >= 2 else "")

        # ✅ 학습은 hard miss만
        if cer > hard_th:
            hard_missed.append(entity)

    pn_recall = sum(c <= match_th for c in cers) / len(cers)
    avg_pn_cer = sum(cers) / len(cers)

    return pn_recall, avg_pn_cer, hyp_pn, hard_missed
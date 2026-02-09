from typing import List, Optional, Tuple
from rapidfuzz.distance import Levenshtein
from mecab import MeCab

from utils.normalizer import TextNormalizer
from config.settings import *

import math, string

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

def calculate_wer(ref: str, hyp: str,  normalizer=None, mecab=None, ref_ents:List[str]=[],hyp_ents:List[str]=[])-> Tuple[float, float, int,List,List]:
    normalizer = normalizer or _get_normalizer()
    mecab = mecab or _get_mecab()

    ref_text = normalizer.normalize(ref, remove_space=False)
    hyp_text = normalizer.normalize(hyp, remove_space=False)

    # 2. 고유명사 보호 (Masking)
    # ref_ents, hyp_ents 리스트에 있는 단어들을 Mecab이 쪼개지 못하도록 치환
    # 예: "제이티비씨" -> "UNKTOKEN0"
    protected_map_ref_ents = {}
    if ref_ents:
        # 긴 단어부터 치환해야 함
        sorted_ents = sorted(ref_ents, key=len, reverse=True)
        for i, ref_ent in enumerate(sorted_ents):
            ent_norm = normalizer.normalize(ref_ent, remove_space=True)
            token = f"PROT{string.ascii_uppercase[i]}"
            protected_map_ref_ents[token] = ent_norm
            
            # 텍스트에서 치환
            ref_text = ref_text.replace(ent_norm, token)

    protected_map_hyp_ents = {}
    if hyp_ents:
        # 긴 단어부터 치환해야 함
        sorted_hyp_ents = sorted(hyp_ents, key=len, reverse=True)
        for i, hyp_ent in enumerate(sorted_hyp_ents):
            hyp_ent_norm = normalizer.normalize(hyp_ent, remove_space=True)
            token = f"PROT{string.ascii_uppercase[i]}"
            protected_map_hyp_ents[token] = hyp_ent_norm
            
            # 텍스트에서 치환
            hyp_text = hyp_text.replace(hyp_ent_norm, token)
            
    ref_morphs, hyp_morphs = mecab.morphs(ref_text), mecab.morphs(hyp_text)
    
    print(protected_map_hyp_ents, hyp_morphs)
    
    # 3. 보호된 토큰 복원 (Unmasking)
    # ['SPECIALent0', '틀어줘'] -> ['제이티비씨', '틀어줘']
    final_ref_morphs = [protected_map_ref_ents.get(m, m) for m in ref_morphs]
    final_hyp_morphs = [protected_map_hyp_ents.get(m, m) for m in hyp_morphs]    
    # if not ref_morphs:
    #     return 0.0 if not hyp_morphs else 1.0
    return Levenshtein.distance(final_ref_morphs, final_hyp_morphs) / len(final_ref_morphs), Levenshtein.distance(final_ref_morphs, final_hyp_morphs), len(final_ref_morphs), final_ref_morphs, final_hyp_morphs

#인식된 고유명사를 정답 고유명사와 비교해 인식 결과를 교정하는 데 도움을 주는 함수
#수정: 고유명사 매칭에서 1글자(너무 짧은 조각)는 후보에서 제외하기
def best_proper_noun_match(ent: str, hyp: str, normalizer, tol: int = RULE_TOL, min_ratio: float = 0.7, min_abs_len: int = 2) -> Tuple[float, str]:
    # min ratio: ent 길이 대비 최소 일치 비율 (조각 매칭 방지), min_abs_len: 절대 최소 substring 길이
    
    normalizer = normalizer or _get_normalizer()
    e = normalizer.normalize(ent, remove_space=True)
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
    ents: List[str],
    hyp_final: str,
    normalizer,
    match_th: float = PN_MATCH_TH,
    hard_th: float = HARD_MISS_TH
) -> Tuple[Optional[float], float, int, Optional[float], int, int, float, List[str], List[str]]: #수정: 엔티티가 없는 경우 pn_recall = 0이 됨 -> None 반환해서 집계에서 제외
    """
    return: (pn_recall, avg_pn_cer, hyp_pn(로그용), hard_missed(학습용))
    """
    if not ents:
        return None, 0.0,0,None,0,0,0.0, [], []

    cers: List[float] = []
    hyp_pn: List[str] = []
    hard_missed: List[str] = []
    #전체 pn_ents 수
    #PN_WER 계산 변수
    total_ents_cnt=len(ents)
    total_wrong_hyp_ents_cnt=0
    #PN_CER 계산 변수
    total_wrong_pn_char_cnt=0
    total_pn_char_cnt=0

    for ent in ents:
        # ✅ normalizer 반드시 전달
        cer, matched_sub = best_proper_noun_match(ent, hyp_final, normalizer)

        #고유명사 CER 계산용 분자, 분모 계산
        total_pn_char_cnt+=len(ent)
        total_wrong_pn_char_cnt+=Levenshtein.distance(ent,matched_sub)*len(ent)

        #hyp_ent가 틀렸으면 틀린 단어 개수에 추가
        if cer>0.01: total_wrong_hyp_ents_cnt+=1
        
        cers.append(cer)
        
        # 수정: 고유명사 결과 정리: 1글자 조각 제거
        cleaned = (matched_sub or "").replace(" ", "").strip()
        hyp_pn.append(cleaned if len(cleaned) >= 2 else "")

        # ✅ 학습은 hard miss만
        if cer > hard_th:
            hard_missed.append(ent)

    pn_recall = sum(c <= match_th for c in cers) / len(cers)
    avg_pn_cer = sum(cers) / len(cers)
    pn_wer=total_wrong_hyp_ents_cnt/total_ents_cnt

    return pn_recall, total_wrong_pn_char_cnt, total_pn_char_cnt, avg_pn_cer, total_wrong_hyp_ents_cnt, total_ents_cnt, pn_wer, hyp_pn, hard_missed

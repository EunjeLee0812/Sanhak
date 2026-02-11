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

# ✅ 1. "이 정도까지 틀린 건 봐줄게(수집할게)" 기준 설정 함수
def get_correction_threshold(length: int) -> float:
    """
    글자 수 별로 '수집할 최대 오차율'을 반환합니다.
    이 값보다 CER이 작거나 같아야(살짝 틀려야) 수집합니다.
    """
    if length <= 2:
        return 1.00 #2글자 :   
    elif length == 3:
        return 0.8  # 3글자: 2글자 틀리는것까지 수집 (0.33 <= 0.35)
    elif length == 4:
        return 0.8  # 4글자: 3글자 틀림(0.25)까지 수집 (0.25 <= 0.26)
    elif length == 5:
        return 0.65  # 5글자: 3글자 틀리는것까지 수집
    else:
        return 0.5  # 6글자 이상: 절반까지 틀리는거 허용
    
# CER, WER 계산 함수
def calculate_cer(ref: str, hyp: str, normalizer) -> Tuple[float, float, int]:
    # 평가용 normalizer로 변경
    r = normalizer.normalize_for_eval(ref, remove_space=True)
    h = normalizer.normalize_for_eval(hyp, remove_space=True)
    # if not r:
    #     return (0.0 if not h else 1.0), (0.0 if not h else float(len(h))), 0
    return Levenshtein.distance(r, h) / len(r), Levenshtein.distance(r,h), len(r)

def calculate_wer(ref: str, hyp: str,  normalizer=None, mecab=None, ref_ents:List[str]=[],hyp_ents:List[str]=[])-> Tuple[float, float, int,List,List]:
    normalizer = normalizer or _get_normalizer()
    mecab = mecab or _get_mecab()

    # 평가용 normalizer로 변경
    ref_text = normalizer.normalize_for_eval(ref, remove_space=False)
    hyp_text = normalizer.normalize_for_eval(hyp, remove_space=False)

    # 2. 고유명사 보호 (Masking)
    # ref_ents, hyp_ents 리스트에 있는 단어들을 Mecab이 쪼개지 못하도록 치환
    # 예: "제이티비씨" -> "UNKTOKEN0"
    protected_map_ref_ents = {}
    if ref_ents:
        # 긴 단어부터 치환해야 함
        sorted_ents = sorted(ref_ents, key=len, reverse=True)
        for i, ref_ent in enumerate(sorted_ents):
            # 평가용 normalizer로 교체
            ent_norm = normalizer.normalize_for_eval(ref_ent, remove_space=True)
            #예외처리
            if ent_norm=="": continue

            token = f"PROT{string.ascii_uppercase[i]}"
            protected_map_ref_ents[token] = ent_norm
            
            # 텍스트에서 치환
            ref_text = ref_text.replace(ent_norm, token)

    protected_map_hyp_ents = {}
    if hyp_ents:
        # 긴 단어부터 치환해야 함
        sorted_hyp_ents = sorted(hyp_ents, key=len, reverse=True)
        for i, hyp_ent in enumerate(sorted_hyp_ents):
            # 평가용 normalizer로 교체
            hyp_ent_norm = normalizer.normalize_for_eval(hyp_ent, remove_space=True)

            if hyp_ent_norm=="": continue

            token = f"PROT{string.ascii_uppercase[i]}"
            protected_map_hyp_ents[token] = hyp_ent_norm
            
            # 텍스트에서 치환
            hyp_text = hyp_text.replace(hyp_ent_norm, token)
            
    ref_morphs, hyp_morphs = mecab.morphs(ref_text), mecab.morphs(hyp_text)
    
    # print(protected_map_hyp_ents, hyp_morphs)
    
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
 
    # 평가용 normalizer로 교체
    e = normalizer.normalize_for_eval(ent, remove_space=True)
    h = normalizer.normalize_for_eval(hyp, remove_space=True)

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



def evaluate_proper_nouns(
    ents: List[str],
    hyp_normalized: str,
    normalizer,
    match_th: float = PN_MATCH_TH,
    hotwords: List[str] = [] 
    # soft_th: float = SOFT_MISS_TH
) -> Tuple[Optional[float], float, int, Optional[float], float, int, float, List[str], List[str], List[str], str]: #수정: 엔티티가 없는 경우 pn_recall = 0이 됨 -> None 반환해서 집계에서 제외
    """
    고유명사(Entities) 인식 성능을 평가하는 함수.
    
    Returns:
        pn_recall (float): 고유명사 인식 재현율
        avg_pn_cer (float): 고유명사 평균 CER
        hyp_pn (List[str]): 인식된 고유명사 텍스트 (로그용)
        soft_missed (List[str]): 살짝 틀려서 교정이 필요한 단어들 (학습용)
    """

    if not ents:
        return None, 0.0,0,None,0.0,0,0.0, [], [], [], hyp_normalized

    cers: List[float] = []
    hyp_pn: List[str] = []
    soft_missed: List[str] = []
    hotwords_hit: List[str] = [] # 핫워드 적중 리스트
    
    hyp_final=hyp_normalized
    #전체 pn_ents 수
    #PN_WER 계산 변수
    ref_total_pn_ents_cnt=len(ents)
    hyp_total_wrong_pn_ents_cnt=0
    #PN_CER 계산 변수
    hyp_total_wrong_pn_char_cnt=0
    ref_total_pn_char_cnt=0
    
    # 비교 편의를 위해 핫워드 정규화 미리 수행 (공백 제거)
    hotwords_norm_set = set()
    if hotwords:
        for hw in hotwords:
            hotwords_norm_set.add(normalizer.normalize(hw, remove_space=True))

    for ent in ents:
        # ✅ normalizer 반드시 전달
        cer, matched_sub = best_proper_noun_match(ent, hyp_final, normalizer)

        #고유명사 CER 계산용 분자, 분모 계산
        ent_n = normalizer.normalize_for_eval(ent, remove_space=True)
        if not ent_n:
            continue

        sub_raw = (matched_sub or "").replace(" ", "").strip()
        sub_n = normalizer.normalize_for_eval(sub_raw, remove_space=True)

        ref_total_pn_char_cnt += len(ent_n)
        hyp_total_wrong_pn_char_cnt += Levenshtein.distance(ent_n, sub_n)




        #hyp_ent가 틀렸으면 틀린 단어 개수에 추가
        if cer>0.01: 
            hyp_total_wrong_pn_ents_cnt+=1
            # 2. 맞힌 경우 (Hit) - 여기서 교정 작업 수행!
        else:
            # [NEW] 텍스트 교정 로직
            # matched_sub(인식된 텍스트, 예: '애플티비 플러스')를 ent(정답, 예: '애플티비플러스')로 교체
            if matched_sub and matched_sub in hyp_final:
                # replace는 모든 등장 횟수를 바꾸므로 주의가 필요하지만, 
                # 고유명사는 문장 내 유일한 경우가 많으므로 현재 단계에선 유효함.
                hyp_final = hyp_final.replace(matched_sub, ent)

            # 핫워드 적중 체크 (맞힌 경우에만 체크하는 것이 논리적으로 맞음)
            ent_norm = normalizer.normalize(ent, remove_space=True)
            if ent_norm in hotwords_norm_set:
                hotwords_hit.append(ent)
        
        # 결과 정리 (1글자 짜리는 노이즈로 보고 빈값 처리)
        cers.append(cer)

        cleaned = (matched_sub or "").replace(" ", "").strip()
        cleaned = normalizer.normalize_for_eval(cleaned, remove_space=True)
        hyp_pn.append(cleaned if len(cleaned) >= 2 else "")

    
        # ✅ 2. 살짝 틀린 것들만 골라내기 (Soft Miss Logic)
        max_tolerable_cer = get_correction_threshold(len(ent))

        # 조건: "완벽하지 않고(0초과) AND 너무 망가지지 않은(THR이하) 것"
        if 0 < cer <= max_tolerable_cer:
            soft_missed.append(ent)

    # Recall 계산 (match_th 이내로 들어온 것들의 비율)
    if len(cers) > 0:
      pn_recall = sum(c <= match_th for c in cers) / len(cers)
      pn_cer = hyp_total_wrong_pn_char_cnt/ref_total_pn_char_cnt
    else:
      pn_recall = 0.0
      pn_cer = 0.0  
    pn_wer=hyp_total_wrong_pn_ents_cnt/ref_total_pn_ents_cnt
    
    return pn_recall, hyp_total_wrong_pn_char_cnt, ref_total_pn_char_cnt, pn_cer, hyp_total_wrong_pn_ents_cnt, ref_total_pn_ents_cnt, pn_wer, hyp_pn, soft_missed, hotwords_hit, hyp_final
        



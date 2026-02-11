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

def calculate_wer(ref: str, hyp: str,  normalizer=None, mecab=None)-> Tuple[float, float, int,List,List]:

    # 2. [핵심 수정] WER 계산 시에는 띄어쓰기 영향을 무시하기 위해 공백을 제거하고 형태소 분석 수행
    #    (한국어 ASR 챌린지 등에서도 '띄어쓰기'는 평가 항목에서 제외하는 경우가 많음)
    
    # 방법 A: normalizer의 remove_space=True 옵션 활용 (추천)
    r_for_wer = normalizer.normalize_for_eval(ref, remove_space=True)
    h_for_wer = normalizer.normalize_for_eval(hyp, remove_space=True)
    
    # 방법 B: 그냥 replace 사용 (normalizer 로직에 따라 선택)
    # r_for_wer = ref.replace(" ", "")
    # h_for_wer = hyp.replace(" ", "")

    # 3. 공백이 제거된 문자열을 MeCab으로 분절
    # 입력이 같으므로 MeCab은 무조건 같은 결과를 뱉음 -> WER 0.0 달성
    ref_morphs = mecab.morphs(r_for_wer)
    hyp_morphs = mecab.morphs(h_for_wer)

    # 4. Levenshtein 거리 계산 (기존 로직 동일)
    dist = Levenshtein.distance(ref_morphs, hyp_morphs)
    length = len(ref_morphs)
    
    wer = dist / length if length > 0 else 0.0

    return wer, dist, length, ref_morphs, hyp_morphs 


#인식된 고유명사를 정답 고유명사와 비교해 인식 결과를 교정하는 데 도움을 주는 함수
#수정: 고유명사 매칭에서 1글자(너무 짧은 조각)는 후보에서 제외하기
def best_proper_noun_match(ref_pn: str, hyp: str, normalizer, tol: int = RULE_TOL, min_ratio: float = 0.7, min_abs_len: int = 2) -> Tuple[float, str]:
    # min ratio: ref_pn 길이 대비 최소 일치 비율 (조각 매칭 방지), min_abs_len: 절대 최소 substring 길이
    
    normalizer = normalizer or _get_normalizer()
 
    e = normalizer.normalize_for_eval(ref_pn, remove_space=True)
    # 2. Hypothesis(예측값) 정규화 및 인덱스 매핑 생성 [핵심 로직]
    # h: 공백 제거된 문자열 ("쿠팡플레이")
    # h_indices: h의 각 문자가 hyp의 어디에 있었는지 기록 ([0, 1, 3, 4, 5])
    h = ""
    h_indices = []
    
    # normalizer.normalize_for_eval 로직을 따르되 인덱스를 추적하기 위해 수동 순회
    # (주의: normalizer 내부 로직이 복잡하다면 normalizer에 매핑 함수를 추가하는 게 정석이지만, 
    #  여기서는 '공백/특수문자 제거' 수준이라고 가정하고 구현합니다.)
    
    # 먼저 전체 정규화(대소문자 등)는 하되 공백은 살려둠
    temp_hyp = normalizer.normalize_for_eval(hyp, remove_space=False)
    
    for i, char in enumerate(temp_hyp):
        if char.strip(): # 공백이 아닌 경우만 h에 추가
            h += char
            h_indices.append(i)

    if not h: return 1.0, ""

    L = len(e)
    
    # Case 1: h가 e보다 짧거나 같으면 전체 비교
    if len(h) <= L: 
        # 전체가 매칭되면 원본 전체("쿠팡 플레이")를 반환해야 교체 가능
        return Levenshtein.distance(e, h) / L, temp_hyp

    min_wlen = max(min_abs_len, L - tol, int(math.ceil(L * min_ratio)))
    max_wlen = min(len(h), L + tol)

    if min_wlen > max_wlen:
        # Fallback: 앞부분만 잘라서 비교
        sub_norm = h[:L]
        
        # 원본 인덱스 복원
        start_idx = h_indices[0]
        end_idx = h_indices[L-1] + 1 # 슬라이싱을 위해 +1
        original_sub = temp_hyp[start_idx:end_idx]
        
        return Levenshtein.distance(e, sub_norm) / L, original_sub

    best_score = 1.0
    best_original_sub = "" # 공백이 포함된 원본 서브스트링

    # 3. 윈도우 슬라이딩 탐색
    for wlen in range(min_wlen, max_wlen + 1):
        for i in range(0, len(h) - wlen + 1):
            # 정규화된 공간에서의 서브스트링
            sub_norm = h[i : i + wlen]
            
            score = Levenshtein.distance(e, sub_norm) / L
            
            if score < best_score:
                best_score = score
                
                # [핵심] 정규화 인덱스 -> 원본 인덱스로 변환하여 원본 문자열 추출
                # 시작 문자의 원본 인덱스
                origin_start = h_indices[i]
                # 끝 문자의 원본 인덱스 (슬라이싱 끝점은 포함되지 않으므로 +1)
                origin_end = h_indices[i + wlen - 1] + 1
                
                # 원본 문자열에서 추출 ("쿠팡 플레이")
                best_original_sub = temp_hyp[origin_start : origin_end]
                
                if best_score == 0.0: 
                    return 0.0, best_original_sub
    
    return best_score, best_original_sub



def evaluate_proper_nouns(
    ref_pns: List[str],
    hyp_normalized: str,
    normalizer,
    match_th: float = PN_MATCH_TH,
    hotwords: List[str] = [] 
    # soft_th: float = SOFT_MISS_TH
) -> Tuple[Optional[float], float, int, Optional[float], float, int, float, List[str], List[str], List[str], str]: #수정: 엔티티가 없는 경우 pn_recall = 0이 됨 -> None 반환해서 집계에서 제외
    """
    고유명사(ref_pnities) 인식 성능을 평가하는 함수.
    
    Returns:
        pn_recall (float): 고유명사 인식 재현율
        avg_pn_cer (float): 고유명사 평균 CER
        hyp_pn (List[str]): 인식된 고유명사 텍스트 (로그용)
        soft_missed (List[str]): 살짝 틀려서 교정이 필요한 단어들 (학습용)
    """

    if not ref_pns:
        return None, 0.0,0,None,0.0,0,0.0, [], [], [], hyp_normalized

    cers: List[float] = []
    hyp_pn: List[str] = []
    soft_missed: List[str] = []
    hotwords_hit: List[str] = [] # 핫워드 적중 리스트
    
    hyp_final=hyp_normalized
    #전체 pn_ref_pn 수
    #PN_WER 계산 변수
    ref_total_pn_cnt=len(ref_pns)
    hyp_total_wrong_pn_cnt=0
    #PN_CER 계산 변수
    hyp_total_wrong_pn_char_cnt=0
    ref_total_pn_char_cnt=0
    
    # 비교 편의를 위해 핫워드 정규화 미리 수행 (공백 제거)
    hotwords_norm_set = set()
    if hotwords:
        for hw in hotwords:
            hotwords_norm_set.add(normalizer.normalize(hw, remove_space=True))

    for ref_pn in ref_pns:
        # ✅ normalizer 반드시 전달
        cer, matched_sub = best_proper_noun_match(ref_pn, hyp_final, normalizer)

        #고유명사 CER 계산용 분자, 분모 계산
        ref_pn_n = normalizer.normalize_for_eval(ref_pn, remove_space=True)
        if not ref_pn_n:
            continue

        sub_raw = (matched_sub or "").replace(" ", "").strip()
        sub_n = normalizer.normalize_for_eval(sub_raw, remove_space=True)

        ref_total_pn_char_cnt += len(ref_pn_n)
        hyp_total_wrong_pn_char_cnt += Levenshtein.distance(ref_pn_n, sub_n)




        #hyp_ref_pn가 틀렸으면 틀린 단어 개수에 추가
        if cer>0.01: 
            hyp_total_wrong_pn_cnt+=1
            # 2. 맞힌 경우 (Hit) - 여기서 교정 작업 수행!
        else:
            # [NEW] 텍스트 교정 로직
            # matched_sub(인식된 텍스트, 예: '애플티비 플러스')를 ref_pn(정답, 예: '애플티비플러스')로 교체
            if matched_sub and matched_sub in hyp_final:
                # replace는 모든 등장 횟수를 바꾸므로 주의가 필요하지만, 
                # 고유명사는 문장 내 유일한 경우가 많으므로 현재 단계에선 유효함.
                hyp_final = hyp_final.replace(matched_sub, ref_pn)

            # 핫워드 적중 체크 (맞힌 경우에만 체크하는 것이 논리적으로 맞음)
            if ref_pn in hotwords_norm_set:
                hotwords_hit.append(ref_pn)
        
        # 결과 정리 (1글자 짜리는 노이즈로 보고 빈값 처리)
        cers.append(cer)

        cleaned = (matched_sub or "").replace(" ", "").strip()
        cleaned = normalizer.normalize_for_eval(cleaned, remove_space=True)
        hyp_pn.append(cleaned if len(cleaned) >= 2 else "")

    
        # ✅ 2. 살짝 틀린 것들만 골라내기 (Soft Miss Logic)
        max_tolerable_cer = get_correction_threshold(len(ref_pn))

        # 조건: "완벽하지 않고(0초과) AND 너무 망가지지 않은(THR이하) 것"
        if 0 < cer <= max_tolerable_cer:
            soft_missed.append(ref_pn)

    # Recall 계산 (match_th 이내로 들어온 것들의 비율)
    if len(cers) > 0:
      pn_recall = sum(c <= match_th for c in cers) / len(cers)
      pn_cer = hyp_total_wrong_pn_char_cnt/ref_total_pn_char_cnt
    else:
      pn_recall = 0.0
      pn_cer = 0.0  
    pn_wer=hyp_total_wrong_pn_cnt/ref_total_pn_cnt
    
    return pn_recall, hyp_total_wrong_pn_char_cnt, ref_total_pn_char_cnt, pn_cer, hyp_total_wrong_pn_cnt, ref_total_pn_cnt, pn_wer, hyp_pn, soft_missed, hotwords_hit, hyp_final
        



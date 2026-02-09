"""
후처리 모듈 (post_processor.py)

이 모듈은 ASR 결과를 핫워드 리스트를 이용하여 자동으로 교정합니다.
주요 기능:
1. 유사도 기반 substring 탐색
2. 핫워드로 자동 교체
3. 너무 짧은 조각 매칭 방지 (오탐 제거)
4. 교체 로그 생성

핵심 개념:
- Fuzzy Matching: 정확히 일치하지 않아도 유사하면 매칭
- Span Detection: 원본 텍스트에서 특정 구간(span) 찾기
- Levenshtein Distance: 편집 거리 기반 유사도 측정
"""

from typing import Tuple, List, Dict, Any, Optional
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
import math

from config.settings import RULE_TOL, RULE_GATE, RULE_WRATIO_TH
from utils.normalizer import build_norm_with_map


# ==============================================================================
# Substring Span 탐색 함수
# ==============================================================================

def best_substring_span_raw(
    entity: str, 
    hyp_raw: str, 
    normalizer, 
    tolerance: int = 2, 
    min_ratio: float = 0.7, 
    min_abs_len: int = 2
) -> Tuple[Optional[int], Optional[int], float]:
    """
    원본 텍스트에서 entity와 가장 유사한 substring의 위치(span)를 찾는 함수
    
    후처리의 핵심 함수로, "원본 텍스트의 어느 부분을 교체해야 하는지"를 결정합니다.
    
    Args:
        entity (str): 찾고자 하는 정답 단어 (핫워드)
            예: "엠비씨"
            
        hyp_raw (str): ASR 인식 결과 (원본 형태 그대로)
            예: "음비씨 뉴스데스크 틀어줘"
            
        normalizer: TextNormalizer 객체 (의존성 주입)
            정규화를 위해 필요
            
        tolerance (int): 길이 허용 오차
            예: 2 → entity 길이 ±2 글자 범위에서 탐색
            기본값: 2 (RULE_TOL)
            
        min_ratio (float): entity 길이 대비 최소 비율
            예: 0.7 → substring 길이가 entity 길이의 70% 이상이어야 함
            기본값: 0.7
            용도: 너무 짧은 조각 매칭 방지
            
        min_abs_len (int): 절대 최소 substring 길이
            예: 2 → substring이 최소 2글자 이상이어야 함
            기본값: 2
            용도: "ㄱ", "1" 같은 1글자 조각 매칭 방지
            
    Returns:
        Tuple[Optional[int], Optional[int], float]:
            - start_idx_raw: 원본 텍스트에서 매칭된 부분의 시작 인덱스
            - end_idx_raw: 원본 텍스트에서 매칭된 부분의 종료 인덱스
            - span_cer: 매칭된 substring의 CER (Character Error Rate)
            
            None, None, 1.0 반환 시 → 매칭 실패
            
    동작 방식:
        1. entity와 hyp_raw를 정규화
        2. hyp_raw의 모든 가능한 substring 생성
        3. 각 substring과 entity의 편집 거리(CER) 계산
        4. CER이 가장 낮은 substring 선택
        5. 원본 텍스트에서의 위치(인덱스) 반환
        
    예시:
        entity = "엠비씨"
        hyp_raw = "음비씨 뉴스"
        
        탐색 과정:
        1. 정규화: ent="엠비씨", hyp_norm="음비씨뉴스"
        2. substring 후보:
           - "음비씨" (3글자): CER = 0.33 (1글자 다름)
           - "음비" (2글자): CER = 0.67
           - "비씨" (2글자): CER = 0.67
           - "비씨뉴" (3글자): CER = 0.67
           - ...
        3. 최적: "음비씨" (CER=0.33)
        4. 원본 위치: start=0, end=3
        5. 반환: (0, 3, 0.33)
        
    조각 매칭 방지 메커니즘:
        문제: "엠비씨"를 찾을 때 "씨"만 매칭되는 오류
        해결:
        - min_ratio: "씨"(1글자)는 "엠비씨"(3글자)의 33%로 70% 미만 → 제외
        - min_abs_len: "씨"는 1글자로 2글자 미만 → 제외
        
    fallback 처리:
        탐색 가능한 substring이 없는 경우:
        - hyp_raw의 앞부분만 비교
        - 에러로 코드가 죽지 않도록 안전장치
        
    Note:
        - build_norm_with_map()으로 원본 인덱스 매핑 유지
        - 정규화된 텍스트로 탐색하되, 원본 인덱스 반환
        - CER = Levenshtein distance / entity 길이
    """
    # 1. entity 정규화
    ent = normalizer.normalize(entity)
    if not ent: 
        return None, None, 1.0  # entity가 비어있으면 실패
    
    # 2. hyp_raw 정규화 (인덱스 맵 포함)
    hyp_norm, hyp_map = build_norm_with_map(hyp_raw)
    if not hyp_norm: 
        return None, None, 1.0  # hyp가 비어있으면 실패

    # 3. entity 길이
    L = len(ent)
    
    # 4. 특수 케이스: hyp가 entity보다 짧거나 같은 경우
    # 전체를 하나의 substring으로 비교
    if len(hyp_norm) <= L:
        span_cer = Levenshtein.distance(ent, hyp_norm) / L
        
        # 원본 인덱스 계산
        start_idx_raw = hyp_map[0] if hyp_map else 0
        end_idx_raw = (hyp_map[-1] + 1) if hyp_map else len(hyp_raw)
        
        return start_idx_raw, end_idx_raw, span_cer

    # 5. 탐색 범위 계산
    # 너무 짧은 substring 후보 차단
    
    # 최소 길이 결정:
    # - min_abs_len: 절대 최소 (예: 2글자)
    # - L - tolerance: entity 길이에서 허용 오차를 뺀 값
    # - L * min_ratio: entity 길이의 비율 (예: 70%)
    # 이 중 최댓값을 최소 길이로 사용
    min_wlen = max(min_abs_len, L - tolerance, int(math.ceil(L * min_ratio)))
    
    # 최대 길이 결정:
    # - hyp 길이와 (L + tolerance) 중 작은 값
    max_wlen = min(len(hyp_norm), L + tolerance)

    # 6. fallback 처리: 탐색 가능한 범위가 없는 경우
    # min_wlen > max_wlen이면 후보가 하나도 없음
    if min_wlen > max_wlen:
        # 안전장치: hyp 앞부분만 비교
        sub = hyp_norm[:L]
        span_cer = Levenshtein.distance(ent, sub) / L
        
        # 원본 인덱스
        start_idx_raw = hyp_map[0] if hyp_map else 0
        end_idx_raw = (hyp_map[min(L - 1, len(hyp_map) - 1)] + 1) if hyp_map else len(hyp_raw)
        
        return start_idx_raw, end_idx_raw, span_cer

    # 7. 모든 가능한 substring 탐색
    best = 1.0           # 최소 CER (낮을수록 좋음)
    best_s_norm = 0      # 최적 substring의 시작 인덱스 (정규화 기준)
    best_e_norm = min(len(hyp_norm), L)  # 최적 substring의 종료 인덱스
    
    # 7-1. 길이별로 탐색 (min_wlen ~ max_wlen)
    # 수정 전 코드에서는 max(1, L - tolerance)부터 시작했지만,
    # 이제는 min_wlen부터 시작하여 조각 매칭 방지
    for wlen in range(min_wlen, max_wlen + 1):
        # 7-2. 시작 위치별로 탐색
        for i in range(0, len(hyp_norm) - wlen + 1):
            # substring 추출
            sub = hyp_norm[i:i + wlen]
            
            # CER 계산
            span_cer = Levenshtein.distance(ent, sub) / L
            
            # 최적값 갱신
            if span_cer < best:
                best = span_cer
                best_s_norm = i
                best_e_norm = i + wlen
                
                # 완벽히 일치하면 조기 종료
                if best == 0.0: 
                    break
        
        # 완벽히 일치하는 걸 찾았으면 종료
        if best == 0.0: 
            break
    
    # 8. 정규화된 인덱스 → 원본 인덱스 변환
    # hyp_map을 사용하여 원본 텍스트의 실제 위치 계산
    
    # 시작 인덱스: 정규화된 시작 위치에 해당하는 원본 인덱스
    start_idx_raw = hyp_map[best_s_norm]
    
    # 종료 인덱스: 정규화된 종료 위치에 해당하는 원본 인덱스 + 1
    # (Python의 슬라이싱은 end가 exclusive이므로)
    end_idx_raw = best_e_norm - 1
    end_idx_raw = (hyp_map[end_idx_raw] + 1) if end_idx_raw < len(hyp_map) else len(hyp_raw)
    
    return start_idx_raw, end_idx_raw, best


# ==============================================================================
# 핫워드 기반 후처리 함수
# ==============================================================================

def postprocess_with_hotwords(
    hyp_raw: str, 
    hotwords: List[str], 
    normalizer, 
    gate: float = RULE_GATE, 
    tol: int = RULE_TOL, 
    wratio_th: int = RULE_WRATIO_TH, 
    min_ratio: float = 0.7, 
    min_abs_len: int = 2
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    핫워드 리스트를 이용하여 ASR 결과를 자동 교정하는 후처리 함수
    
    동작 방식:
    1. 각 핫워드에 대해 ASR 결과에서 가장 유사한 부분 탐색
    2. 유사도가 임계값 이상이면 해당 부분을 핫워드로 교체
    3. 너무 짧은 조각은 제외 (오탐 방지)
    4. 교체 이력을 로그로 기록
    
    Args:
        hyp_raw (str): ASR 인식 결과 (원본)
            예: "음비씨 뉴스데스크 틀어줘"
            
        hotwords (List[str]): 교정에 사용할 핫워드 리스트
            예: ["엠비씨", "뉴스데스크"]
            
        normalizer: TextNormalizer 객체 (의존성 주입)
            
        gate (float): CER 임계값 (0~1 범위)
            예: 0.34 → CER이 34% 이하인 경우만 교체
            기본값: RULE_GATE (설정값)
            
        tol (int): 길이 허용 오차
            예: 2 → 핫워드 길이 ±2 글자 범위에서 탐색
            기본값: RULE_TOL (설정값)
            
        wratio_th (int): WRatio 유사도 임계값 (0~100)
            예: 92 → WRatio가 92% 이상인 경우만 교체
            기본값: RULE_WRATIO_TH (설정값)
            
        min_ratio (float): 핫워드 길이 대비 최소 비율
            예: 0.7 → substring이 핫워드 길이의 70% 이상이어야 함
            용도: 조각 매칭 방지
            
        min_abs_len (int): 절대 최소 substring 길이
            예: 2 → 1글자 조각 매칭 방지
            
    Returns:
        Tuple[str, List[Dict[str, Any]]]:
            - hyp_final: 후처리된 최종 텍스트
            - replog: 교체 로그 리스트
            
            replog 형식:
            [
                {
                    "type": "replace",
                    "from": "음비씨",
                    "to": "엠비씨",
                    "span_cer": 0.3333,
                    "wratio": 95.24
                },
                ...
            ]
            
    동작 예시:
        hyp_raw = "음비씨 뉴스데스크 틀어줘"
        hotwords = ["엠비씨", "뉴스데스크"]
        
        처리 과정:
        1. "엠비씨" 처리:
           - "음비씨" 발견 (CER=0.33, WRatio=95)
           - 임계값 통과 → 교체
           - hyp_final = "엠비씨 뉴스데스크 틀어줘"
           
        2. "뉴스데스크" 처리:
           - "뉴스데스크" 발견 (CER=0.0, 완벽 일치)
           - 이미 일치하므로 교체 안 함
           
        결과:
        - hyp_final = "엠비씨 뉴스데스크 틀어줘"
        - replog = [{"from": "음비씨", "to": "엠비씨", ...}]
        
    교체 기준 (AND 조건):
        1. CER <= gate (편집 거리 기준)
        2. WRatio >= wratio_th (fuzzy matching 기준)
        3. 길이 >= min_abs_len (절대 최소 길이)
        4. 길이 >= hw 길이 * min_ratio (상대 최소 길이)
        5. 이미 일치하지 않음 (중복 교체 방지)
        
    조각 매칭 방지 메커니즘:
        문제: "엠비씨"를 찾을 때 "씨"만 매칭
        해결:
        1. min_abs_len=2: "씨"(1글자)는 제외
        2. min_ratio=0.7: "씨"는 "엠비씨"의 33%로 제외
        3. 양쪽 조건 모두 충족해야 교체
        
    긴 핫워드 우선 처리:
        hotwords를 길이 내림차순으로 정렬하여 처리.
        이유: 긴 단어를 먼저 교체해야 짧은 단어와 겹치는 오류 방지
        
        예:
        - hotwords = ["뉴스", "뉴스데스크"]
        - "뉴스"를 먼저 교체하면 "뉴스데스크"를 못 찾음
        - "뉴스데스크"를 먼저 교체하면 정상 작동
        
    Note:
        - 교체는 순차적으로 진행 (이전 교체가 다음 교체에 영향)
        - 원본 인덱스 기반으로 정확한 위치 교체
        - 교체 로그는 디버깅 및 분석에 활용
    """
    # 0. 예외 처리: 입력이 비어있으면 그대로 반환
    if not hyp_raw or not hotwords: 
        return hyp_raw, []
    
    # 1. 핫워드를 길이 내림차순으로 정렬
    # 긴 단어를 먼저 처리하여 짧은 단어와의 충돌 방지
    sorted_hotwords = sorted(hotwords, key=len, reverse=True)
    
    # 2. hyp_raw 정규화 (공백 유지)
    hyp_final = normalizer.normalize(hyp_raw)
    
    # 3. 교체 로그 초기화
    replog = []
    
    # 4. 각 핫워드에 대해 순차적으로 처리
    for hw in sorted_hotwords:
        # 4-1. 가장 유사한 substring 찾기
        # normalizer 객체를 인자로 전달 (의존성 주입)
        start_idx, end_idx, cer = best_substring_span_raw(
            hw, 
            hyp_final, 
            normalizer, 
            tolerance=tol
        )
        
        # 4-2. 매칭 실패 또는 CER이 임계값 초과
        # None: 매칭 실패
        # cer > gate: 너무 다름 (예: cer=0.5, gate=0.34)
        if start_idx is None or end_idx is None or cer > gate: 
            continue  # 다음 핫워드로
        
        # 4-3. 매칭된 surface 추출
        surface = hyp_final[start_idx:end_idx]
        
        # 4-4. surface와 hotword를 정규화하여 비교
        # remove_space=True: 공백 제거하여 비교
        surf_n = normalizer.normalize(surface, remove_space=True)
        hw_n = normalizer.normalize(hw, remove_space=True)

        # 4-5. 조각 매칭 방지: 너무 짧으면 skip
        # 조건 1: 절대 최소 길이
        if len(surf_n) < min_abs_len: 
            continue
        
        # 조건 2: 상대 최소 길이 (hw 길이의 비율)
        if len(surf_n) < int(math.ceil(len(hw_n) * min_ratio)): 
            continue

        # 4-6. 이미 일치하면 교체 불필요
        if surf_n == hw_n: 
            continue
        
        # 4-7. WRatio 유사도 계산
        # WRatio: 가중 비율 유사도 (0~100 점수)
        # Levenshtein보다 부분 문자열 매칭에 강함
        similarity_score = fuzz.WRatio(surf_n, hw_n)
        
        # WRatio가 임계값 미만이면 skip
        if similarity_score < wratio_th: 
            continue
        
        # 4-8. 모든 조건 통과 → 교체 실행
        # 원본 텍스트에서 [start_idx:end_idx] 부분을 hw로 교체
        hyp_final = hyp_final[:start_idx] + hw + hyp_final[end_idx:]
        
        # 4-9. 교체 로그 기록
        replog.append({
            "type": "replace",           # 동작 타입
            "from": surface,             # 원본 (교체 전)
            "to": hw,                    # 교체 후
            "span_cer": round(float(cer), 4),  # CER 값
            "wratio": round(float(similarity_score), 4)  # WRatio 값
        })
        
    # 5. 최종 결과 반환
    return hyp_final, replog
"""
데이터 로더 모듈 (data_loader.py)

이 모듈은 실험에 필요한 데이터를 JSON 파일에서 로드하고 가공합니다.
주요 기능:
1. JSON 파일 안전하게 읽기
2. 오디오 파일별 정답 전사(transcript) 데이터 로드
3. 고유명사(entities) 정보 추출
4. 중복 제거 유틸리티
"""

import os
import re
import json
from typing import Dict, List, Any, Tuple


# ===========================================================================
# JSON 파일 로드 함수
# ===========================================================================

def load_json(path: str) -> Dict[str, Any]:
    """
    JSON 파일을 안전하게 로드하는 함수
    
    Args:
        path (str): JSON 파일 경로
        
    Returns:
        Dict[str, Any]: JSON 파일의 내용을 딕셔너리로 반환
                        파일이 없거나 오류 시 빈 딕셔너리 반환
                        
    동작 방식:
        1. 경로가 비어있거나 파일이 없으면 빈 딕셔너리 반환
        2. 파일을 UTF-8 인코딩으로 읽어서 JSON 파싱
        3. 파싱 실패 시에도 안전하게 빈 딕셔너리 반환 (암묵적 예외 처리)
        
    예시:
        data = load_json("transcripts.json")
        # data = {"audio1.mp4": {"text": "볼륨 올려줘", ...}, ...}
        
    Note:
        - 명시적 예외 처리는 없지만 json.load() 실패 시 자동으로 빈 딕셔너리 반환됨
        - 프로덕션에서는 try-except로 명시적 처리 권장
    """
    # 경로가 비어있거나 파일이 존재하지 않으면 빈 딕셔너리 반환
    if not path or not os.path.isfile(path): 
        return {}
    
    # UTF-8 인코딩으로 파일 읽기 및 JSON 파싱
    with open(path, "r", encoding="utf-8") as f: 
        return json.load(f)


# ===========================================================================
# 전사(Transcript) 데이터 로드 함수
# ===========================================================================

def load_transcripts(path: str) -> Dict[str, Dict[str, Any]]:
    """
    오디오 파일별 정답 전사(transcript) 데이터를 로드하는 함수
    
    전사 데이터는 ASR 평가를 위한 정답(ground truth)입니다.
    각 오디오 파일에 대해 다음 정보를 포함합니다:
    - text: 정답 텍스트 (예: "엠비씨 뉴스데스크 틀어줘")
    - entities: 고유명사 리스트 (예: ["엠비씨", "뉴스데스크"])
    
    Args:
        path (str): transcripts.json 파일 경로
        
    Returns:
        Dict[str, Dict[str, Any]]: 
            파일명을 키로, 전사 정보를 값으로 하는 딕셔너리
            
            형식:
            {
                "audio1.mp4": {
                    "text": "볼륨 올려줘",
                    "entities": ["볼륨"]
                },
                "audio2.mp4": {
                    "text": "엠비씨 뉴스",
                    "entities": ["엠비씨"]
                },
                ...
            }
            
    입력 JSON 형식 (2가지 지원):
        1. 딕셔너리 형식:
           {
               "audio1.mp4": {
                   "text": "볼륨 올려줘",
                   "entities": ["볼륨"]
               }
           }
           
        2. 문자열 형식 (간단한 경우):
           {
               "audio1.mp4": "볼륨 올려줘"
           }
           → entities는 자동으로 빈 리스트로 설정됨
           
    동작 방식:
        1. JSON 파일 로드
        2. 각 항목의 형식 확인
        3. 문자열이면 {"text": 문자열, "entities": []} 형태로 변환
        4. 딕셔너리면 text와 entities 추출 (없으면 기본값 사용)
        5. 공백 제거 및 None 값 처리
        
    예시:
        transcripts = load_transcripts("transcripts.json")
        
        # 특정 파일의 정답 가져오기
        meta = transcripts.get("audio1.mp4", {"text": "", "entities": []})
        ref_text = meta["text"]
        ref_entities = meta["entities"]
        
    Note:
        - text가 None이거나 비어있으면 빈 문자열("")로 처리
        - entities가 None이거나 없으면 빈 리스트([])로 처리
        - 파일이 없거나 로드 실패 시 빈 딕셔너리 반환
    """
    # JSON 파일 로드
    data = load_json(path)
    
    # 결과 저장용 딕셔너리
    out = {}
    
    # JSON 데이터의 각 항목을 순회하며 표준 형식으로 변환
    for k, v in (data or {}).items():
        # k: 파일명 (예: "audio1.mp4")
        # v: 전사 데이터 (문자열 또는 딕셔너리)
        
        # 케이스 1: v가 문자열인 경우 (간단한 형식)
        # 예: {"audio1.mp4": "볼륨 올려줘"}
        if isinstance(v, str): 
            out[k] = {
                "text": v,           # 문자열을 text로 저장
                "entities": []       # entities는 빈 리스트
            }
            
        # 케이스 2: v가 딕셔너리인 경우 (표준 형식)
        # 예: {"audio1.mp4": {"text": "볼륨 올려줘", "entities": ["볼륨"]}}
        elif isinstance(v, dict): 
            out[k] = {
                # text 추출 (없거나 None이면 빈 문자열)
                "text": (v.get("text") or "").strip(),
                
                # entities 추출 (없거나 None이면 빈 리스트)
                "entities": v.get("entities", []) or []
            }
            
        # 케이스 3: 그 외의 경우 (예상치 못한 형식)
        # 안전하게 빈 값으로 초기화
        else: 
            out[k] = {
                "text": "",
                "entities": []
            }
    
    return out


# ===========================================================================
# 중복 제거 유틸리티 함수
# ===========================================================================

def dedup_clean(words: List[str]) -> List[str]:
    """
    단어 리스트에서 중복을 제거하고 정제하는 함수
    
    주요 동작:
    1. 빈 문자열, None, 공백만 있는 문자열 제거
    2. 2글자 미만 단어 제거 (노이즈 방지)
    3. 중복 단어 제거 (순서 유지)
    
    Args:
        words (List[str]): 정제할 단어 리스트
        
    Returns:
        List[str]: 중복이 제거되고 정제된 단어 리스트
        
    예시:
        words = ["엠비씨", "뉴스", "", "엠비씨", "ㄱ", None, "  ", "뉴스"]
        cleaned = dedup_clean(words)
        # 결과: ["엠비씨", "뉴스"]
        
        설명:
        - "": 빈 문자열 제거
        - "ㄱ": 2글자 미만 제거
        - None: None 값 제거
        - "  ": 공백만 있는 문자열 제거
        - 중복된 "엠비씨", "뉴스"는 첫 번째 것만 유지
        
    사용 사례:
        - 후처리 결과에서 중복 교체 방지
        - 학습 데이터에서 노이즈 제거
        - 핫워드 리스트 정제
        
    Note:
        - 순서는 원본 리스트의 첫 번째 출현 순서로 유지됨
        - 2글자 미만 필터링 이유: "ㄱ", "1" 같은 조각은 의미 없는 경우가 많음
        - None과 빈 문자열을 모두 처리하여 안전성 확보
    """
    # 중복 확인용 Set (O(1) 조회 속도)
    seen = set()
    
    # 결과 저장용 리스트 (순서 유지)
    out = []
    
    # words가 None이면 빈 리스트로 처리
    for w in words or []:
        # 1단계: 공백 제거 및 문자열 변환
        # w가 None이면 빈 문자열로 처리
        w = (w or "").strip()
        
        # 2단계: 유효성 검사
        # - 빈 문자열 제거
        # - 2글자 미만 제거 (예: "ㄱ", "1" 등)
        if not w or len(w) < 2: 
            continue
        
        # 3단계: 중복 확인
        # 이미 본 단어면 건너뛰기
        if w in seen: 
            continue
        
        # 4단계: 새로운 단어 추가
        seen.add(w)   # Set에 등록 (중복 방지)
        out.append(w) # 리스트에 추가 (순서 유지)
    
    return out
# ============================================
# [프로젝트: TV 음성비서 시스템 (Faster-Whisper 기반)]
#
# 작성일: 2026-01-23
# 설명: 이 코드는 LLM 없이 경량화된 ASR 모델과 규칙 기반 NLU를 사용하여
#       TV 제어(채널, 볼륨, 앱, 콘텐츠 재생)를 수행하는 파이프라인입니다.
#
# [주요 파이프라인 단계]
# 1. ASR (Automatic Speech Recognition): 사용자의 음성을 텍스트로 변환 (Whisper 활용)
#    - 특징: 'Hotwords' 기능을 통해 방송 프로그램명 인식률을 높임.
# 2. Post-processing (후처리): ASR이 잘못 인식한 단어를 방송 편성표(EPG) 기준으로 교정.
#    - 안전장치(Gate): 무조건 바꾸지 않고, 유사도가 일정 수준 이상일 때만 교정하여 부작용 방지.
# 3. NLU (Natural Language Understanding): 텍스트에서 '의도(Intent)'와 '정보(Slot)' 추출.
#    - 방식: 딥러닝이 아닌 키워드 매칭(Rule-based) 방식 사용 (속도 빠름, 디버깅 용이).
# 4. Evaluation (평가): 정답 데이터와 비교하여 정확도(CER, Intent Acc) 산출.
# ============================================

import os, re, json, glob, csv, random, glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from g2pk import G2p
from faster_whisper import WhisperModel
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process, fuzz


# -----------------------------
# 0) 경로/파라미터 설정 (실험 환경 구성)
# -----------------------------
# 팀원 가이드: 이 부분은 실험의 '입력 데이터'와 '조절 변수'를 정의합니다.

# 데이터 경로 설정 (Colab 환경 기준)
AUDIO_FOLDER      = os.environ.get("AUDIO_FOLDER", "../../voice_record")            # 평가할 음성 파일(.wav) 폴더
TRANSCRIPTS_PATH  = os.environ.get("TRANSCRIPTS_PATH", "../../transcripts.json") # 정답지 (Ground Truth)
BIAS_PATH         = os.environ.get("BIASING_LIST_PATH", "../../biasing_list.json") # ASR 힌트 단어 리스트
EPG_PATH          = os.environ.get("EPG_PATH", "../../epg.json")                 # 실시간 방송 편성표 DB
CATALOG_PATH      = os.environ.get("CATALOG_PATH", "../../catalog.json")         # VOD 영화/드라마 목록 DB

# 실험 변수 (Hyper-parameters)
TOPK_SWEEP = [0, 10, 20, 50, 100]             # Hotwords 개수 변화 실험 (0개일 때 vs 20개일 때 인식률 차이 비교)
POSTPROCESS_SWEEP = [0, 1]           # 후처리 적용 여부 (0: OFF, 1: ON)
WL_TOPN    = int(os.environ.get("WL_TOPN", "80"))    # 교정 시 검색할 후보 단어의 최대 개수

# [후처리 안전장치] 과교정(Over-correction) 방지 Gate 파라미터
# 설명: ASR 결과가 '무한도전'인데 DB에 '무한도전 레전드'가 있다고 무조건 바꾸면 안 됨.
#       두 단어 간의 유사도가 아래 임계값을 넘어야만 교정을 수행함.
RULE_WRATIO_TH = int(os.environ.get("RULE_WRATIO_TH", "92")) # 문자열 유사도(0~100)가 92점 이상이어야 함
RULE_GATE      = float(os.environ.get("RULE_GATE", "0.34"))  # 부분 오타율이 34% 이내여야 함 (너무 다르면 다른 단어로 판단)
RULE_TOL       = int(os.environ.get("RULE_TOL", "2"))        # 글자 수 차이 허용 범위

# ASR 모델 하드웨어 설정
ASR_MODEL   = os.environ.get("ASR_MODEL", "medium")
ASR_DEVICE  = os.environ.get("ASR_DEVICE", "cuda")     # GPU 사용 (없으면 cpu)
ASR_COMPUTE = os.environ.get("ASR_COMPUTE", "float16") 
ASR_LANG    = os.environ.get("ASR_LANG", "ko")         # 한국어 설정
ASR_BEAM    = int(os.environ.get("ASR_BEAM", "5"))     # 탐색 폭 (클수록 정확하나 느림)

# 결과 저장 경로
OUT_ROWS = "./asr_detail.csv"     # 상세 로그: 파일 하나하나의 인식 결과 및 점수
OUT_SUM  = "./asr_summary.csv"  # 요약 로그: 실험 설정별 평균 점수 (보고서용 표 제작에 사용)


# -----------------------------
# 1) 텍스트 정규화 및 평가 함수 (Metrics)
# -----------------------------
# 설명: 컴퓨터가 채점하기 좋게 텍스트를 다듬고, 점수를 매기는 함수들입니다.

def norm_ko(text: str) -> str:
    if not text: return ""
    
    # 1. 소문자화 및 기본 청소
    s = text.lower().strip()
    
    # 3. g2pk 투입 (일반 단어인 'cake'나 숫자를 한글 발음으로 변환)
    # 이제 s에는 'mbc' 대신 '엠비씨'가 들어있으므로 안전합니다.
    s = g2p(s)
    
    # 4. 특수문자 제거 및 공백 제거 (CER 측정용)
    s = re.sub(r"[^0-9\uac00-\ud7a3]", "", s)
    
    return s

def cer_norm(ref: str, hyp: str) -> float:
    """
    [평가 지표] CER (Character Error Rate, 문자 오류율) 계산
    - 의미: 정답(ref) 대비 예측(hyp)이 얼마나 틀렸는가? (0.0이면 완벽, 낮을수록 좋음)
    - Levenshtein 거리 알고리즘을 사용해 편집 거리 계산.
    """
    r = norm_ko(ref)
    h = norm_ko(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)

def wer_norm(ref: str, hyp: str) -> float:
    """
    [평가 지표] WER (Word Error Rate, 단어 오류율) 계산
    - 의미: 정답(ref) 대비 예측(hyp) 단어가 얼마나 틀렸는가?
    - 문장을 공백 기준으로 나누어 단어 단위의 Levenshtein 거리를 계산.
    """
    # 한글 전처리 (기존에 정의하신 norm_ko 함수 활용)
    r_text = norm_ko(ref)
    h_text = norm_ko(hyp)
    
    # 단어 단위로 분리 (리스트 생성)
    r_words = r_text.split()
    h_words = h_text.split()
    
    if not r_words:
        return 0.0 if not h_words else 1.0
    
    # Levenshtein.distance는 문자열뿐만 아니라 리스트(시퀀스) 비교도 지원합니다.
    # 단, 사용하시는 라이브러리 버전에 따라 리스트 직접 입력이 안 될 경우 
    # 아래와 같이 단어 단위 편집 거리를 계산해야 합니다.
    return Levenshtein.distance(r_words, h_words) / len(r_words)

# # 1. 알파벳 발음 매핑 (영어가 포함된 경우 대비)
# ALPHABET_TO_KO = {
#     "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지",
#     "h": "에이치", "i": "아이", "j": "제이", "k": "케이", "l": "엘", "m": "엠",
#     "n": "엔", "o": "오", "p": "피", "q": "큐", "r": "알", "s": "에스",
#     "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이", "z": "제트"
# }

# # 2. TV 도메인 전용 영어 단어 발음 매핑
# ENG_WORD_TO_KO = {
#     "netflix": "넷플릭스", "youtube": "유튜브", "disney": "디즈니",
#     "tving": "티빙", "apple": "애플", "plus": "플러스", "tv": "티비"
# }

# # 3. 한글 수사 통일 (하나 -> 일, 둘 -> 이 등)
# NUM_SENSE_MAP = {
#     "하나": "일", "한": "일", "둘": "이", "두": "이", 
#     "셋": "삼", "세": "삼", "넷": "사", "네": "사",
#     "여덟": "팔", "열": "십"
# }

# def num_to_ko(num_str: str) -> str:
#     """숫자를 한국어 읽기 방식(0~999)으로 변환"""
#     try:
#         n = int(num_str)
#         if n == 0: return "영"
#         units = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
#         tens = ["", "십", "이십", "삼십", "사십", "오십", "육십", "칠십", "팔십", "구십"]
#         hundreds = ["", "백", "이백", "삼백", "사백", "오백", "육백", "칠백", "팔백", "구백"]

#         if n >= 100:
#             h_val, rest = divmod(n, 100)
#             t_val, u_val = divmod(rest, 10)
#             return hundreds[h_val] + (tens[t_val] if t_val != 1 else "십") + units[u_val]
#         elif n >= 10:
#             t_val, u_val = divmod(n, 10)
#             return (tens[t_val] if t_val != 1 else "십") + units[u_val]
#         else:
#             return units[n]
#     except:
#         return num_str

# def norm_ko(s: str) -> str:
#     """
#     [강화된 정규화] 모든 숫자, 영어, 수사를 한글 발음으로 변환 후 공백 제거
#     """
#     if not s: return ""
    
#     # 소문자 변환 및 양끝 공백 제거
#     s = s.strip().lower()

#     # (1) 도메인 특화 영어 단어 변환 (netflix -> 넷플릭스)
#     for eng, ko in ENG_WORD_TO_KO.items():
#         s = s.replace(eng, ko)

#     # (2) 한글 수사 통일 (여덟 -> 팔)
#     for k, v in NUM_SENSE_MAP.items():
#         s = s.replace(k, v)

#     # (3) 숫자 -> 한글 발음 변환 (8 -> 팔)
#     s = re.sub(r'\d+', lambda m: num_to_ko(m.group()), s)

#     # (4) 남은 영어 알파벳 변환 (mbc -> 엠비씨)
#     temp_s = ""
#     for char in s:
#         temp_s += ALPHABET_TO_KO.get(char, char)
#     s = temp_s

#     # (5) 한글, 숫자 외 특수문자 제거 및 공백 제거
#     s = re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", s)
    
#     return s

def best_substring_cer(entity: str, hyp: str, tol: int = 2) -> float:
    """
    [평가 지표] 핵심어(Entity) 부분 인식률 평가
    - 목적: 문장 전체가 틀려도, 핵심 단어(예: '아이유')만 맞으면 성공으로 간주하기 위함.
    - 방식: 문장(hyp) 안에서 정답 단어(entity)와 가장 비슷한 부분 문자열을 찾아 오차율 계산.
    """
    e = norm_ko(entity)
    h = norm_ko(hyp)
    if not e: return 0.0
    if not h: return 1.0

    L = len(e)
    # 문장이 핵심어보다 짧으면 전체 비교
    if len(h) <= L:
        return Levenshtein.distance(e, h) / L

    # Sliding Window 방식으로 문장을 훑으며 가장 비슷한 구간 탐색
    best = 1.0
    for wlen in range(max(1, L - tol), min(len(h), L + tol) + 1):
        for i in range(0, len(h) - wlen + 1):
            sub = h[i:i + wlen]
            score = Levenshtein.distance(e, sub) / L
            if score < best:
                best = score
                if best == 0.0: return 0.0
    return best

def proper_metrics(entities: List[str], hyp: str, th: float = 0.2) -> Tuple[Optional[float], Optional[float]]:
    """
    [종합 평가] 고유명사(PN) 성능 측정
    - Recall(재현율): 정답 키워드 중 몇 개를 맞췄는가? (th=0.2, 즉 오타 20% 이내면 정답 인정)
    - PN CER: 키워드 부분만의 평균 오타율.
    """
    if not entities:
        return None, None
    cers = [best_substring_cer(e, hyp) for e in entities]
    recall = sum(1 for c in cers if c <= th) / len(cers)
    pn_cer = sum(cers) / len(cers)
    return recall, pn_cer


# -----------------------------
# 2) 데이터 로드 (Input)
# -----------------------------

def load_json(path: str) -> Dict[str, Any]:
    """JSON 파일 읽기 유틸리티"""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_transcripts(path: str) -> Dict[str, Dict[str, Any]]:
    """
    [정답지 로드] transcripts.json 파싱
    - 구조: {"파일명.wav": {"text": "정답문장", "entities": ["키워드"], "intent": "의도"}}
    """
    data = load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (data or {}).items():
        if isinstance(v, str):
            out[k] = {"text": v, "entities": []}
        elif isinstance(v, dict):
            out[k] = {
                "text": (v.get("text") or "").strip(),
                "entities": v.get("entities", []) or []
            }
        else:
            out[k] = {"text": "", "entities": []}


# -----------------------------
# 3) EPG/Catalog + Hotwords pool 구성
# -----------------------------
# 설명: ASR 모델에게 "이 단어들이 나올 확률이 높아"라고 힌트(Hotwords)를 주기 위한 데이터 준비 단계

def dedup_clean(words: List[str]) -> List[str]:
    """리스트 내 중복 제거 및 짧은 단어 필터링"""
    seen, out = set(), []
    for w in words or []:
        w = (w or "").strip()
        if not w or len(w) < 2: continue
        if w in seen: continue
        seen.add(w)
        out.append(w)
    return out

def epg_titles(epg: Dict[str, Any]) -> List[str]:
    """방송 편성표(EPG)에서 현재/오늘 방송 제목 추출"""
    titles = []
    for section in ("now", "today"):
        for item in epg.get(section, []) or []:
            if isinstance(item, dict) and item.get("title"):
                titles.append(item["title"])
            elif isinstance(item, str):
                titles.append(item)
    return dedup_clean(titles)

def catalog_titles(catalog: Dict[str, Any]) -> List[str]:
    """VOD 카탈로그에서 영화/드라마 제목 추출"""
    if not isinstance(catalog, dict) or not catalog: return []
    if isinstance(catalog.get("titles"), list):
        return dedup_clean([x for x in catalog["titles"] if isinstance(x, str)])
    out = []
    for _, v in catalog.items():
        if isinstance(v, list): out += [x for x in v if isinstance(x, str)]
    return dedup_clean(out)

def coerce_terms(v: Any) -> List[str]:
    """Biasing List(가중치 사전) 포맷을 단순 리스트로 변환"""
    if not v: return []
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return [x.strip() for x in v if x and len(x.strip()) >= 2]
    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
        items = []
        for d in v:
            term = (d.get("term") or "").strip()
            score = d.get("score", 0)
            if term and len(term) >= 2:
                items.append((term, score))
        items.sort(key=lambda t: t[1], reverse=True)
        return [t for t, _ in items]
    return []


# -----------------------------
# 4) 최소 NLU (규칙 기반 Intent/Slot 파서)
# -----------------------------
# 설명: 복잡한 딥러닝 모델 대신, 특정 단어가 포함되면 의도를 추론하는 'Rule-based' 방식.
# LLM 없이도 기본적인 TV 제어는 가능하다는 것을 보여주는 부분.

# INTENTS = {
#     "POWER_OFF": "전원 끄기", "POWER_ON": "전원 켜기",
#     "VOLUME_SET": "볼륨 설정", "VOLUME_UP": "볼륨 올리기", "VOLUME_DOWN": "볼륨 내리기",
#     "CHANNEL_TUNE": "채널 변경", "APP_OPEN": "앱 실행", "CONTENT_PLAY": "콘텐츠 재생",
#     "UNKNOWN": "미분류",
# }

# # 앱 이름 동의어 사전 (유저가 '유튭'이라 해도 '유튜브'로 인식하기 위함)
# APP_ALIASES = {
#     "유튜브": ["유튜브", "youtube", "you tube"],
#     "넷플릭스": ["넷플릭스", "netflix"],
#     "디즈니플러스": ["디즈니플러스", "disney plus", "disney+", "디즈니+"],
#     "티빙": ["티빙", "tving", "tv ing", "tv-ing"],
# }

# # 채널명 동의어 사전
# CHANNEL_ALIASES = {
#     "MBC":  ["엠비씨", "mbc", "엠비시"],
#     "SBS":  ["에스비에스", "sbs"],
#     "JTBC": ["제이티비씨", "jtbc"],
#     "KBS1": ["케이비에스1", "kbs1", "케이비에스 1", "kbs 1"],
#     "KBS2": ["케이비에스2", "kbs2", "케이비에스 2", "kbs 2"],
#     "TVN":  ["티비엔", "tvn", "티브이엔", "tv n"],
#     "YTN":  ["와이티엔", "ytn"],
# }

NUM_KO_0_10 = {
    "영": 0, "공": 0, "일": 1, "하나": 1, "한": 1, "이": 2, "둘": 2, "두": 2,
    "삼": 3, "셋": 3, "세": 3, "사": 4, "넷": 4, "네": 4, "오": 5, "다섯": 5,
    "육": 6, "여섯": 6, "칠": 7, "일곱": 7, "팔": 8, "여덟": 8, "구": 9, "아홉": 9, "십": 10,
}

# def channel_terms_for_hotwords() -> List[str]:
#     """Hotwords(힌트)에 채널명들도 추가하여 인식률 향상"""
#     terms: List[str] = []
#     for ch_id, keys in (CHANNEL_ALIASES or {}).items():
#         if ch_id and isinstance(ch_id, str):
#             terms.append(ch_id)
#         for k in keys or []:
#             if isinstance(k, str) and k.strip():
#                 terms.append(k.strip())
#     return dedup_clean(terms)


def _contains_any(s: str, keys: List[str]) -> bool:
    """문자열 s 안에 keys 중 하나라도 포함되어 있는지 확인"""
    sl = (s or "").lower()
    return any(k.lower() in sl for k in keys)

# def extract_app(text: str) -> Optional[str]:
#     """텍스트에서 앱 이름 추출 (동의어 처리 포함)"""
#     if not text: return None
#     for app, keys in APP_ALIASES.items():
#         if _contains_any(text, keys):
#             return app
#     return None

# def extract_channel(text: str) -> Optional[str]:
#     """텍스트에서 채널명 추출 (정규화 포함)"""
#     if not text: return None
#     tl = (text or "").lower()
#     tl = re.sub(r"\s+", " ", tl).strip()
#     tl_ns = tl.replace(" ", "")

#     for ch_id, keys in CHANNEL_ALIASES.items():
#         for k in keys:
#             k_l = k.lower()
#             # 띄어쓰기 있는 버전과 없는 버전 모두 검사
#             if k_l in tl or k_l.replace(" ", "") in tl_ns:
#                 return ch_id
#     return None

def extract_int_number_0_10(text: str) -> Optional[int]:
    """
    [볼륨 추출] 텍스트에서 0~10 사이의 숫자를 찾음
    - '볼륨 5로 해줘', '볼륨 다섯' 등을 모두 처리
    """
    if not text: 
        return None

    # 1. 아라비아 숫자 찾기 (정규식)
    m = re.search(r"(?<!\d)(\d{1,2})(?!\d)", text)
    if m:
        try:
            n = int(m.group(1))
            if 0 <= n <= 10:
                return n
        except:
            pass

    # 2. 한글 숫자 찾기 (매핑 테이블 활용)
    t = re.sub(r"\s+", "", text)
    for k, v in NUM_KO_0_10.items():
        if k in t:
            return v
    return None

# def nlu_parse(text: str, catalog_pool: List[str]) -> Dict[str, Any]:
#     """
#     [NLU 핵심 로직] 텍스트를 입력받아 Intent(의도)와 Slot(상세정보) 반환
#     - 우선순위: 전원 > 볼륨 > 채널 > 앱 > 콘텐츠 검색 순으로 검사
#     """
#     out = {"intent": "UNKNOWN", "slots": {}}
#     if not text: return out
#     tl = text.lower()

#     # 1) 전원 제어
#     if ("꺼" in tl or "끄" in tl) and ("tv" in tl or "티비" in tl or "전원" in tl):
#         out["intent"] = "POWER_OFF"
#         return out
#     if ("켜" in tl or "켜줘" in tl) and ("tv" in tl or "티비" in tl or "전원" in tl):
#         out["intent"] = "POWER_ON"
#         return out

#     # 2) 볼륨 제어
#     if "볼륨" in tl or "volume" in tl:
#         n = extract_int_number_0_10(text)
#         if n is not None:
#             out["intent"] = "VOLUME_SET"
#             out["slots"]["volume"] = int(n)
#             return out
#         if "올" in tl or "키" in tl or "높" in tl:
#             out["intent"] = "VOLUME_UP"
#             return out
#         if "내" in tl or "줄" in tl or "낮" in tl:
#             out["intent"] = "VOLUME_DOWN"
#             return out

#     # 3) 채널 제어
#     ch = extract_channel(text)
#     if ch:
#         # 채널명이 있다고 무조건 채널 변경이 아님 (예: "MBC에서 하는 드라마 찾아줘")
#         # 따라서 명령어가 같이 있어야 함.
#         if ("채널" in tl) or ("바꿔" in tl) or ("변경" in tl) or ("틀" in tl) or ("켜" in tl) or ("봐" in tl) or ("보여" in tl):
#             out["intent"] = "CHANNEL_TUNE"
#             out["slots"]["channel_id"] = ch
#             return out

#     # 4) 앱 실행
#     app = extract_app(text)
#     if app and ("켜" in tl or "열" in tl or "실행" in tl or "틀" in tl):
#         out["intent"] = "APP_OPEN"
#         out["slots"]["app"] = app
#         return out

#     # 5) 콘텐츠 검색 (Fuzzy Matching)
#     # ASR 결과가 정확하지 않아도 DB에 있는 제목과 가장 유사하면 검색 의도로 파악
#     if ("틀" in tl or "재생" in tl or "보여" in tl or "켜" in tl) and catalog_pool:
#         best = process.extractOne(text, catalog_pool, scorer=fuzz.WRatio)
#         if best:
#             title, score, _ = best
#             if score >= 75:  # 유사도가 75점 이상일 때만 인정
#                 out["intent"] = "CONTENT_PLAY"
#                 out["slots"]["title"] = title
#                 out["slots"]["title_score"] = float(score)
#                 return out

#     return out

# def intent_slot_metrics(ref_intent, ref_slots, hyp_intent, hyp_slots):
#     """NLU 결과 채점 (정답과 비교)"""
#     if not ref_intent: return None, None
#     intent_acc = 1 if (ref_intent == hyp_intent) else 0 # 의도 정확도

#     if not ref_slots or not isinstance(ref_slots, dict):
#         return intent_acc, None

#     keys = list(ref_slots.keys())
#     if not keys: return intent_acc, 1.0

#     # Slot 정확도 (예: 채널명은 맞췄는지, 볼륨 숫자는 맞췄는지)
#     ok = 0
#     for k in keys:
#         if k in hyp_slots and hyp_slots.get(k) == ref_slots.get(k):
#             ok += 1
#     return intent_acc, ok / len(keys)

# def hotwords_from_context(bias: Dict[str, Any], epg: Dict[str, Any], catalog: Dict[str, Any], top_k: int) -> List[str]:
#     """
#     [Context Injection] 현재 상황에 맞춰 ASR에게 힌트로 줄 단어장 생성
#     - 편성표 제목, VOD 제목, 채널명 등을 모아서 top_k개 만큼 추림.
#     """
#     if top_k <= 0: return []
#     pool = dedup_clean(
#         epg_titles(epg)
#         + coerce_terms(bias.get("global"))
#         + catalog_titles(catalog)
#         + channel_terms_for_hotwords()
#     )
#     return pool[:top_k]

def hotwords_from_context(bias: Dict[str, Any], epg: Dict[str, Any], catalog: Dict[str, Any], top_k: int) -> List[str]:
    """
    [Context Injection] 현재 상황에 맞춰 ASR에게 힌트로 줄 단어장 생성
    - 전체 리스트에서 중복을 제거한 후, 랜덤하게 top_k개를 선정하여 반환합니다.
    """
    if top_k <= 0: 
        return []
    
    # 1. 전체 후보 단어 풀 생성 (중복 제거 포함)
    pool = dedup_clean(
        epg_titles(epg)
        + coerce_terms(bias.get("global"))
        + catalog_titles(catalog)
    )

    # 2. 리스트보다 요청한 개수(top_k)가 많을 경우를 대비한 예외 처리
    actual_k = min(top_k, len(pool))

    # 3. 랜덤 샘플링 수행 (순서 섞임 효과 포함)
    return random.sample(pool, actual_k)


# -----------------------------
# 5~6) 후처리 (Whitelist Correction)
# -----------------------------
# 보고서 포인트: ASR의 한계를 극복하는 핵심 기술.
# ASR이 "무한 도잔 틀어줘"라고 했을 때 DB의 "무한도전"으로 고쳐주는 로직.

def make_whitelist(hyp_raw: str, epg: Dict[str, Any], catalog: Dict[str, Any], top_n: int = 80) -> List[str]:
    """
    현재 인식된 문장(hyp_raw)과 가장 비슷한 DB 내의 제목 후보군(Top N)을 뽑음.
    전체 DB를 다 뒤지면 느리므로, 1차적으로 가능성 있는 것만 추림.
    """
    pool = dedup_clean(epg_titles(epg) + catalog_titles(catalog))
    if not pool: return []
    # RapidFuzz를 사용하여 빠르게 유사도 검색
    scored = process.extract(hyp_raw, pool, scorer=fuzz.WRatio, limit=min(top_n, len(pool)))
    return [t for t, _, _ in scored]

def build_norm_with_map(raw: str) -> Tuple[str, List[int]]:
    """원본 문자열의 인덱스를 보존하기 위한 유틸리티 (후처리 교체 위치 계산용)"""
    if not raw: return "", []
    raw_l = raw.lower()
    norm_chars, idx_map = [], []
    for i, ch in enumerate(raw_l):
        if re.match(r"[0-9a-z\u3131-\u318e\uac00-\ud7a3]", ch):
            norm_chars.append(ch)
            idx_map.append(i) # 정규화된 문자가 원본의 몇 번째 문자인지 저장
    return "".join(norm_chars), idx_map

def best_substring_span_raw(entity: str, hyp_raw: str, tol: int = 2) -> Tuple[Optional[int], Optional[int], float]:
    """
    [교정 위치 탐색] '무한도잔 틀어줘'에서 '무한도잔'이 어디서부터 어디까지인지(Index) 찾음.
    - 정규화된 문자열에서 위치를 찾은 뒤, `build_norm_with_map`을 통해 원본 문자열 인덱스로 변환.
    """
    e = norm_ko(entity)
    if not e: return None, None, 1.0
    h_norm, h_map = build_norm_with_map(hyp_raw)
    if not h_norm: return None, None, 1.0

    L = len(e)
    # 문장이 너무 짧으면 전체 비교
    if len(h_norm) <= L:
        span_cer = Levenshtein.distance(e, h_norm) / L
        s_raw = h_map[0] if h_map else 0
        e_raw = (h_map[-1] + 1) if h_map else len(hyp_raw)
        return s_raw, e_raw, span_cer

    # 부분 문자열 탐색
    best = 1.0
    best_s_norm = 0
    best_e_norm = min(len(h_norm), L)

    for wlen in range(max(1, L - tol), min(len(h_norm), L + tol) + 1):
        for i in range(0, len(h_norm) - wlen + 1):
            sub = h_norm[i:i + wlen]
            span_cer = Levenshtein.distance(e, sub) / L
            if span_cer < best:
                best = span_cer
                best_s_norm = i
                best_e_norm = i + wlen
                if best == 0.0: break
        if best == 0.0: break

    # 찾은 위치를 원본 인덱스로 변환
    s_raw = h_map[best_s_norm]
    e_raw = h_map[best_e_norm - 1] + 1
    return s_raw, e_raw, best

def postprocess_rule_whitelist(
    hyp_raw: str,
    whitelist: List[str],
    wratio_th: int = 92,
    gate: float = 0.34,
    tol: int = 2
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    [후처리 메인 로직] Gate Mechanism 적용
    - 기능: ASR 결과(hyp_raw)를 화이트리스트에 있는 정확한 제목으로 치환.
    - Gate 1 (유사도): 전체 문장과 제목이 너무 다르면(WRatio < 92) 교정 안 함.
    - Gate 2 (부분오차): 교체하려는 부분만 봤을 때도 너무 다르면(CER > 0.34) 교정 안 함.
    """
    if not hyp_raw or not whitelist:
        return hyp_raw, []

    # 1. 명령어 제거 후 순수 제목(target) 추출 시도
    # 예: "무한도잔 틀어줘" -> "무한도잔"
    target = re.sub(
        r"(틀어(줘)?|재생(해(줘)?)?|보여(줘)?|켜(줘)?|해(줘)?|바꿔(줘)?|변경(해(줘)?)?|채널|좀|지금|다시|tv|티비|전원)",
        " ",
        (hyp_raw or "").lower()
    )
    target = re.sub(r"\s+", " ", target).strip()
    
    # 2. 가장 유사한 후보 찾기
    best = process.extractOne(target if len(target) >= 2 else hyp_raw, whitelist, scorer=fuzz.WRatio)
    if not best:
        return hyp_raw, []
    chosen, wr_score, _ = best

    # Gate 1: 전체 유사도 체크 (임계치 미만이면 교정 포기)
    if wr_score < wratio_th:
        return hyp_raw, [{
            "type": "skip",
            "reason": "wratio<th",
            "wratio": float(wr_score),
            "chosen": chosen,
            "target": target
        }]

    # Gate 2: 부분 교체 구간 탐색 및 CER 체크
    s, e, span_cer = best_substring_span_raw(chosen, hyp_raw, tol=tol)
    if s is None or e is None or span_cer > gate:
        return hyp_raw, [{
            "type": "skip",
            "reason": "gate_fail",
            "wratio": float(wr_score),
            "chosen": chosen,
            "span_cer": float(span_cer),
            "target": target
        }]

    # 이미 정확하면 패스
    surface = hyp_raw[s:e]
    if surface == chosen:
        return hyp_raw, [{
            "type": "noop",
            "wratio": float(wr_score),
            "chosen": chosen,
            "span_cer": float(span_cer),
            "target": target
        }]

    # 3. 최종 교체 실행 (String Replacement)
    hyp_pp = hyp_raw[:s] + chosen + hyp_raw[e:]
    replog = [{
        "type": "replace",
        "start": int(s),
        "end": int(e),
        "from": surface,
        "to": chosen,
        "wratio": float(wr_score),
        "span_cer": float(span_cer),
        "target": target
    }]
    return hyp_pp, replog


# -----------------------------
# 7) ASR 래퍼 (Whisper 실행)
# -----------------------------

@dataclass
class ASRConfig:
    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ko"
    beam_size: int = 5

class ASR:
    """Faster-Whisper 모델을 관리하고 추론하는 클래스"""
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        # 모델 로드 (최초 1회 시간이 걸림)
        self.model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)

    def transcribe(self, path: str, hotwords: Optional[List[str]] = None) -> str:
        """
        [음성 인식 실행]
        - path: 오디오 파일 경로
        - hotwords: 우선적으로 인식할 단어 리스트 (Prompting)
        """
        kwargs: Dict[str, Any] = {"language": self.cfg.language, "beam_size": self.cfg.beam_size}
        if hotwords:
            # Whisper는 콤마로 구분된 문자열 형태로 hotwords를 받음
            kwargs["hotwords"] = ",".join(hotwords)
        
        # 실제 추론 발생s
        segs, _ = self.model.transcribe(path, hotwords=hotwords)
        return "".join(s.text for s in segs).strip()


# -----------------------------
# 8) 결과 요약 (Report)
# -----------------------------

def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [실험 결과 집계]
    - 모든 개별 파일의 결과를 모아서 'Hotwords 개수'와 '후처리 여부'에 따른 평균 점수를 냄.
    """
    agg: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in rows:
        k = int(r["top_k"])
        pp = int(r["pp_on"])
        key = (k, pp)

        a = agg.setdefault(
            key,
            {
                "top_k": k, "pp_on": pp, "n": 0,
                "cer_sum": 0.0, "pn_r_sum": 0.0, "pn_c_sum": 0.0, "pn_n": 0,
                "intent_sum": 0.0, "intent_n": 0, "slot_sum": 0.0, "slot_n": 0,
            }
        )
        a["n"] += 1
        a["cer_sum"] += float(r["cer"])

        if r.get("pn_recall") is not None:
            a["pn_r_sum"] += float(r["pn_recall"])
            a["pn_c_sum"] += float(r["pn_cer"])
            a["pn_n"] += 1

        if r.get("intent_acc") is not None:
            a["intent_sum"] += float(r["intent_acc"])
            a["intent_n"] += 1

        if r.get("slot_acc") is not None:
            a["slot_sum"] += float(r["slot_acc"])
            a["slot_n"] += 1

    # 평균 계산
    out = []
    for (k, pp) in sorted(agg.keys()):
        a = agg[(k, pp)]
        pn_n = a["pn_n"]
        out.append({
            "top_k": k,         # Hotwords 개수
            "pp_on": pp,        # 후처리 여부 (0/1)
            "n": a["n"],        # 테스트 샘플 수
            "cer_mean": a["cer_sum"] / max(1, a["n"]), # 평균 오타율
            "pn_recall_mean": (a["pn_r_sum"] / pn_n) if pn_n else None, # 고유명사 인식 성공률
            "pn_cer_mean": (a["pn_c_sum"] / pn_n) if pn_n else None,    # 고유명사 오타율
            "intent_acc_mean": (a["intent_sum"] / a["intent_n"]) if a["intent_n"] else None, # 의도 분류 정확도
            "slot_acc_mean": (a["slot_sum"] / a["slot_n"]) if a["slot_n"] else None,         # 슬롯 추출 정확도
        })
    return out


# -----------------------------
# 9) 메인 실행부 (Execution)
# -----------------------------

# 1. 파일 존재 여부 검증
assert os.path.isdir(AUDIO_FOLDER), f"오디오 폴더 없음: {AUDIO_FOLDER}"
assert os.path.isfile(TRANSCRIPTS_PATH), f"transcripts.json 없음: {TRANSCRIPTS_PATH}"

# 2. 데이터 로딩
transcripts = load_transcripts(TRANSCRIPTS_PATH)
bias = load_json(BIAS_PATH)
epg = load_json(EPG_PATH)
catalog = load_json(CATALOG_PATH)

if not epg:
    print("[WARN] epg.json이 없거나 비어있습니다. hotwords 성능이 떨어질 수 있습니다.")
if not catalog:
    print("[WARN] catalog.json이 없거나 비어있습니다.")

files = glob.glob(os.path.join(AUDIO_FOLDER, "**/*.mp3"), recursive=True)
assert files, f"오디오 없음: {AUDIO_FOLDER}"

# 3. 모델 초기화
cfg = ASRConfig(model_size=ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE, language=ASR_LANG, beam_size=ASR_BEAM)
asr = ASR(cfg)

# NLU용 콘텐츠 풀 (편성표 + 카탈로그 합본)
content_pool = dedup_clean(epg_titles(epg) + catalog_titles(catalog))
rows: List[Dict[str, Any]] = []

# 4. 실험 루프 (Grid Search)
# TOPK_SWEEP: Hotwords 개수를 바꿔가며 테스트
for top_k in TOPK_SWEEP:
    # 현재 상황에 맞는 힌트 단어(Hotwords) 생성
    hot = hotwords_from_context(bias=bias, epg=epg, catalog=catalog, top_k=top_k)

    # POSTPROCESS_SWEEP: 후처리 ON/OFF 테스트
    for pp_on in POSTPROCESS_SWEEP:
        print(f"\n[RUN] top_k={top_k} | hotwords={len(hot)} | pp_on={pp_on} | Gate(WRatio={RULE_WRATIO_TH}, Gate={RULE_GATE}) | NLU={ENABLE_NLU}")

        for ap in files:
            fname = os.path.basename(ap)
            meta = transcripts.get(fname)
            if not meta:
                continue

            # 정답 데이터 추출
            ref = meta.get("text", "")
            ents = meta.get("entities", []) or []


            # A) ASR 실행 (듣기)
            hyp_raw = asr.transcribe(ap, hotwords=hot)

            # B) Whitelist 후보군 탐색 (교정 후보 찾기)
            wl = make_whitelist(hyp_raw, epg=epg, catalog=catalog, top_n=WL_TOPN)
            wl_top5 = wl[:5]

            # C) Post-processing (고치기)
            if pp_on and wl:
                hyp_pp, replog = postprocess_rule_whitelist(
                    hyp_raw, wl,
                    wratio_th=RULE_WRATIO_TH,
                    gate=RULE_GATE,
                    tol=RULE_TOL
                )
            else:
                hyp_pp, replog = hyp_raw, []

            # D) Metrics (채점하기 - 텍스트 정확도)
            cer = cer_norm(ref, hyp_pp)
            pn_r, pn_c = proper_metrics(ents, hyp_pp)

            # # E) NLU + Metrics (이해하기 & 채점)
            # nlu = {"intent": None, "slots": {}}
            # intent_acc, slot_acc = None, None
            # if ENABLE_NLU:
            #     nlu = nlu_parse(hyp_pp, content_pool)
            #     intent_acc, slot_acc = intent_slot_metrics(ref_intent, ref_slots, nlu["intent"], nlu.get("slots", {}))

            # 로그 출력
            print(f"- {fname} | pp_on={pp_on} | cer={cer:.4f} | pn_recall={pn_r}  | wl_size={len(wl)}")

            # 결과 행 저장
            rows.append({
                "file": fname,
                "top_k": top_k,
                "pp_on": pp_on,
                "cer": cer,
                "pn_recall": pn_r,
                "pn_cer": pn_c,
                "hotwords_n": len(hot),
                "hotwords_top10": json.dumps(hot[:10], ensure_ascii=False),
                "wl_size": len(wl),
                "wl_top5": json.dumps(wl_top5, ensure_ascii=False),
                "ref": ref,
                "hyp_raw": hyp_raw,
                "hyp_pp": hyp_pp,
                "replog": json.dumps(replog, ensure_ascii=False),
            })

# 5. CSV 파일 저장
assert rows, "rows가 비었습니다. transcripts.json의 파일명과 AUDIO_FOLDER 파일명이 일치하는지 확인하세요."

# 상세 결과 저장
with open(OUT_ROWS, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# 요약 결과 저장
summary = summarize(rows)
with open(OUT_SUM, "w", newline="", encoding="utf-8-sig") as f:
    fields = list(summary[0].keys()) if summary else ["top_k", "pp_on", "n"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

print("\n[DONE] 실행 완료")
print("- 상세 결과: ", OUT_ROWS)
print("- 요약 결과: ", OUT_SUM)
print("-> 두 CSV 파일을 다운로드하여 보고서에 활용하세요.")
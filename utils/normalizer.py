import re
from g2pk import G2p
from typing import Dict, List, Optional, Any, Tuple


class TextNormalizer:
    def __init__(self):
        self.g2p = G2p()

        #알파벳 -> 한글 변환 맵
        self.char_map = {
            "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지",
            "h": "에이치", "i": "아이", "j": "제이", "k": "케이", "l": "엘", "m": "엠",
            "n": "엔", "o": "오", "p": "피", "q": "큐", "r": "알", "s": "에스",
            "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이", "z": "제트"
        }
        #한글 숫자 발음 단일화 맵
        self.num_sense_map = {"하나": "일", "한": "일", "둘": "이", "두": "이", "셋": "삼", "세": "삼", "여덟": "팔", "열": "십"}

    #숫자 한글 변환 
    def _num_to_ko(self, num_str: str) -> str:
        try:
            n = int(num_str)
            if n == 0: return "영"
            u, t, h = ["","일","이","삼","사","오","육","칠","팔","구"], ["","십","이십","삼십","사십","오십","육십","칠십","팔십","구십"], ["","백","이백","삼백","사백","오백","육백","칠백","팔백","구백"]
            if n >= 100:
                hv, r = divmod(n, 100); tv, uv = divmod(r, 10)
                return h[hv] + (t[tv] if tv != 1 else "십") + u[uv]
            elif n >= 10:
                tv, uv = divmod(n, 10)
                return (t[tv] if tv != 1 else "십") + u[uv]
            return u[n]
        except: return num_str

    #텍스트 정규화(영어, 숫자 -> 한글 변환 및 띄어쓰기 제거)
    def normalize(self, text: str, remove_space: bool = False) -> str:
        if not text: return ""
        s = str(text).lower().strip()

        #영어, 숫자 -> 한글 변환
        s = re.sub(r'\d+', lambda m: self._num_to_ko(m.group()), s)
        for k, v in self.num_sense_map.items():
            s = re.sub(rf'\b{k}\b', v, s)
        for eng, ko in self.char_map.items():
            s = s.replace(eng, ko)

        #remove_space 값에 따라 띄어쓰기 제거
        if remove_space:
            s = re.sub(r"[^0-9\uac00-\ud7a3]", "", s)
        else:
            s = re.sub(r"[^0-9\uac00-\ud7a3\s]", "", s)
            s = re.sub(r"\s+", " ", s).strip()
        return s

"""원본 문자열의 인덱스를 보존하기 위한 유틸리티"""
def build_norm_with_map(raw: str) -> Tuple[str, List[int]]:
    if not raw: return "", []
    raw_l = raw.lower()
    norm_chars, idx_map = [], []
    for i, ch in enumerate(raw_l):
        if re.match(r"[0-9a-z\u3131-\u318e\uac00-\ud7a3]", ch):
            norm_chars.append(ch)
            idx_map.append(i)
    return "".join(norm_chars), idx_map
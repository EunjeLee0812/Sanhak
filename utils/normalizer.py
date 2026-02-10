"""
텍스트 정규화 모듈 (normalizer.py)

- 기존 normalize(): 그대로 유지 (postprocess/span 매칭 안정성)
- 신규 normalize_for_eval(): 평가용(숫자 표준형=아라비아 숫자) 정규화
"""

import re
from g2pk import G2p
from typing import Dict, List, Tuple


class TextNormalizer:
    def __init__(self):
        self.g2p = G2p()

        self.char_map = {
            "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이",
            "f": "에프", "g": "지", "h": "에이치", "i": "아이", "j": "제이",
            "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오",
            "p": "피", "q": "큐", "r": "알", "s": "에스", "t": "티",
            "u": "유", "v": "브이", "w": "더블유", "x": "엑스",
            "y": "와이", "z": "제트"
        }

        self.num_sense_map = {
            "하나": "일",
            "한": "일",
            "둘": "이",
            "두": "이",
            "셋": "삼",
            "세": "삼",
            "여덟": "팔",
            "열": "십"
        }

        # ==========================
        # 평가용 숫자 표준화 세팅
        # ==========================
        self.units = [
            "번", "개", "단계", "편", "화", "회", "위", "배", "퍼센트",
            "분", "초", "시", "시간", "점", "일", "주", "달", "년대", "년",
            "채널"
        ]

        # 긴 단위가 먼저 오도록 정렬 (시간 > 시)
        _units_sorted = sorted(self.units, key=len, reverse=True)
        self._unit_re = "(?:" + "|".join(map(re.escape, _units_sorted)) + ")"

        # 한 자리 숫자어(자리수 읽기용): 일구팔팔 -> 1988
        self._digit_word = {
            "영": "0", "공": "0",
            "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
            "육": "6", "칠": "7", "팔": "8", "구": "9",
        }

        # 한자식 수사 파싱: 이천십 -> 2010, 이십이 -> 22
        self._sino_val = {
            "영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
            "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9
        }
        self._sino_unit = {"십": 10, "백": 100, "천": 1000}

    # -----------------------------
    # (기존) 0~999까지만 한글로 바꾸는 함수
    #  - normalize() 유지 목적이라 그대로 둠
    # -----------------------------
    def _num_to_ko(self, num_str: str) -> str:
        try:
            n = int(num_str)
            if n == 0:
                return "영"

            u = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
            t = ["", "십", "이십", "삼십", "사십", "오십", "육십", "칠십", "팔십", "구십"]
            h = ["", "백", "이백", "삼백", "사백", "오백", "육백", "칠백", "팔백", "구백"]

            if n >= 100:
                hv, r = divmod(n, 100)
                tv, uv = divmod(r, 10)
                tens_part = t[tv] if tv != 1 else "십"
                return h[hv] + tens_part + u[uv]
            elif n >= 10:
                tv, uv = divmod(n, 10)
                tens_part = t[tv] if tv != 1 else "십"
                return tens_part + u[uv]
            return u[n]
        except:
            return num_str

    # ==========================
    # 평가용 숫자 표준화 헬퍼
    # ==========================
    def _convert_digit_sequence(self, s: str) -> str:
        """
        자리수 읽기: "일구팔팔" -> "1988"
        3~6자리 연속 숫자어를 숫자로 변환(연도/번호에 흔함)
        """
        pattern = rf"(?<![0-9\uac00-\ud7a3])(?:영|공|일|이|삼|사|오|육|칠|팔|구){{3,6}}(?=\s*(?:{self._unit_re}|\s|$))"

        def repl(m):
            return "".join(self._digit_word[ch] for ch in m.group())

        return re.sub(pattern, repl, s)

    def _parse_sino_number(self, token: str):
        """
        한자식 수사(0~9999 중심): "이천십" -> 2010, "이십이" -> 22
        """
        has_unit = any(u in token for u in self._sino_unit.keys())
        if not has_unit:
            # 단독 1글자 한자수사는 여기서도 안전 변환 가능하지만
            # eval에서는 단위 붙은 경우만 변환하도록 별도 처리한다.
            return None

        total = 0
        num = 0
        for ch in token:
            if ch in self._sino_val:
                num = self._sino_val[ch]
            elif ch in self._sino_unit:
                unit = self._sino_unit[ch]
                if num == 0:
                    num = 1  # "십"==10 관례
                total += num * unit
                num = 0
            else:
                return None

        total += num
        return total

    def _convert_sino_numbers(self, s: str) -> str:
        """
        '십/백/천' 포함 한자식 수사만 숫자로 변환.
        단, 단어 내부(추천의 '천' 등) 오염 방지.
        """
        # ? 길이 2 이상 + 앞이 한글/숫자면 금지 + (뒤는 단위/공백/끝이면 OK)
        pattern = rf"(?<![0-9\uac00-\ud7a3])[영공일이삼사오육칠팔구십백천]{{2,}}(?=\s*(?:{self._unit_re}|\s|$))"

        def repl(m):
            tok = m.group()
            if not any(u in tok for u in ("십", "백", "천")):
                return tok
            val = self._parse_sino_number(tok)
            return str(val) if val is not None else tok

        return re.sub(pattern, repl, s)

    def _convert_single_sino_if_unit(self, s: str) -> str:
        """
        1글자 한자수사는 '단위가 붙은 경우'에만 숫자화.
        단, '영/공'은 영화/공연 등 오탐이 너무 커서 제외.
        예) "오 화"->"5 화", "사점"->"4점", "오시간"->"5시간"
        """
        allowed = {k: v for k, v in self._sino_val.items() if k not in ("영", "공")}
        # ? 추가: '십'도 단독 수사로 허용 (10분/10시/10개/10위 등)
        allowed["십"] = 10

        sino_single = "|".join(map(re.escape, allowed.keys()))



        # - 앞이 한글/숫자면(단어 내부) 금지: 이후/토요일 같은 케이스 방지
        # - 뒤는 단위가 오면 OK (붙여쓰기/띄어쓰기 모두)
        # - \b 사용 금지(한글에서는 오작동): "오화재생" 같은 케이스를 못 잡음
        pattern = rf"(?<![0-9\uac00-\ud7a3])({sino_single})(?=\s*{self._unit_re})"

        return re.sub(pattern, lambda m: str(allowed[m.group(1)]), s)


    def _convert_native_with_units(self, s: str) -> str:
        """
        구어 수사 + 단위: "한 번" -> "1번", "두단계" -> "2단계"
        단위가 붙은 경우에만 변환해서 과잉 변환을 줄임
        """
        native = {
            "한": 1, "두": 2, "세": 3, "네": 4,
            "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
            "열": 10,
        }
        pattern = rf"({'|'.join(map(re.escape, native.keys()))})\s*({self._unit_re})"

        def repl(m):
            nword = m.group(1)
            unit = m.group(2)
            return f"{native[nword]}{unit}"

        return re.sub(pattern, repl, s)

    def _join_number_units(self, s: str) -> str:
        # "10 개" -> "10개"
        return re.sub(rf"(\d+)\s+({self._unit_re})", r"\1\2", s)

    # -----------------------------
    # (기존) normalize(): 그대로 유지
    # -----------------------------
    def normalize(self, text: str, remove_space: bool = False) -> str:
        if not text:
            return ""

        s = str(text).lower().strip()

        # 2단계: 숫자 -> 한글 (기존 유지; 1000+는 지원X)
        s = re.sub(r"\d+", lambda m: self._num_to_ko(m.group()), s)

        # 3단계: 한글 숫자 발음 통일 (기존 유지)
        for k, v in self.num_sense_map.items():
            s = re.sub(rf"\b{k}\b", v, s)

        # 4단계: 영어 -> 한글 발음 (기존 유지)
        for eng, ko in self.char_map.items():
            s = s.replace(eng, ko)

        # 5단계: 특수문자 제거/공백 정리 (기존 유지)
        s = re.sub(r",", " ", s)

        if remove_space:
            s = re.sub(r"[^0-9\uac00-\ud7a3]", "", s)
        else:
            s = re.sub(r"[^0-9\uac00-\ud7a3\s]", "", s)
            s = re.sub(r"\s+", " ", s).strip()

        return s

    # -----------------------------
    # (신규) 평가용 normalize_for_eval():
    #  - 숫자 표준형=아라비아 숫자
    #  - "열개/십 개/10개" -> "10개"
    #  - "이천십/2010" -> "2010"
    #  - "일구팔팔/1988" -> "1988"
    # -----------------------------
    def normalize_for_eval(self, text: str, remove_space: bool = False) -> str:
        if not text:
            return ""

        s = str(text).lower().strip()
        s = re.sub(r",", " ", s)

        # 0) 기존 구어 치환(열->십 등) 최소 반영 (단, eval에서는 1글자 치환 금지)
        #    - "한/두/세/열" 같은 1글자들은 단위 있을 때만 숫자화하도록(아래 단계1)로 보냄
        for k, v in self.num_sense_map.items():
            if len(k) == 1:
                continue
            s = s.replace(k, v)

        # 1) 구어 수사 + 단위 -> 숫자+단위 (한번/두단계 등)
        s = self._convert_native_with_units(s)

        # 1.5) ? 1글자 한자수사 + 단위 -> 숫자화 (오 화/사 점/이 년 등)
        s = self._convert_single_sino_if_unit(s)

        # 2) 자리수 읽기 -> 숫자 (일구팔팔 -> 1988)
        s = self._convert_digit_sequence(s)

        # 3) 한자식 수사(십/백/천 포함) -> 숫자 (이천십 -> 2010, 이십이 -> 22)
        s = self._convert_sino_numbers(s)

        # 4) 숫자+단위 결합
        s = self._join_number_units(s)

        # ==========================
        # eval 전용: 4K 표준화 (정답 표기= "사 케이"로 통일)
        # ==========================
        # 4K / 4 케이 / 사케이 / 사 케이 / 포케이 / 포 케이  -> "사 케이"

        WS = r"[\s\u200b\u200c\u200d\uFEFF]*"

        s = re.sub(rf"4{WS}k", "사 케이", s, flags=re.IGNORECASE)
        s = re.sub(rf"4{WS}케{WS}이", "사 케이", s)
        s = re.sub(rf"사{WS}케{WS}이", "사 케이", s)
        s = re.sub(rf"포{WS}케{WS}이", "사 케이", s)


        # 5) 영어 -> 한글 발음 (기존과 동일 규칙 적용)
        for eng, ko in self.char_map.items():
            s = s.replace(eng, ko)

        # 6) 특수문자 제거/공백 정리
        if remove_space:
            s = re.sub(r"[^0-9\uac00-\ud7a3]", "", s)
        else:
            s = re.sub(r"[^0-9\uac00-\ud7a3\s]", "", s)
            s = re.sub(r"\s+", " ", s).strip()

        return s


def build_norm_with_map(raw: str) -> Tuple[str, List[int]]:
    """
    기존 그대로 유지 (postprocess에서 raw index mapping에 중요)
    """
    if not raw:
        return "", []

    raw_l = raw.lower()
    norm_chars = []
    idx_map = []

    for i, ch in enumerate(raw_l):
        if re.match(r"[0-9a-z\u3131-\u318e\uac00-\ud7a3]", ch):
            norm_chars.append(ch)
            idx_map.append(i)

    return "".join(norm_chars), idx_map
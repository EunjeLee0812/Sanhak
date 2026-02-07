from typing import List, Optional, Tuple
from rapidfuzz.distance import Levenshtein
from mecab import MeCab

from utils.normalizer import TextNormalizer


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
def calculate_cer(ref: str, hyp: str, *, normalizer=None) -> float:
    normalizer = normalizer or _get_normalizer()
    r = normalizer.normalize(ref, remove_space=True)
    h = normalizer.normalize(hyp, remove_space=True)
    if not r: 
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)

def calculate_wer(ref: str, hyp: str, *, normalizer=None, mecab=None) -> float:
    normalizer = normalizer or _get_normalizer()
    mecab = mecab or _get_mecab()
    r_text = normalizer.normalize(ref, remove_space=False)
    h_text = normalizer.normalize(hyp, remove_space=False)
    r_morphs, h_morphs = mecab.morphs(r_text), mecab.morphs(h_text)
    if not r_morphs:
        return 0.0 if not h_morphs else 1.0
    return Levenshtein.distance(r_morphs, h_morphs) / len(r_morphs)

#인식된 고유명사를 정답 고유명사와 비교해 인식 결과를 교정하는 데 도움을 주는 함수
def best_proper_noun_match(entity: str, hyp: str, normalizer, tol: int = 2) -> Tuple[float, str]:
    normalizer = normalizer or _get_normalizer()
    e = normalizer.normalize(entity, remove_space=True)
    h = normalizer.normalize(hyp, remove_space=True)
    if not e: return 0.0, ""
    if not h: return 1.0, ""

    L = len(e)
    if len(h) <= L: return Levenshtein.distance(e, h) / L, h

    best_score, best_sub = 1.0, ""
    for wlen in range(max(1, L - RULE_TOL), min(len(h), L + RULE_TOL) + 1):
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
) -> Tuple[float, float, List[str], List[str]]:
    """
    return: (pn_recall, avg_pn_cer, hyp_pn(로그용), hard_missed(학습용))
    """
    if not entities:
        return 0.0, 0.0, [], []

    cers: List[float] = []
    hyp_pn: List[str] = []
    hard_missed: List[str] = []

    for entity in entities:
        # ✅ normalizer 반드시 전달
        cer, matched_sub = best_proper_noun_match(entity, hyp_final, normalizer)

        cers.append(cer)
        hyp_pn.append((matched_sub or "").replace(" ", ""))

        # ✅ 학습은 hard miss만
        if cer > hard_th:
            hard_missed.append(entity)

    pn_recall = sum(c <= match_th for c in cers) / len(cers)
    avg_pn_cer = sum(cers) / len(cers)

    return pn_recall, avg_pn_cer, hyp_pn, hard_missed
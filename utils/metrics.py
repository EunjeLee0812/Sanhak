from rapidfuzz.distance import Levenshtein
from typing import List, Tuple, Dict, Any, Optional
# normalizer는 main에서 주입받는 것이 성능상 좋지만, 편의를 위해 import 가능
# from normalizer import TextNormalizer 

def calculate_cer(ref: str, hyp: str, normalizer) -> float:
    r = normalizer.normalize(ref, remove_space=True)
    h = normalizer.normalize(hyp, remove_space=True)
    if not r: return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)

def calculate_wer(ref: str, hyp: str, normalizer, mecab_instance) -> float:
    r_text = normalizer.normalize(ref, remove_space=True)
    h_text = normalizer.normalize(hyp, remove_space=True)
    r_morphs = mecab_instance.morphs(r_text)
    h_morphs = mecab_instance.morphs(h_text)
    if not r_morphs: return 0.0 if not h_morphs else 1.0
    return Levenshtein.distance(r_morphs, h_morphs) / len(r_morphs)

def best_proper_noun_match(entity: str, hyp: str, normalizer, tol: int = 2) -> Tuple[float, str]:
    e = normalizer.normalize(entity, remove_space=True)
    h = normalizer.normalize(hyp, remove_space=True)
    if not e: return 0.0, ""
    if not h: return 1.0, ""
    L = len(e)
    if len(h) <= L: return Levenshtein.distance(e, h) / L, h
    best_score, best_sub = 1.0, ""
    for wlen in range(max(1, L - tol), min(len(h), L + tol) + 1):
        for i in range(0, len(h) - wlen + 1):
            sub = h[i:i + wlen]
            score = Levenshtein.distance(e, sub) / L
            if score < best_score:
                best_score, best_sub = score, sub
                if best_score == 0.0: return 0.0, best_sub
    return best_score, best_sub

def evaluate_proper_nouns(entities: List[str], hyp: str, normalizer, threshold: float = 0.2) -> Tuple[float, float, List[str]]:
    if not entities: return 0.0, 0.0, []
    results = [best_proper_noun_match(e, hyp, normalizer) for e in entities]
    cers = [res[0] for res in results]
    matched_texts = [res[1] for res in results]
    recall = sum(1 for c in cers if c <= threshold) / len(cers)
    avg_pn_cer = sum(cers) / len(cers)
    return recall, avg_pn_cer, matched_texts
import json, random
from typing import List

from config.settings import BIAS_ITERATION_CYCLE_SWEEP


# ==============================================================================
# 3. 적응형 바이어싱 매니저 (Adaptive Bias Manager)
# ==============================================================================

import json, random
from typing import List

class BiasManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with open(db_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.ref_count = self.data.get("ref_count", 0)
        self.data["ref_count"] += 1
        self.session_hits = {}

    def get_weighted_hotwords(self, top_k: int, mode: int = 1) -> List[str]:
        if mode == 1:
            return self.get_random_strategy(top_k)
        elif mode == 2:
            return self.get_hybrid_strategy(top_k)
        return self.get_random_strategy(top_k)

    def get_random_strategy(self, top_k: int) -> List[str]:
        if top_k <= 0: return []
        words = list(self.data["global"].keys())
        weights = [float(self.data["global"][w]) + 1.0 for w in words]
        return random.choices(words, weights=weights, k=min(top_k, len(words)))

    def get_hybrid_strategy(self, top_k: int) -> List[str]:
        if top_k <= 0: return []
        words = list(self.data["global"].keys())
        sorted_words = sorted(words, key=lambda w: self.data["global"][w], reverse=True)
        n_exploit = int(top_k * 0.7)
        n_explore = top_k - n_exploit
        
        candidates_high = sorted_words[:len(words)//2]
        weights_high = [self.data["global"][w] + 1.0 for w in candidates_high]
        chosen_exploit = random.choices(candidates_high, weights=weights_high, k=min(n_exploit, len(candidates_high)))
        
        candidates_low = sorted_words[len(words)//2:]
        chosen_explore = random.sample(candidates_low, k=min(n_explore, len(candidates_low)))
        
        return list(set(chosen_exploit + chosen_explore))

    def add_hit(self, matched_entities: List[str]):
        for ent in matched_entities:
            if ent in self.data["global"]:
                self.session_hits[ent] = self.session_hits.get(ent, 0) + 1

    def finalize(self, iteration_cycle: int):
        # settings에서 정의된 주기(iteration_cycle)를 받아 처리
        if self.ref_count > 0 and self.ref_count % iteration_cycle == 0:
            print(f"\n[LEARNING] {iteration_cycle}회 주기 학습 실행 (현재 {self.ref_count}회차). 가중치를 갱신합니다.")
            for word, count in self.session_hits.items():
                self.data["global"][word] += count
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        else:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_hit(self, matched_entities: List[str]):
        for ent in matched_entities:
            if ent in self.data["global"]:
                self.session_hits[ent] = self.session_hits.get(ent, 0) + 1

    def finalize(self, iteration_cycle: int):
        # settings에서 정의된 주기(iteration_cycle)를 받아 처리
        if self.ref_count > 0 and self.ref_count % iteration_cycle == 0:
            print(f"\n[LEARNING] {iteration_cycle}회 주기 학습 실행 (현재 {self.ref_count}회차). 가중치를 갱신합니다.")

    #biasing_list.json 내 파일참조횟수 count 및 고유명사 count를 0으로 초기화
    def reset_biasing_list(self,path: str):
        # 2. 데이터 수정
        self.data["ref_count"] = 0
        if "global" in self.data:
            for key in self.data["global"]:
                self.data["global"][key] = 0

        # 3. 쓰기 모드(w)로 다시 열어 덮어쓰기
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] {path} 데이터가 모두 0으로 초기화되었습니다.")

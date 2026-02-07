import json, random
from typing import List

from config.settings import BIAS_WEIGHT_UPDATE_CYCLE_SWEEP


# ==============================================================================
# 3. 적응형 바이어싱 매니저 (Adaptive Bias Manager)
# ==============================================================================

class BiasManager:

    def __init__(self, db_path: str):
        self.db_path = db_path
        with open(db_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.ref_count = self.data.get("ref_count", 0)
        self.data["ref_count"] += 1
        # self.session_hit = {}
        self.session_missed = {}

    # 통합 인터페이스: 이 함수가 외부 루프에서 호출됩니다.
    def get_weighted_hotwords(self, top_k: int, mode:int=1) -> List[str]:
        self.data["ref_count"]+=1
        if mode == 1:
            return self.get_weighted_hotwords_random(top_k)
        elif mode == 2:
            return self.get_weighted_hotwords_hybrid(top_k, n_fixed=8)  # 상위 8개 고정(5~10)
    #
    def get_weighted_hotwords_random(self, top_k: int) -> List[str]:
        if top_k <= 0: return []
        words = list(self.data["global"].keys())
        weights = [float(self.data["global"][w]) + 1.0 for w in words]
        return random.choices(words, weights=weights, k=min(top_k, len(words)))
    

    def get_weighted_hotwords_hybrid(self, top_k: int, n_fixed: int = 8) -> List[str]:
        """
        Hybrid 전략(개선판):
        - Exploit: 가중치 상위 n_fixed개는 "항상" 포함 (고정 슬롯)
        - Explore: 나머지는 (1) 가중치 기반 random.choices + (2) 균등 random.sample 섞어서 선택
        - 최종 결과는 길이 top_k를 최대한 맞추고, 중복 제거
        """
        if top_k <= 0: return []

        words = list(self.data.get("global", {}).keys())
        if not words: return []

        # 가중치 내림차순 정렬
        sorted_words = sorted(words, key=lambda w: float(self.data["global"].get(w, 0.0)), reverse=True)

        # n_fixed는 top_k 및 전체 단어 수를 넘지 않게 클램프
        n_fixed = max(0, min(int(n_fixed), top_k, len(sorted_words)))

        # 1) Exploit(고정 포함)
        fixed = sorted_words[:n_fixed]

        # 2) Explore(나머지에서 채우기)
        remain_k = top_k - len(fixed)
        if remain_k <= 0: return fixed[:top_k]

        explore_pool = sorted_words[n_fixed:]
        if not explore_pool: # 단어가 부족하면 fixed만 반환
            return fixed

        # explore를 두 파트로 나눔: weighted + uniform
        n_weighted = remain_k // 2
        n_uniform  = remain_k - n_weighted

        # (a) weighted 선택
        weights = [float(self.data["global"].get(w, 0.0)) + 1.0 for w in explore_pool]  # +1로 0가중치도 선택 가능
        chosen_weighted = random.choices(explore_pool, weights=weights, k=min(n_weighted, len(explore_pool)))

        # (b) uniform 선택 (weighted에서 뽑힌 것 제외하고 샘플링)
        explore_pool_2 = [w for w in explore_pool if w not in set(chosen_weighted)]
        chosen_uniform = random.sample(explore_pool_2, k=min(n_uniform, len(explore_pool_2))) if explore_pool_2 else []

        # 3) 합치고 중복 제거(순서 유지)
        out = []
        seen = set()
        for w in fixed + chosen_weighted + chosen_uniform:
            if w not in seen:
                seen.add(w)
                out.append(w)

        # 4) 그래도 top_k가 덜 찼으면(중복 제거로 줄어든 경우), 남은 풀에서 추가로 채움
        if len(out) < top_k:
            leftovers = [w for w in sorted_words if w not in seen]
            if leftovers:
                need = top_k - len(out)
                # 남은 건 가중치 기반으로 채움
                lw = [float(self.data["global"].get(w, 0.0)) + 1.0 for w in leftovers]
                extra = random.choices(leftovers, weights=lw, k=min(need, len(leftovers)))
                for w in extra:
                    if w not in seen:
                        seen.add(w)
                        out.append(w)

        return out[:top_k]


    def add_hit(self, matched_entities: List[str]):
        for ent in matched_entities:
            if ent in self.data["global"]:
                self.session_missed[ent] = self.session_missed.get(ent, 0) + 1
   
    #add_hit 지우고 add_miss로 변경 (miss인 경우 가중치 누적)
    def add_miss(self, missed_entities: List[str]):
        for ent in missed_entities:
            if ent in self.data["global"]:
                self.session_missed[ent] = self.session_missed.get(ent, 0) + 1

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

    def finalize(self, bias_weight_update_cnt):
        # ✅ 매 iteration마다 누적 반영
        if self.session_missed:
            for word, count in self.session_missed.items():
                self.data["global"][word] += count

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print("[LEARNING] 가중치 데이터베이스 저장 완료.")

        print(f"\n[LEARNING] {bias_weight_update_cnt}회마다 누적 저장 완료. (반복횟수={self.data['ref_count']})")
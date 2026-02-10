"""
바이어싱 매니저 모듈 (bias_manager.py)

이 모듈은 적응형 핫워드 바이어싱(Adaptive Hotwords Biasing)을 관리합니다.
주요 기능:
1. 가중치 기반 핫워드 선택 (Random/Hybrid 전략)
2. 인식 실패한 고유명사 학습 (가중치 증가)
3. 학습 결과를 JSON 파일에 지속적으로 저장
4. 바이어싱 리스트 초기화

핵심 개념:
- Exploit vs Explore: 잘 아는 것 활용 vs 새로운 것 탐색
- Adaptive Learning: 실험을 거듭하며 가중치 자동 조정
"""

import json
import random
from typing import List

from config.settings import BIAS_WEIGHT_UPDATE_ITERATION_SWEEP


# ==============================================================================
# 적응형 바이어싱 매니저 클래스
# ==============================================================================

class BiasManager:
    """
    적응형 핫워드 바이어싱 관리 클래스
    
    이 클래스는 고유명사 인식 성능을 높이기 위해:
    1. 가중치가 높은 단어들을 우선적으로 선택
    2. 인식 실패한 단어들의 가중치를 증가
    3. 학습 결과를 파일에 저장하여 다음 실험에 활용
    
    사용 흐름:
        # 1. 초기화 (가중치 파일 로드)
        bias_mgr = BiasManager("biasing_list.json")
        
        # 2. 핫워드 선택
        hotwords = bias_mgr.get_weighted_hotwords(top_k=20, mode=2)
        
        # 3. ASR 실행 및 평가
        # ... (생략)
        
        # 4. 놓친 고유명사 학습
        bias_mgr.add_miss(["엠비씨", "뉴스데스크"])
        
        # 5. 학습 결과 저장
        bias_mgr.finalize(repeat=1)
    
    데이터 구조 (biasing_list.json):
        {
            "ref_count": 10,  // 파일이 참조된 횟수
            "global": {
                "엠비씨": 5.3,     // 단어: 가중치
                "뉴스데스크": 3.1,
                "티브이엔": 8.7,
                ...
            }
        }
    """
    
    def __init__(self, db_path: str):
        """
        BiasManager 초기화 및 가중치 데이터베이스 로드
        
        Args:
            db_path (str): 가중치 JSON 파일 경로
                예: "/data/lists/biasing_list.json"
                
        동작:
            1. JSON 파일에서 기존 가중치 데이터 로드
            2. 참조 횟수(ref_count) 증가
            3. 세션별 학습 데이터 초기화
            
        파일 형식:
            {
                "ref_count": 0,      // 이 파일이 사용된 횟수
                "global": {          // 전역 가중치 딕셔너리
                    "단어1": 가중치1,
                    "단어2": 가중치2,
                    ...
                }
            }
        """
        # 1. 파일 경로 저장
        self.db_path = db_path
        
        # 2. JSON 파일 로드
        with open(db_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 3. 참조 횟수 증가
        # 기존 ref_count 값 저장 (로그용)
        self.ref_count = self.data.get("ref_count", 0)
        
        # ref_count 1 증가 (이 매니저가 생성될 때마다 +1)
        self.data["ref_count"] += 1
        
        # 4. 세션별 학습 데이터 초기화
        # session_missed: 현재 실험 세션에서 놓친 고유명사들
        # 형식: {"단어": 놓친 횟수}
        # 예: {"엠비씨": 3, "뉴스데스크": 1}
        self.session_missed = {}

    # ==========================================================================
    # 핫워드 선택 메서드
    # ==========================================================================

    def get_weighted_hotwords(self, top_k: int, mode: int = 1) -> List[str]:
        """
        가중치 기반으로 핫워드를 선택하는 통합 인터페이스
        
        이 메서드는 실험 루프에서 호출되며, 전략(mode)에 따라
        적절한 선택 메서드를 호출합니다.
        
        Args:
            top_k (int): 선택할 핫워드 개수
                예: 20 → 20개의 핫워드 선택
                
            mode (int): 선택 전략
                - 1: Random 전략 (완전 가중치 기반 랜덤)
                - 2: Hybrid 전략 (상위 고정 + 나머지 탐색)
                
        Returns:
            List[str]: 선택된 핫워드 리스트
                예: ["엠비씨", "뉴스데스크", "티브이엔", ...]
                
        동작:
            1. ref_count 증가 (이 메서드가 호출될 때마다)
            2. mode에 따라 적절한 선택 메서드 호출
            
        Note:
            - ref_count는 이 메서드가 몇 번 호출되었는지 추적
            - 호출 횟수를 파일에 저장하여 통계 분석에 활용
        """
        # 참조 횟수 증가
        self.data["ref_count"] += 1
        
        # 전략에 따라 분기
        if mode == 1:
            # Random 전략: 가중치 기반 확률적 샘플링
            return self.get_weighted_hotwords_random(top_k)
        elif mode == 2:
            # Hybrid 전략: Exploit + Explore
            return self.get_weighted_hotwords_hybrid(top_k, n_fixed=8)

    def get_weighted_hotwords_random(self, top_k: int) -> List[str]:
        """
        Random 전략: 가중치 기반 확률적 샘플링
        
        가중치가 높은 단어일수록 선택될 확률이 높지만,
        가중치가 낮은 단어도 확률적으로 선택될 수 있습니다.
        
        Args:
            top_k (int): 선택할 핫워드 개수
            
        Returns:
            List[str]: 선택된 핫워드 리스트
            
        동작 방식:
            1. 모든 단어와 가중치 추출
            2. 각 가중치에 +1.0 (0 가중치 단어도 선택 가능하도록)
            3. random.choices로 가중치 기반 샘플링
            
        예시:
            가중치: {"엠비씨": 5.0, "뉴스": 2.0, "티비": 1.0}
            top_k = 2
            
            선택 확률:
            - 엠비씨: 6/(6+3+2) = 54.5%
            - 뉴스: 3/(6+3+2) = 27.3%
            - 티비: 2/(6+3+2) = 18.2%
            
            가능한 결과:
            - ["엠비씨", "뉴스"] (확률 높음)
            - ["엠비씨", "티비"]
            - ["뉴스", "티비"] (확률 낮음)
            
        장점:
            - 구현이 단순
            - 모든 단어에 선택 기회 부여
            
        단점:
            - 가중치가 매우 높은 단어가 빠질 수 있음
            - Exploration이 과도할 수 있음
        """
        # 예외 처리: top_k가 0 이하면 빈 리스트 반환
        if top_k <= 0: 
            return []
        
        # 1. 모든 단어 추출
        # self.data["global"].keys() → ["엠비씨", "뉴스", ...]
        words = list(self.data["global"].keys())
        
        # 2. 각 단어의 가중치 추출 및 +1.0
        # +1.0 이유: 가중치가 0인 단어도 선택 가능하도록
        # 예: 가중치 0.0 → 1.0 (최소한의 선택 확률 보장)
        weights = [float(self.data["global"][w]) + 1.0 for w in words]
        
        # 3. 가중치 기반 랜덤 샘플링
        # random.choices: 복원 추출 (중복 가능)
        # k: 선택할 개수 (top_k와 전체 단어 수 중 작은 값)
        return random.choices(words, weights=weights, k=min(top_k, len(words)))
"""
    def get_weighted_hotwords_hybrid(self, top_k: int, n_fixed: int = 8) -> List[str]:
        
        # Hybrid 전략(개선판):
        # - Exploit: 가중치 상위 n_fixed개는 "항상" 포함 (고정 슬롯)
        # - Explore: 나머지는 (1) 가중치 기반 random.choices + (2) 균등 random.sample 섞어서 선택
        # - 최종 결과는 길이 top_k를 최대한 맞추고, 중복 제거
        
        if top_k <= 0: return []

        words = list(self.data.get("global", {}).keys())
        if not words: 
            return []

        # 2. 가중치 내림차순 정렬
        # key: 정렬 기준 함수 (가중치 값으로 정렬)
        # reverse=True: 높은 값부터 (내림차순)
        sorted_words = sorted(
            words, 
            key=lambda w: float(self.data["global"].get(w, 0.0)), 
            reverse=True
        )
        # 결과 예: ["A"(10.0), "B"(8.0), "C"(5.0), ...]

        # 3. n_fixed 범위 조정
        # n_fixed는 top_k와 전체 단어 수를 넘을 수 없음
        # max(0, ...): 음수 방지
        # min(...): 상한 제한
        n_fixed = max(0, min(int(n_fixed), top_k, len(sorted_words)))

        # 4. Exploit: 상위 n_fixed개 고정 포함
        fixed = sorted_words[:n_fixed]
        # 예: n_fixed=8이면 상위 8개를 fixed에 저장

        # 5. Explore: 나머지 top_k - n_fixed개 채우기
        remain_k = top_k - len(fixed)
        
        # 고정 슬롯으로 이미 다 채워진 경우
        if remain_k <= 0: 
            return fixed[:top_k]

        # 탐색 풀: 고정된 단어를 제외한 나머지
        explore_pool = sorted_words[n_fixed:]
        
        # 탐색 풀이 비어있으면 fixed만 반환
        if not explore_pool: 
            return fixed

        # 6. Explore를 두 부분으로 나눔
        # 절반은 가중치 기반, 절반은 균등 샘플링
        n_weighted = remain_k // 2      # 가중치 기반 개수
        n_uniform = remain_k - n_weighted  # 균등 샘플링 개수

        # 6-1. 가중치 기반 샘플링 (Weighted Random)
        # explore_pool의 각 단어에 대해 가중치 추출 (+1.0)
        weights = [float(self.data["global"].get(w, 0.0)) + 1.0 for w in explore_pool]
        
        # random.choices: 가중치 기반 복원 추출
        chosen_weighted = random.choices(
            explore_pool, 
            weights=weights, 
            k=min(n_weighted, len(explore_pool))
        )

        # 6-2. 균등 샘플링 (Uniform Random)
        # 이미 가중치 기반으로 뽑힌 단어는 제외
        explore_pool_2 = [w for w in explore_pool if w not in set(chosen_weighted)]
        
        # random.sample: 균등 확률로 비복원 추출
        chosen_uniform = random.sample(
            explore_pool_2, 
            k=min(n_uniform, len(explore_pool_2))
        ) if explore_pool_2 else []

        # 7. 합치고 중복 제거
        # 순서: fixed → chosen_weighted → chosen_uniform
        out = []
        seen = set()
        
        for w in fixed + chosen_weighted + chosen_uniform:
            # 중복 체크
            if w not in seen:
                seen.add(w)
                out.append(w)

        # 8. 부족분 채우기
        # 중복 제거로 인해 out 길이가 top_k보다 작을 수 있음
        if len(out) < top_k:
            # 아직 선택되지 않은 단어들
            leftovers = [w for w in sorted_words if w not in seen]
            
            if leftovers:
                need = top_k - len(out)  # 필요한 개수
                
                # 남은 단어들의 가중치 추출
                lw = [float(self.data["global"].get(w, 0.0)) + 1.0 for w in leftovers]
                
                # 가중치 기반으로 추가 샘플링
                extra = random.choices(leftovers, weights=lw, k=min(need, len(leftovers)))
                
                # 중복 체크하며 추가
                for w in extra:
                    if w not in seen:
                        seen.add(w)
                        out.append(w)

        # 9. 최종 결과 반환 (정확히 top_k개)
        return out[:top_k]
"""


    def get_weighted_hotwords_ticket(self, top_k: int) -> List[str]:
        """
        [수정된 알고리즘]
        1. Top 50%: 가중치가 가장 높은 상위 단어들을 고정으로 선택 (Exploit)
        2. Bottom 50%: 나머지 단어들 중에서 가중치 기반 확률(Ticket)로 선택 (Explore)
        """
        if top_k <= 0: return []

        # 1. 전체 단어 가져오기
        words = list(self.data.get("global", {}).keys())
        if not words: return []

        # 2. 전체를 가중치 내림차순으로 정렬 (1등 ~ 꼴등 줄세우기)
        sorted_words = sorted(words, key=lambda w: float(self.data["global"].get(w, 0.0)), reverse=True)

        # 3. 고정 슬롯 개수 계산 (전체의 50%)
        # top_k가 10이면 5개, 11이면 5개
        n_fixed = top_k // 2
        
        # 예외 처리: 전체 단어 수가 top_k보다 적을 경우 안전하게 조정
        n_fixed = min(n_fixed, len(sorted_words))

        # --- [Part 1] 상위 50% 고정 선택 (Exploit) ---
        fixed_part = sorted_words[:n_fixed]

        # --- [Part 2] 나머지 50% 티켓 알고리즘 (Weighted Random) ---
        remain_k = top_k - len(fixed_part)
        
        # 티켓 추첨을 할 후보군 (이미 뽑힌 상위 50% 제외)
        pool = sorted_words[n_fixed:]

        # 더 이상 뽑을 자리가 없거나, 후보가 없으면 고정된 것만 반환
        if remain_k <= 0 or not pool:
            return fixed_part

        # 가중치(티켓) 부여: (기존 가중치 + 1.0)
        # +1.0을 하는 이유는 가중치가 0인 새 단어도 최소한의 당첨 기회를 주기 위함
        weights = [float(self.data["global"].get(w, 0.0)) + 1.0 for w in pool]

        # 티켓 추첨 (중복 방지 로직 포함)
        # random.choices는 복원 추출(중복 허용)이므로, 
        # 목표 개수(remain_k)보다 넉넉하게 뽑은 뒤 중복을 제거하며 채웁니다.
        ticket_part = []
        seen = set()
        
        while len(ticket_part) < remain_k:
            # 넉넉하게 2배수 정도 뽑아봄
            candidates = random.choices(pool, weights=weights, k=remain_k * 2)
            
            added_any = False
            for w in candidates:
                if w not in seen:
                    seen.add(w)
                    ticket_part.append(w)
                    if len(ticket_part) == remain_k:
                        break
                    added_any = True
            
            # 만약 pool에 있는 모든 종류를 다 뽑았는데도 자리가 남으면 루프 탈출 (무한루프 방지)
            if not added_any or len(seen) >= len(pool):
                break
        
        # 4. 최종 결과 반환 (고정 50% + 티켓 50%)
        return fixed_part + ticket_part

    def add_miss(self, missed_entities: List[str]):
        # 만약 data에 'global' 키 자체가 없다면 생성 (빈 파일에서 시작할 때 안전장치)
        if "global" not in self.data:
            self.data["global"] = {}

        for ent in missed_entities:
            # [수정된 부분]
            # 기존: if ent in self.data["global"]: (있을 때만 처리)
            # 변경: 없으면 0.0으로 초기화하여 추가해버림
            if ent not in self.data["global"]:
                self.data["global"][ent] = 0.0 
           
            # 이제 리스트에 확실히 존재하므로, 세션 카운트 증가
            self.session_missed[ent] = self.session_missed.get(ent, 0) + 1



    def reset_biasing_list(self, path: str):
        # 1. 데이터 완전 초기화 (단어 리스트 삭제 및 카운트 0)
        self.data = {
            "global": {},   # 단어 목록을 싹 비움
            "ref_count": 0
        }
        
        # (선택사항) 현재 세션에서 집계 중이던 미스 카운트도 초기화
        # 이 줄이 없으면, reset 후 finalize가 호출될 때 방금 틀린 단어가 다시 추가될 수 있음
        self.session_missed = {} 

        # 2. 파일 덮어쓰기
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] {path} 데이터가 모두 0으로 초기화되었습니다.")

    def finalize(self, bias_weight_update_cnt):
        # ✅ 매 iteration마다 누적 반영
        if self.session_missed:
            for word, count in self.session_missed.items():
                # 기존 가중치에 놓친 횟수만큼 추가
                self.data["global"][word] += count

        # 2. session_missed 초기화
        # 중요: 초기화하지 않으면 다음 반복에서 중복 누적됨
        self.session_missed = {}

        # 3. JSON 파일에 저장
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        # 4. 로그 출력
        print(f"\n[LEARNING] {repeat+1}회차 가중치 누적 저장 완료. (반복횟수={self.data['ref_count']})")
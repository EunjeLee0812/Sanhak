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

    def get_weighted_hotwords_hybrid(self, top_k: int, n_fixed: int = 8) -> List[str]:
        """
        Hybrid 전략: Exploit(활용) + Explore(탐색) 균형
        
        이 전략은 다음과 같이 동작합니다:
        1. Exploit: 가중치 상위 n_fixed개는 항상 포함 (고정 슬롯)
        2. Explore: 나머지는 가중치 기반 + 균등 샘플링 혼합
        
        Args:
            top_k (int): 선택할 총 핫워드 개수
                예: 20
                
            n_fixed (int): 고정으로 포함할 상위 단어 개수
                기본값: 8 (권장 범위: 5~10)
                
        Returns:
            List[str]: 선택된 핫워드 리스트 (중복 없음, 순서 유지)
            
        동작 방식:
            top_k=20, n_fixed=8인 경우:
            
            1. Exploit (8개): 가중치 상위 8개 고정 포함
               → 확실히 중요한 단어들은 항상 포함
               
            2. Explore (12개): 나머지 12개는 탐색
               - 6개: 가중치 기반 샘플링 (높은 가중치 선호)
               - 6개: 균등 샘플링 (모든 단어 동등한 기회)
               
            3. 중복 제거 및 부족분 채우기
               - Set을 사용하여 중복 제거
               - 여전히 top_k 미만이면 남은 풀에서 추가 샘플링
        
        예시:
            가중치: {"A": 10, "B": 8, "C": 5, "D": 3, "E": 1, "F": 0.5}
            top_k=5, n_fixed=2
            
            결과:
            - 고정 (2개): ["A", "B"]
            - 가중치 기반 (1~2개): ["C"] 또는 ["D"]
            - 균등 (1~2개): ["E"] 또는 ["F"]
            
            최종: ["A", "B", "C", "E"] (중복 제거 후)
            
        장점:
            - 중요한 단어는 항상 포함 (안정성)
            - 덜 중요한 단어도 탐색 기회 (다양성)
            - Exploitation과 Exploration 균형
            
        단점:
            - Random보다 복잡
            - n_fixed 값 튜닝 필요
            
        Note:
            - n_fixed가 너무 크면 탐색 부족 (과적합)
            - n_fixed가 너무 작으면 불안정 (중요 단어 누락)
            - 권장: top_k의 30~50% (예: top_k=20이면 n_fixed=6~10)
        """
        # 예외 처리: top_k가 0 이하면 빈 리스트 반환
        if top_k <= 0: 
            return []

        # 1. 전체 단어 리스트 추출
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

    # ==========================================================================
    # 학습 메서드
    # ==========================================================================

    def add_miss(self, missed_entities: List[str]):
        """
        인식 실패한 고유명사를 학습 데이터에 추가
        
        이 메서드는 ASR이 놓친(miss) 고유명사들을 기록합니다.
        나중에 finalize()에서 이 정보를 바탕으로 가중치를 증가시킵니다.
        
        Args:
            missed_entities (List[str]): 놓친 고유명사 리스트
                예: ["엠비씨", "뉴스데스크"]
                
        동작:
            1. 각 놓친 고유명사에 대해
            2. 해당 단어가 global에 있는 경우에만
            3. session_missed에 놓친 횟수 누적
            
        예시:
            # 첫 번째 파일 처리 후
            bias_mgr.add_miss(["엠비씨"])
            # session_missed = {"엠비씨": 1}
            
            # 두 번째 파일 처리 후
            bias_mgr.add_miss(["엠비씨", "뉴스"])
            # session_missed = {"엠비씨": 2, "뉴스": 1}
            
        Note:
            - 실제 가중치 업데이트는 finalize()에서 수행
            - global에 없는 단어는 무시 (오타 방지)
            - session_missed는 현재 세션(반복)에서만 유효
        """
        # 각 놓친 고유명사에 대해
        for ent in missed_entities:
            # global 딕셔너리에 존재하는 경우에만 처리
            # (오타나 잘못된 단어 필터링)
            if ent in self.data["global"]:
                # session_missed에 놓친 횟수 누적
                # get(ent, 0): 기존 값이 없으면 0으로 시작
                self.session_missed[ent] = self.session_missed.get(ent, 0) + 1

    # ==========================================================================
    # 유틸리티 메서드
    # ==========================================================================

    def reset_biasing_list(self, path: str):
        """
        바이어싱 리스트를 초기 상태로 리셋
        
        모든 가중치를 0으로 초기화하고 참조 횟수도 0으로 되돌립니다.
        실험을 처음부터 다시 시작할 때 사용합니다.
        
        Args:
            path (str): 리셋할 JSON 파일 경로
            
        동작:
            1. ref_count를 0으로 설정
            2. global의 모든 가중치를 0으로 설정
            3. 파일에 덮어쓰기
            
        주의:
            - 기존 학습 결과가 모두 사라짐
            - 실행 전 백업 권장
            
        사용 예시:
            bias_mgr.reset_biasing_list("/data/biasing_list.json")
        """
        # 1. ref_count 초기화
        self.data["ref_count"] = 0
        
        # 2. 모든 가중치를 0으로 설정
        if "global" in self.data:
            for key in self.data["global"]:
                self.data["global"][key] = 0

        # 3. 파일에 덮어쓰기
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] {path} 데이터가 모두 0으로 초기화되었습니다.")

    def finalize(self, repeat: int):  # ✅ 수정: 괄호 불일치 해결
        """
        현재 반복(iteration)의 학습 결과를 파일에 저장
        
        이 메서드는 각 학습 반복이 끝날 때 호출되며:
        1. session_missed에 기록된 놓친 횟수를 global 가중치에 반영
        2. 업데이트된 가중치를 JSON 파일에 저장
        3. session_missed 초기화 (다음 반복 준비)
        
        Args:
            repeat (int): 현재 반복 횟수
                예: 1, 2, 3, ...
                (로그 출력용)
                
        동작 방식:
            1. session_missed의 각 단어에 대해
            2. global[단어] += 놓친_횟수
            3. JSON 파일에 저장
            4. session_missed 초기화
            
        예시:
            # 반복 1 실행 전
            global = {"엠비씨": 5.0, "뉴스": 3.0}
            
            # 반복 1 실행 중
            add_miss(["엠비씨"])  # session_missed = {"엠비씨": 1}
            add_miss(["엠비씨", "뉴스"])  # session_missed = {"엠비씨": 2, "뉴스": 1}
            
            # 반복 1 종료
            finalize(repeat=1)
            # global = {"엠비씨": 7.0, "뉴스": 4.0}
            # session_missed = {} (초기화)
            
        학습 효과:
            - 놓친 고유명사의 가중치가 증가
            - 다음 반복에서 해당 단어가 선택될 확률 증가
            - 반복을 거듭할수록 인식 성능 향상
            
        파일 저장 형식:
            {
                "ref_count": 15,
                "global": {
                    "엠비씨": 7.0,
                    "뉴스": 4.0,
                    ...
                }
            }
            
        Note:
            - 중복 누적 방지를 위해 session_missed 반드시 초기화
            - JSON 파일은 ensure_ascii=False로 저장 (한글 깨짐 방지)
            - indent=2로 가독성 있게 저장
        """
        # 1. 학습 결과 반영
        # session_missed에 기록된 놓친 횟수를 global 가중치에 누적
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
"""
ASR 엔진 모듈 (asr_engine.py)

이 모듈은 Faster-Whisper 기반 음성인식(ASR) 엔진을 래핑합니다.
주요 기능:
1. Whisper 모델 초기화 및 관리
2. 오디오 파일 음성 인식 (Transcription)
3. Hotwords(핫워드) 바이어싱을 통한 인식률 향상
"""

from typing import Optional, List, Dict, Any
from faster_whisper import WhisperModel


class ASR:
    """
    Faster-Whisper 기반 음성인식 엔진 클래스
    
    Faster-Whisper는 OpenAI Whisper 모델을 CTranslate2로 최적화한 버전으로,
    원본보다 4배 빠르고 메모리도 적게 사용합니다.
    
    주요 특징:
    - GPU 가속 지원 (CUDA)
    - VAD (Voice Activity Detection) 내장
    - Hotwords 바이어싱 지원
    - Beam search를 통한 정확도 향상
    
    사용 예시:
        asr = ASR(
            model_size="small",
            device="cuda",
            compute_type="float16",
            initial_prompt="엠비씨 뉴스데스크"
        )
        
        result = asr.transcribe(
            path="audio.mp4",
            lang="ko",
            beam_size=5,
            hotwords=["엠비씨", "뉴스데스크"]
        )
        
        print(result)  # "엠비씨 뉴스데스크 틀어줘"
    """
    
    def __init__(self, model_size: str, device: str, compute_type: str, initial_prompt: str):
        """
        ASR 엔진 초기화 및 Whisper 모델 로드
        
        Args:
            model_size (str): Whisper 모델 크기
                - "tiny": 39M 파라미터, 가장 빠름, 정확도 낮음
                - "base": 74M 파라미터, 빠름, 정확도 보통
                - "small": 244M 파라미터, 균형잡힘 (권장)
                - "medium": 769M 파라미터, 느림, 정확도 높음
                - "large": 1550M 파라미터, 매우 느림, 최고 정확도
                - "large-v2", "large-v3": large 모델의 개선 버전
                
            device (str): 연산 장치
                - "cuda": GPU 사용 (NVIDIA)
                - "cpu": CPU 사용 (느림)
                - "auto": 자동 감지
                
            compute_type (str): 연산 정밀도
                - "float32": 32비트 부동소수점 (가장 정확, 가장 느림)
                - "float16": 16비트 부동소수점 (빠름, GPU 전용)
                - "int8": 8비트 정수 (가장 빠름, 정확도 소폭 감소)
                - "int8_float16": int8과 float16 혼합
                
            initial_prompt (str): 초기 프롬프트 (한국어 맥락 제공용)
                Whisper 모델에게 "이런 스타일의 한국어를 인식할 거야"라고 
                미리 알려주는 역할. 인식 정확도 향상에 도움.
                
                예시:
                    "엠비씨 뉴스데스크, 티브이엔 유퀴즈, 넷플릭스 파친코 틀어줘"
                
        동작 방식:
            1. 설정값들을 인스턴스 변수로 저장
            2. WhisperModel 객체 생성 (모델 다운로드 및 로드)
            3. 모델이 메모리에 올라가면 transcribe() 호출 준비 완료
            
        Note:
            - 모델 로드 시간: tiny(수초) ~ large(수십초)
            - GPU 메모리: small(~2GB), large(~10GB)
            - 첫 실행 시 Hugging Face에서 모델 다운로드 (1회만)
            - 다운로드 위치: ~/.cache/huggingface/
        """
        # 1. 한국어 맥락 프롬프트 저장
        # 모든 transcribe 호출에서 기본값으로 사용됨
        self.korean_only_prompt = initial_prompt
        
        # 2. Whisper 모델 로드
        # WhisperModel은 faster-whisper 라이브러리의 핵심 클래스
        self.model = WhisperModel(
            model_size,        # 모델 크기 (tiny, base, small, medium, large)
            device=device,     # 연산 장치 (cuda, cpu)
            compute_type=compute_type  # 연산 정밀도 (float16, int8 등)
        )
        
        # 모델이 성공적으로 로드되면 인스턴스 사용 준비 완료
        # 이후 transcribe() 메서드로 음성 인식 수행 가능

    def transcribe(self, path: str, lang: str, beam_size: int, hotwords: Optional[List[str]] = None) -> str:
        """
        오디오 파일을 텍스트로 변환하는 음성인식 함수
        
        Args:
            path (str): 인식할 오디오 파일 경로
                지원 형식: mp3, mp4, wav, m4a, flac, ogg 등
                예: "/data/audio/sample.mp4"
                
            lang (str): 음성 언어 코드
                - "ko": 한국어
                - "en": 영어
                - "ja": 일본어
                - 등등... (ISO 639-1 코드)
                
            beam_size (int): Beam search 크기
                - 1: Greedy search (가장 빠름, 정확도 낮음)
                - 3~5: 균형잡힘 (권장)
                - 10+: 느리지만 정확도 향상
                
                동작 원리:
                    각 타임스텝마다 가능성 높은 상위 N개 경로를 유지.
                    beam_size가 클수록 더 많은 경로를 탐색하여 정확도 향상.
                    
            hotwords (Optional[List[str]]): 핫워드 리스트 (선택사항)
                모델이 특별히 주목해야 할 단어들의 리스트.
                이 단어들이 오디오에 나올 가능성이 높다고 모델에게 힌트 제공.
                
                예시:
                    hotwords = ["엠비씨", "뉴스데스크", "티브이엔"]
                    
                효과:
                    - "엠비씨" 발음을 "mbc" 대신 "엠비씨"로 인식할 확률 증가
                    - 드문 고유명사 인식률 향상
                    - 정확도 5~15% 향상 (데이터에 따라 다름)
                    
        Returns:
            str: 인식된 텍스트 (전체 세그먼트 결합)
                예: "엠비씨 뉴스데스크 틀어줘"
                
        동작 방식:
            1. 파라미터 딕셔너리 구성
            2. Hotwords가 있으면 쉼표로 연결하여 추가
            3. model.transcribe() 호출하여 세그먼트 단위 인식
            4. 모든 세그먼트의 텍스트를 하나로 결합
            5. 양쪽 공백 제거 후 반환
            
        세그먼트란?
            Whisper는 오디오를 30초 단위로 나누어 처리.
            각 30초 조각을 "세그먼트"라고 하며, 
            각 세그먼트마다 텍스트, 시작/종료 시간, 확신도 등을 반환.
            
        예시:
            asr = ASR("small", "cuda", "float16", "뉴스")
            
            # 핫워드 없이 인식
            text1 = asr.transcribe("audio.mp4", "ko", 5)
            # 결과: "음비씨 뉴스데스크 틀어줘"
            
            # 핫워드와 함께 인식
            text2 = asr.transcribe("audio.mp4", "ko", 5, ["엠비씨"])
            # 결과: "엠비씨 뉴스데스크 틀어줘"
            
        Note:
            - VAD(Voice Activity Detection) 필터가 자동으로 적용되어
              무음 구간은 자동으로 건너뜀
            - 긴 오디오는 자동으로 청크로 나뉘어 처리됨
            - 인식 속도: 실시간보다 3~5배 느림 (GPU 기준)
            - initial_prompt는 모든 세그먼트에 자동으로 적용됨
        """
    # [수정] 프롬프트에 핫워드 주입 (가장 강력한 힌트)
    # 예: "가요톱텐, 강호동, 엠비씨 뉴스데스크, ..." 형태로 만듦
        dynamic_prompt = self.korean_only_prompt

        PROMPT_HOTWORD_LIMIT = 10 
        
        if hotwords and len(hotwords) > 0:
            # 리스트 슬라이싱으로 개수 제한
            safe_hotwords = hotwords[:PROMPT_HOTWORD_LIMIT]
            hotword_str = ", ".join(safe_hotwords)
            dynamic_prompt = f"{hotword_str}, {self.korean_only_prompt}"

        kwargs: Dict[str, Any] = {
            "language": lang,
            "beam_size": beam_size,
            "initial_prompt": dynamic_prompt, 
            "vad_filter": True
        }
        
        # [중요] 파라미터용 핫워드(logit biasing)는 개수 제한 없이 전체(50개 등) 다 넣어도 됨
        # (이건 토큰 길이가 아니라 확률 계산에만 영향을 주므로 안전함)
        if hotwords and len(hotwords) > 0:
            kwargs["hotwords"] = ",".join(hotwords)

        # 에러가 발생하던 지점
        segs, _ = self.model.transcribe(path, **kwargs)
        return "".join(s.text for s in segs).strip()
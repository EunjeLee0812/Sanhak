# 25-2 산학협력 프로젝트2 : 바이어싱 리스트를 이용한 음성 인식 성능 개선
# 📺 TV Voice ASR: Adaptive Proper Noun Biasing & Post-Processing

본 프로젝트는 TV 시청 환경에서 발생하는 **고유명사(프로그램 명, 연예인 이름, 영화 제목 등)의 인식 성능 저하** 문제를 해결하기 위해, 음성 인식 모델(Whisper)에 **적응형 바이어싱(Adaptive Biasing)** 및 **퍼지 매칭 기반 후처리(Post-processing)** 기술을 결합한 연구 프로젝트입니다.

---

## 🚀 주요 특징 (Key Features)

* **Adaptive Bias Manager**: 인식에 실패한 'Hard Miss' 단어들에 대해 가중치를 누적하고, 다음 인식 시 우선적으로 반영하는 **Hybrid(Exploit + Explore)** 추출 전략을 사용합니다.
* **Robust Normalization**: `g2pk` 기반의 한국어 음소 변환과 숫자/영문 정규화를 통해 ASR 모델이 혼동하기 쉬운 텍스트를 최적화합니다.
* **Fuzzy Post-Processor**: Levenshtein Distance와 `WRatio`를 활용하여 인식 결과 내의 오타를 고유명사 리스트와 대조하여 자동 교정합니다.
* **Specialized Metrics**: 단순 CER/WER을 넘어, **PN Recall(고유명사 재현율)**과 **Avg PN CER**을 통해 실질적인 서비스 품질을 정밀하게 측정합니다.

---

## 🏗️ 시스템 아키텍처 (System Architecture)



1.  **Input**: TV/미디어 환경 음성 데이터 (`.mp4`, `.wav`)
2.  **Biasing**: `BiasManager`가 현재 가중치 데이터베이스(`biasing_list.json`)를 기반으로 최적의 Hotwords 추출
3.  **ASR Engine**: `faster-whisper` 엔진이 추출된 Hotwords를 주입받아 음성 인식 수행
4.  **Post-Process**: 인식된 결과 중 유사도가 높은 구간을 고유명사 리스트와 대조하여 최적의 단어로 치환
5.  **Feedback**: 인식 결과 분석 후 미인식된 고유명사를 추적하여 가중치 업데이트 및 다음 이터레이션에 반영

---

## 📂 파일 구조 (File Structure)

| 파일명 | 역할 및 상세 설명 |
| :--- | :--- |
| `config/settings.py` | 프로젝트 경로, 하이퍼파라미터(Weight Update Cycle 등), 모델 설정 관리 |
| `core/bias_manager.py` | 가중치 기반 Hotwords 추출(Hybrid 전략) 및 학습 데이터 업데이트 로직 |
| `core/asr_engine.py` | `faster-whisper` 모델 로드 및 Hotwords 주입 기반 Transcribe 수행 |
| `core/post_processor.py` | Levenshtein 거리 기반 문자열 슬라이딩 윈도우 매칭 및 텍스트 교정 로직 |
| `utils/normalizer.py` | 한글 숫자 변환, 영문 발음 표기화, 특수문자 제거 등 텍스트 정규화 |
| `utils/metrics.py` | CER, WER 및 고유명사 특화 평가 지표(PN Recall) 계산 모듈 |
| `utils/data_loader.py` | JSON 데이터 로드 및 원본 인덱스 보존을 위한 텍스트 매핑 유틸리티 |

---

## 📊 평가 지표 (Evaluation Metrics)

본 프로젝트는 인식 정확도 검증을 위해 다음과 같은 수식을 활용합니다.

* **Character Error Rate (CER)**: 띄어쓰기를 제외한 음절 단위 오차율
    $$CER = \frac{S + D + I}{N}$$
    ($$S$$: 대체, $$D$$: 삭제, $$I$$: 삽입, $$N$$: 전체 음절 수)

  * **Word Error Rate (WER)**: 띄어쓰기를 제외한 어절 단위 오차율(한국어에선 띄어쓰기를 제외한 형태소 단위 오차율)
    $$WER = \frac{S + D + I}{N}$$
    ($$S$$: 대체, $$D$$: 삭제, $$I$$: 삽입, $$N$$: 전체 어절 수)

 **Proper Noun CER (PN_CER)**: 
    음성인식된 텍스트 속 고유명사와 정답 고유명사 간의 CER 값

* **Proper Noun Recall (PN Recall)**: 
    정답 고유명사 중 설정된 임계값($$PN\_MATCH\_TH$$) 이내의 CER로 인식된 단어의 비율

---

## 🛠️ 시작하기 (Getting Started)

### 1. 경로 설정
  config/settings.py 내 파일 경로를 본인 작업공간의 경로에 맞게 재설정해야함. 파일 참조

### 2. 실행
  main.ipynb 실행

### 3. 주요 실행 흐름 (Core Workflow)
본 프로젝트는 다음과 같은 순서로 고유명사 인식 및 학습을 진행합니다.

1.  **가중치 로드**: `BiasManager`를 통해 기존 고유명사 가중치 DB(`biasing_list.json`)를 불러옵니다.
2.  **핫워드 추출**: `get_weighted_hotwords`를 사용하여 현재 가중치가 높은 상위 단어들을 추출합니다. (Hybrid 전략 적용)
3.  **음성 인식**: `ASR.transcribe` 수행 시 추출된 핫워드를 주입하여 고유명사 인식률을 높입니다.
4.  **후처리 교정**: 인식 결과 내 오타를 `postprocess_with_hotwords`를 통해 최종 교정합니다.
5.  **가중치 업데이트**: 인식에 실패한 단어를 추적하여 가중치를 누적 저장(`finalize`)하고 다음 차례에 반영합니다.

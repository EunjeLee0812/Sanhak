import os

# ==============================================================================
# 1. COLAB용 경로
# # ==============================================================================
# BASE_DIR="/content/drive/MyDrive/25-2 산학협력프로젝트/26.1_최종발표/results/" 
# RESULTS_DIR = "/content/drive/MyDrive/25-2 산학협력프로젝트/26.1_최종발표/results/results"
# AUDIO_FOLDER = "/content/drive/MyDrive/25-2 산학협력프로젝트/26.1_최종발표/results/voice_record"
# TRANSCRIPTS_PATH = "/content/drive/MyDrive/25-2 산학협력프로젝트/26.1_최종발표/results/transcripts.json"
# BIAS_PATH = "/content/drive/MyDrive/25-2 산학협력프로젝트/26.1_최종발표/results/lists/biasing_list.json"

# ==============================================================================
# 2. lighteningAI용 서원렬 경로
# ==============================================================================
BASE_DIR="/teamspace/studios/this_studio/Sanhak" 
RESULTS_DIR = "/teamspace/studios/this_studio/Sanhak/results"
AUDIO_FOLDER = "/teamspace/studios/this_studio/Sanhak/voice_record/Audio_files"
TRANSCRIPTS_PATH = "/teamspace/studios/this_studio/Sanhak/voice_record/Audio_files/transcripts.json"
BIAS_PATH = "/teamspace/studios/this_studio/Sanhak/lists/biasing_list.json"

# ==============================================================================
# 2. 실험 변수 (Hyper-parameters)
# ==============================================================================
HOTWORD_TOPK_SWEEP = [20] #20고정
BIAS_WEIGHT_UPDATE_ITERATION_SWEEP = [2]  # 전체 학습 반복 횟수
POSTPROCESS_SWEEP = [1] # 0: OFF, 1: ON
HOTWORD_STRATEGY_SWEEP = [2] # 1: Random, 2: Hybrid 2로 고정
RESET_BIASING_LIST=1 #BIAS_ITERATION_SWEEP[index]만큼의 학습 회차를 돈 후 다음 biasing_list를 초기화할지 [0:OFF, 1:ON]
AUDIO_FILE_MAX= 3        # 사용할 최대 AUDIO FILE의 개수

# ==============================================================================
# 3. 모델 및 로직 설정
# ==============================================================================
ASR_MODEL = "medium"
ASR_DEVICE = "cuda"
ASR_COMPUTE = "float32"
ASR_LANG = "ko"
ASR_BEAM = 5
KOREAN_ONLY_PROMPT = "엠비씨 뉴스데스크, 티브이엔 유퀴즈, 넷플릭스 파친코 틀어줘, 볼륨 십으로 올려줘, 삼십 분 뒤에 티비 꺼줘, 채널 이십이 번으로 바꿔줘, 에이, 비, 씨, 디, 하나, 둘, 셋, 삼십 초, 오 분, 세 칸, 일 배속"

# 후처리 임계값
RULE_WRATIO_TH = 92
RULE_GATE = 0.34
RULE_TOL = 2

#가중치 업데이트 관련 파라미터 추가
PN_MATCH_TH   = float(os.environ.get("PN_MATCH_TH", "0.20"))   # pn_recall 계산용
HARD_MISS_TH  = float(os.environ.get("HARD_MISS_TH", "0.35"))  # 학습용 hard miss 기준
# #Googledrive 마운트(Colab 사이트 사용 시 주석 해제)
from google.colab import drive
drive.mount('/content/drive')

import torch, gc, sys
import os, re, json, glob, csv, random, glob, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from g2pk import G2p
from faster_whisper import WhisperModel
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process, fuzz
from mecab import MeCab
import importlib

# 1. 파일들이 위치한 경로를 시스템 경로에 추가
BASE_PATH = "/content/drive/MyDrive/25-2 산학협력프로젝트/26.1_최종발표/results/"
%cd {BASE_PATH}

# 모듈 Import
from config.settings import *
from utils.normalizer import TextNormalizer
from utils.data_loader import load_transcripts
from utils.metrics import calculate_cer, calculate_wer, evaluate_proper_nouns
from core.asr_engine import ASR
from core.bias_manager import BiasManager
from core.post_processor import postprocess_with_hotwords

#그래픽카드 메모리 남용을 막기 위한 캐시 초기화

gc.collect()
torch.cuda.empty_cache()

# 1-5. 결과 저장 경로[현재 시간 반영해서 파일별 구분 용이]
#results 폴더 없으면 생성
if not os.path.exists(os.path.join(BASE_PATH,"results")):
    os.makedirs(os.path.join(BASE_PATH,"results"))

now = time.gmtime(time.time()+(9*3600)) #한국 시간
formatted = time.strftime("[%Y%m%d_%H%M]", now)
OUT_ROWS = f"./results/{formatted}_asr_detail.csv"
OUT_SUM  = f"./results/{formatted}_asr_summary.csv"

def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # key는 4개이므로 Tuple[int, int, str, int]
    agg: Dict[Tuple[int, int, str, int], Dict[str, Any]] = {}

    for r in rows:
        hotwords_strategy = r.get("hotwords_strategy", "random")
        bias_weight_update_iteration = int(r.get("bias_weight_update_iteration", 0))

        key = (
            int(r["top_k"]),
            int(r["postprocess_on"]),   # 🔥 preprocess_on → postprocess_on
            hotwords_strategy,
            bias_weight_update_iteration,
        )

        a = agg.setdefault(
            key,
            {
                "top_k": key[0],
                "postprocess_on": key[1],
                "hotwords_strategy": key[2],
                "bias_weight_update_iteration": key[3],
                "files_num": 0,
                "cer_sum": 0.0,
                "wer_sum": 0.0,
                "pn_recall_sum": 0.0,
                "pn_cer_sum": 0.0,
            },
        )

        a["files_num"] += 1
        a["cer_sum"] += float(r["cer"])
        a["wer_sum"] += float(r["wer"])
        a["pn_recall_sum"] += float(r.get("pn_recall", 0.0))
        a["pn_cer_sum"] += float(r.get("pn_cer", 0.0))

    out = []
    for _, a in sorted(agg.items()):
        n = max(1, a["files_num"])
        out.append({
            "top_k": a["top_k"],
            "postprocess_on": a["postprocess_on"],
            "hotwords_strategy": a["hotwords_strategy"],
            "bias_weight_update_iteration": a["bias_weight_update_iteration"],
            "files_num": a["files_num"],
            "cer_avg": round(a["cer_sum"] / n, 4),
            "wer_avg": round(a["wer_sum"] / n, 4),
            "pn_recall_avg": round(a["pn_recall_sum"] / n, 4),
            "pn_cer_avg": round(a["pn_cer_sum"] / n, 4),
        })

    return out

# ==============================================================================
# 메인 실행 로직
# ==============================================================================

# 1. 초기화 및 로드
if not os.path.exists(BASE_DIR):
    print("[WARN] Base path not found. Checking local..")

normalizer = TextNormalizer()
mecab = MeCab()
bias_mgr = BiasManager(BIAS_PATH)
transcripts = load_transcripts(TRANSCRIPTS_PATH)
files = glob.glob(os.path.join(AUDIO_FOLDER, "**/*.mp4"), recursive=True)[:AUDIO_FILE_MAX]

# ASR 모델 로드
asr = ASR(ASR_MODEL, ASR_DEVICE, ASR_COMPUTE,initial_prompt=KOREAN_ONLY_PROMPT)

rows: List[Dict[str, Any]] = []  # [수정] 결과 데이터를 저장할 리스트

# 2. 실험 루프
for top_k in HOTWORD_TOPK_SWEEP:
    for hotwords_strategy in HOTWORD_STRATEGY_SWEEP:

        # (선택) 각 전략 시작마다 bias 초기화
        if RESET_BIASING_LIST:
            bias_mgr.reset_biasing_list(BIAS_PATH)

        # ✅ 1번 방식: 반복 횟수는 BIAS_UPDATE_ITERATION
        for repeat in range(BIAS_UPDATE_ITERATION):

            # ✅ repeat마다 hotwords 새로 샘플링 (1번 방식)
            current_hotwords = bias_mgr.get_weighted_hotwords(top_k, mode=hotwords_strategy)

            for pp_on in POSTPROCESS_SWEEP:
                print(f"\n[RUN] Top-K: {top_k} | Iteration: {repeat+1}/{BIAS_UPDATE_ITERATION} | PostProcess: {pp_on}")
                print(f"hotwords : {current_hotwords}\n")

                for audio_path in files:
                    fname = os.path.basename(audio_path)
                    meta = transcripts.get(fname, {"text": "", "entities": []})

                    # 1) ASR
                    hyp_raw = asr.transcribe(audio_path, "ko", ASR_BEAM, hotwords=current_hotwords)

                    # 2) 후처리
                    if pp_on:
                        hyp_final, replog = postprocess_with_hotwords(
                            hyp_raw, current_hotwords, normalizer,
                            gate=RULE_GATE, tol=RULE_TOL, wratio_th=RULE_WRATIO_TH
                        )
                    else:
                        hyp_final, replog = hyp_raw, []

                    # 3) Metrics
                    cer = calculate_cer(meta["text"], hyp_final, normalizer)
                    wer = calculate_wer(meta["text"], hyp_final, normalizer, mecab)

                    # ✅ 4) PN 평가: 너가 수정한 4개 리턴 버전 사용
                    pn_recall, pn_cer, hyp_ents, hard_missed_ents = evaluate_proper_nouns(
                        meta.get("entities", []), hyp_final, normalizer, match_th=PN_MATCH_TH, hard_th=HARD_MISS_TH)

                    # ✅ 1번 방식: hard miss만 학습
                    bias_mgr.add_miss(hard_missed_ents)

                    pn_recall = pn_recall if pn_recall is not None else 0.0

                    # 로그
                    print(
                        f"- file: {os.path.dirname(audio_path).split('/')[-1]}/{fname} | "
                        f"pp_on={pp_on} | cer={cer:.4f} | wer={wer:.4f} | pn_cer={pn_cer:.4f} | pn_recall={pn_recall:.4f}"
                    )
                    print(
                        f"ref_text:  [{meta['text']}]\n"
                        f"hyp_raw:   [{normalizer.normalize(hyp_raw)}]\n"
                        f"hyp_final: [{hyp_final}]\n"
                        f"ref_pn:    {meta.get('entities', [])}\n"
                        f"hyp_pn:    {hyp_ents}\n"
                        f"hard_miss: {hard_missed_ents}\n"
                    )

                    # 결과 저장(컬럼명 정리 권장)
                    rows.append({
                        "file": f"{os.path.dirname(audio_path).split('/')[-1]}/{fname}",
                        "top_k": top_k,
                        "postprocess_on": int(pp_on),  # 이름 명확히
                        "hotwords_strategy": "random" if hotwords_strategy == 1 else "hybrid",
                        "hotwords": current_hotwords,
                        "bias_weight_update_iteration": repeat,  # 

                        "cer": f"{cer:.4f}",
                        "wer": f"{wer:.4f}",
                        "pn_recall": f"{pn_recall:.4f}",
                        "pn_cer": f"{pn_cer:.4f}",

                        "ref_text": meta["text"],
                        "hyp_raw": normalizer.normalize(hyp_raw),
                        "hyp_final": hyp_final,

                        "ref_text_pn": meta.get("entities", []),
                        "hyp_pn": hyp_ents,
                        "hard_missed_pn": hard_missed_ents,

                        "replog": json.dumps(replog, ensure_ascii=False),
                    })

            # repeat 끝에 학습 반영
            bias_mgr.finalize()

with open(OUT_ROWS, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = summarize(rows)
with open(OUT_SUM, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

print("\n[DONE] All experiments finished.")
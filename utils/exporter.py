import os, csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from config.settings import AUDIO_FILE_MAX

def save_results_with_summary(rows: List[Dict[str, Any]], output_path: str):
    if not rows:
        print("[WARN] 저장할 데이터가 없습니다.")
        return

    # 1. 상세 데이터 저장 (Detail)
    # --------------------------------------------------------------------------
    # 상세 데이터에 있는 키들만 추출하여 헤더로 사용
    detail_fieldnames = list(rows[0].keys())
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        # [Writer 1] 상세 데이터용 Writer 생성
        w_detail = csv.DictWriter(f, fieldnames=detail_fieldnames)
        w_detail.writeheader()
        w_detail.writerows(rows)
        
        # 2. 요약 데이터 집계 (Aggregation)
        # ----------------------------------------------------------------------
        agg = {} 

        # [Grand Total] 전체 합계를 위한 변수 초기화
        gt = {
            "total_wrong_char_cnt": 0, "total_char_cnt": 0,
            "total_wrong_morph_cnt": 0, "total_morph_cnt": 0,
            "total_pn_hits": 0.0, "total_pn_ents": 0, # Recall용 (분자/분모)
            "total_wrong_pn_char_cnt": 0.0, "total_pn_char_cnt": 0, # CER용 (분자/분모)
            "file_count": 0
        }

        for r in rows:
            # 그룹화 키 생성
            hotwords_key = str(r["hotwords"])
            
            key = (
                r["top_k"],
                r["postprocess_on"],
                r["hotwords_strategy"],
                r["bias_weight_update_cnt"],
                hotwords_key, 
            )

            if key not in agg:
                agg[key] = {
                    "count": 0, 
                    # "bias_weight_update_cnt": 0, 
                    # "hotwords":"",
                    "total_wrong_char_cnt": 0, "total_char_cnt": 0,
                    "total_wrong_morph_cnt": 0, "total_morph_cnt": 0,
                    "pn_recall_sum": 0.0, "pn_cer_sum": 0.0, "pn_count": 0
                }

            # 데이터 누적
            g = agg[key]
            g["count"] += 1
            g["total_wrong_char_cnt"] += float(r.get("wrong_char_cnt", 0))
            g["total_char_cnt"] += float(r.get("char_cnt", 0))
            g["total_wrong_morph_cnt"] += float(r.get("wrong_morph_cnt", 0)) # 변수명 확인 (wrong_morpheme_cnt)
            g["total_morph_cnt"] += float(r.get("morph_cnt", 0))          # 변수명 확인 (morpheme_cnt)
            # g["bias_weight_update_cnt"]=r.get("bias_weight_update_cnt",0)
            # g["hotwords"]=r.get("hotwords","")

            # PN 지표는 값이 있는 경우(None이 아닌 경우)에만 합산
            if r.get("pn_recall") is not None:
                g["pn_recall_sum"] += float(r["pn_recall"])
                g["pn_cer_sum"] += float(r["pn_cer"])
                g["pn_count"] += 1

            # (2) 전체 합계 집계 (Grand Total용)
            gt["file_count"] += 1
            gt["total_wrong_char_cnt"] += float(r.get("wrong_char_cnt", 0))
            gt["total_char_cnt"] += float(r.get("char_cnt", 0))
            gt["total_wrong_morph_cnt"] += float(r.get("wrong_morph_cnt", 0))
            gt["total_morph_cnt"] += float(r.get("morph_cnt", 0))

                # PN 지표 정밀 계산 (역산 로직)
            # rows에 'ref_text_pn'(정답 리스트)이 있으므로 이를 이용해 분모를 구함
            ents = r.get("ref_pn", [])
            if ents and r.get("pn_recall") is not None:
                # Recall: 분자(맞춘 개수) = 비율 * 전체개수
                n_ents = len(ents)
                gt["total_pn_ents"] += n_ents
                gt["total_pn_hits"] += float(r["pn_recall"]) * n_ents
                
                # CER: 분자(편집거리) = 비율 * 전체글자수
                # 글자수는 공백 제거 기준으로 추산 (normalizer가 없으므로 근사치 사용)
                n_chars = sum(len(str(e).replace(" ", "")) for e in ents)
                gt["total_pn_char_cnt"] += n_chars
                gt["total_wrong_pn_char_cnt"] += float(r["pn_cer"]) * n_chars

        # 3. 요약 데이터 리스트 생성
        # ----------------------------------------------------------------------
        summary_rows = []
        # sorted_keys = sorted(agg.keys())

        for key in agg:
            top_k, pp_on, strat, bias_cnt, hotwords= key
            stats = agg[key]
            
            # bias_cnt=stats["bias_weight_update_cnt"]
            # hw_str=stats["hotwords"]

            # Global Average 계산
            global_cer = stats["total_wrong_char_cnt"] / stats["total_char_cnt"] if stats["total_char_cnt"] > 0 else 0.0
            global_wer = stats["total_wrong_morph_cnt"] / stats["total_morph_cnt"] if stats["total_morph_cnt"] > 0 else 0.0
            
            # PN 지표 (Macro Average)
            avg_pn_recall = stats["pn_recall_sum"] / stats["pn_count"] if stats["pn_count"] > 0 else 0.0
            avg_pn_cer = stats["pn_cer_sum"] / stats["pn_count"] if stats["pn_count"] > 0 else 0.0
            
            summary_row = {
                "file": "Total_Average", # 파일명 대신 요약임을 표시
                "top_k": top_k,
                "postprocess_on": pp_on,
                "hotwords_strategy": strat,
                "bias_weight_update_cnt": bias_cnt,
                "hotwords": hotwords,
                
                "cer": f"{global_cer:.4f}",
                "wer": f"{global_wer:.4f}",
                "pn_recall": f"{avg_pn_recall:.4f}",
                "pn_cer": f"{avg_pn_cer:.4f}",
                
                # [중요] Summary에만 존재하는 필드들
                "replog": f"Total Files: {stats['count']}",
                "total_wrong_char_cnt": stats["total_wrong_char_cnt"],
                "total_char_cnt": stats["total_char_cnt"],
                "total_wrong_morph_cnt": stats["total_wrong_morph_cnt"],
                "total_morph_cnt": stats["total_morph_cnt"]
            }
            summary_rows.append(summary_row)

        # (2) 최종 요약 (Grand Total) - 맨 마지막에 추가
        grand_cer = gt["total_wrong_char_cnt"] / gt["total_char_cnt"] if gt["total_char_cnt"] > 0 else 0.0
        grand_wer = gt["total_wrong_morph_cnt"] / gt["total_morph_cnt"] if gt["total_morph_cnt"] > 0 else 0.0
        grand_pn_recall = gt["total_pn_hits"] / gt["total_pn_ents"] if gt["total_pn_ents"] > 0 else 0.0
        grand_pn_cer = gt["total_wrong_pn_char_cnt"] / gt["total_pn_char_cnt"] if gt["total_pn_char_cnt"] > 0 else 0.0
        
        grand_total_row = {
            "file": "GRAND_TOTAL", # 눈에 띄게 표시
            "top_k": "", "postprocess_on": "", "hotwords_strategy": "",
            "hotwords": "Total_mean", "bias_weight_update_cnt": "",
            
            "cer": f"{grand_cer:.4f}",
            "wer": f"{grand_wer:.4f}",
            "pn_recall": f"{grand_pn_recall:.4f}",
            "pn_cer": f"{grand_pn_cer:.4f}",
            
            "replog": "Total_sum",
            "total_wrong_char_cnt": gt["total_wrong_char_cnt"],
            "total_char_cnt": gt["total_char_cnt"],
            "total_wrong_morph_cnt": gt["total_wrong_morph_cnt"],
            "total_morph_cnt": gt["total_morph_cnt"]
        }
        summary_rows.append(grand_total_row)
        

        # 4. 요약 데이터 저장 (새로운 Writer 사용)
        # ----------------------------------------------------------------------
        if summary_rows:
            f.write("\n") # 상세 데이터와 구분하기 위해 빈 줄 추가
            
            # [핵심 수정] 요약 데이터용 fieldnames를 새로 정의
            # 기존 detail 필드 + 새로 추가된 집계 필드들
            summary_fieldnames = [
                "file", "top_k", "postprocess_on", "hotwords_strategy", 
                "bias_weight_update_cnt", "hotwords", 
                "cer", "wer", "pn_recall", "pn_cer", "replog",
                # 상세 데이터엔 없던 새로운 필드들 추가
                "total_wrong_char_cnt", "total_char_cnt", 
                "total_wrong_morph_cnt", "total_morph_cnt"
            ]
            
            # [Writer 2] 요약 데이터용 Writer 생성
            w_summary = csv.DictWriter(f, fieldnames=summary_fieldnames)
            w_summary.writeheader() # 요약용 헤더를 다시 씀 (구분 명확화)
            w_summary.writerows(summary_rows)

    print(f"[SUCCESS] 상세 및 요약 결과가 {output_path}에 저장되었습니다.")


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # key는 4개이므로 Tuple[int, int, str, int]
    agg: Dict[Tuple[int, int, str, int], Dict[str, Any]] = {}

    for r in rows:
        hotwords_strategy = r.get("hotwords_strategy", "random")
        bias_weight_update_cnt = int(r.get("bias_weight_update_cnt", 0))

        key = (
            int(r["top_k"]),
            int(r["postprocess_on"]),   # 🔥 postprocess_on → postprocess_on
            hotwords_strategy,
            bias_weight_update_cnt,
        )

        a = agg.setdefault(
            key,
            {
                "top_k": key[0],
                "postprocess_on": key[1],
                "hotwords_strategy": key[2],
                "bias_weight_update_cnt": key[3],
                "files_num": 0,
                "global_cer": 0.0,
                "global_wer": 0.0,
                "pn_recall_sum": 0.0,
                "pn_cer_sum": 0.0,
                "pn_count": 0, # 추가
                "total_wrong_char_cnt":0,
                "total_char_cnt":0, 
                "total_wrong_morph_cnt":0, 
                "total_morph_cnt":0
            },
        )

        #파일 개수 및 전체 cer, wer 계산
        a["files_num"] += 1
        a["total_wrong_char_cnt"] += float(r["wrong_char_cnt"])
        a["total_wrong_morph_cnt"] += float(r["wrong_morph_cnt"])
        a["total_char_cnt"]+=r["char_cnt"]
        a["total_morph_cnt"]+=r["morph_cnt"]

        if r.get("pn_recall") is not None:
            a["pn_recall_sum"] += float(r["pn_recall"])
            a["pn_cer_sum"] += float(r["pn_cer"])
            a["pn_count"] += 1

    out = []
    for _, a in sorted(agg.items()):
        n = max(1, a["files_num"])
        out.append({
                "top_k": a["top_k"],
                "postprocess_on": a["postprocess_on"],
                "hotwords_strategy": a["hotwords_strategy"],
                "bias_weight_update_cnt": a["bias_weight_update_cnt"],
                "used_file_num": AUDIO_FILE_MAX,
                # round(값, 4)를 통해 소수점 4자리까지 반올림합니다.
                "cer": round(a["total_wrong_char_cnt"]/max(a["total_char_cnt"],1), 4),
                "wer": round(a["total_wrong_morph_cnt"] / max(1, a["total_morph_cnt"]), 4),
                "pn_recall_avg": round(a["pn_recall_sum"] / max(1, a["pn_count"]), 4), # 수정
                "pn_cer_avg": round(a["pn_cer_sum"] / max(1, a["pn_count"]), 4) # 수정
            })

    return out

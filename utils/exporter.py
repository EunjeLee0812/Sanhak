import os, csv
from typing import Dict, List, Any, Tuple

def save_results_with_summary(rows: List[Dict[str, Any]], output_path: str):
    """
    실험 결과를 CSV로 저장하는 함수.
    
    [기능 변경 사항]
    1. 단순히 데이터를 저장하는 것을 넘어, 'experiment_type' 키를 기준으로
       'Proposed(제안 방법)'과 'Baseline(베이스라인)' 데이터를 분리하여 저장합니다.
    2. CSV 파일 내에서 섹션을 나누어 가독성을 높였습니다.
    3. 요약(Summary) 통계도 실험 타입별로 각각 계산하여 하단에 첨부합니다.
    """
    if not rows:
        print("[WARN] 저장할 데이터가 없습니다.")
        return

    # 1. 데이터 분리: 리스트 컴프리헨션을 사용하여 실험 타입별로 행을 나눕니다.
    # - proposed: 핫워드 바이어싱 + 프롬프트 적용 (우리가 연구한 모델)
    # - baseline: 순정 Whisper (비교군)
    rows_proposed = [r for r in rows if r.get("experiment_type") == "proposed"]
    rows_baseline = [r for r in rows if r.get("experiment_type") == "baseline"]

    # CSV 헤더(Fieldnames) 추출 (모든 행이 동일한 키를 가진다고 가정)
    fieldnames = list(rows[0].keys()) if rows else []
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)


        # ======================================================================
        # [SECTION 1] 상세 데이터 저장 (Detail)
        # ====================================================================++
        # (1) Baseline Data (베이스라인) 저장

        if rows_baseline:
            f.write("=== [SECTION 1] DETAIL: WITHOUT Prompt & Hotwords (Baseline / 비교군) ===\n")
            f.write("\n")
            writer.writeheader()
            writer.writerows(rows_baseline)
            f.write("\n")
        # (2) Proposed Data (제안 방법) 저장
        if rows_proposed:
            f.write("=== [SECTION 2] DETAIL: WITH Prompt & Hotwords (Proposed / 제안 방법) ===\n")
            f.write("\n") # 가독성을 위한 빈 줄 추가
            writer.writeheader()
            writer.writerows(rows_proposed)


        # ======================================================================
        # [SECTION 2] 요약 데이터 집계 및 저장 (Summary Statistics)
        # ======================================================================
        
        # 요약 데이터 전용 필드 정의 (기존 상세 데이터 필드와 다름)
        summary_fieldnames = [
            "experiment_type", "file", "top_k", "postprocess_on", "hotwords_strategy", 
            "bias_weight_update_cnt", "hotwords", 
            "cer", "wer", "pn_recall", "pn_cer", "pn_wer", "replog",
            "total_wrong_char_cnt", "total_char_cnt", 
            "total_wrong_morph_cnt", "total_morph_cnt",
            "total_wrong_pn_morph_cnt", "total_pn_morph_cnt", 
            "total_file_cnt"
        ]
        
        f.write("\n\n")
        f.write("=== [SECTION 3] SUMMARY STATISTICS (실험 요약) ===\n\n")
        writer_sum = csv.DictWriter(f, fieldnames=summary_fieldnames)

        # (1) Baseline 요약 계산 및 저장
        if rows_baseline:
            writer_sum.writeheader()
            # Baseline은 파라미터 변화가 없으므로 하나의 행으로 요약됩니다.
            summary_rows_base = _calculate_summary(rows_baseline, "BASELINE_TOTAL_AVG")
            writer_sum.writerows(summary_rows_base)

        # (2) Proposed 요약 계산 및 저장
        if rows_proposed:
            f.write("\n")
            writer_sum.writeheader()
            summary_rows_prop = _calculate_summary(rows_proposed, "PROPOSED_TOTAL_AVG")
            writer_sum.writerows(summary_rows_prop)


    print(f"[SUCCESS] 상세 및 비교 요약 결과가 {output_path}에 저장되었습니다.")


def _calculate_summary(target_rows: List[Dict[str, Any]], total_label: str) -> List[Dict[str, Any]]:
    
    """
    [내부 함수] 특정 행 리스트(rows)를 받아 그룹별 요약 통계를 계산합니다.
    [수정 함수] 
    baseline 리스트를 순회하며 '연속된' 동일 설정 그룹끼리만 묶어서 요약합니다.
    예: Baseline(cnt=1) -> Baseline(cnt=2) -> Baseline(cnt=1) 순서로 들어오면
        3개의 별도 요약 행이 생성됩니다. (기존엔 Key가 같으면 다 합쳐졌음)
    """

    summary_rows = []
    
    # 전체 합계(Grand Total)를 계산하기 위한 변수
    gt = { 
        "total_wrong_char_cnt": 0, "total_char_cnt": 0,
        "total_wrong_morph_cnt": 0, "total_morph_cnt": 0,
        "total_wrong_pn_char_cnt": 0, "total_pn_char_cnt": 0,
        "total_wrong_pn_morph_cnt": 0, "total_pn_morph_cnt": 0,
        "pn_recall_sum": 0.0, "pn_cer_sum": 0.0, "pn_count": 0,
        "file_count": 0
    }

    # 현재 처리 중인 그룹의 상태 (초기화)
    current_key = None
    current_stats = None

    for r in target_rows:
        # 그룹화 키 생성
        key = (
            r.get("top_k", ""),
            r.get("postprocess_on", ""),
            r.get("hotwords_strategy", ""),
            r.get("bias_weight_update_cnt", ""),
            str(r.get("hotwords", "")),
            r.get("experiment_type", "")
        )

        # 1. 키가 바뀌었는지 확인 (새로운 그룹 시작)
        if key != current_key:
            # 기존 그룹이 있었다면 저장 (Flush)
            if current_stats is not None:
                summary_rows.append(_create_summary_row(current_key, current_stats))
            
            # 새 그룹 초기화
            current_key = key
            current_stats = {
                "count": 0,
                "total_wrong_char_cnt": 0, "total_char_cnt": 0,
                "total_wrong_morph_cnt": 0, "total_morph_cnt": 0,
                "total_wrong_pn_char_cnt": 0, "total_pn_char_cnt": 0,
                "total_wrong_pn_morph_cnt": 0, "total_pn_morph_cnt": 0,
                "pn_recall_sum": 0.0, "pn_cer_sum": 0.0, "pn_count": 0
            }

        # 2. 현재 그룹에 데이터 누적 (Dictionary Aggregation이 아님)
        g = current_stats
        g["count"] += 1
        g["total_wrong_char_cnt"] += float(r.get("wrong_char_cnt", 0))
        g["total_char_cnt"] += float(r.get("char_cnt", 0))
        g["total_wrong_morph_cnt"] += float(r.get("wrong_morph_cnt", 0))
        g["total_morph_cnt"] += float(r.get("morph_cnt", 0))
        g["total_wrong_pn_char_cnt"] += float(r.get("wrong_pn_char_cnt", 0))
        g["total_pn_char_cnt"] += float(r.get("pn_char_cnt", 0))
        g["total_wrong_pn_morph_cnt"] += float(r.get("wrong_pn_morph_cnt", 0))
        g["total_pn_morph_cnt"] += float(r.get("pn_morph_cnt", 0))

        if r.get("pn_recall") is not None:
            g["pn_recall_sum"] += float(r["pn_recall"])
            g["pn_cer_sum"] += float(r["pn_cer"])
            g["pn_count"] += 1
            
            # GT 누적
            gt["pn_recall_sum"] += float(r["pn_recall"])
            gt["pn_count"] += 1

        # Grand Total 누적
        gt["file_count"] += 1
        gt["total_wrong_char_cnt"] += float(r.get("wrong_char_cnt", 0))
        gt["total_char_cnt"] += float(r.get("char_cnt", 0))
        gt["total_wrong_morph_cnt"] += float(r.get("wrong_morph_cnt", 0))
        gt["total_morph_cnt"] += float(r.get("morph_cnt", 0))
        gt["total_wrong_pn_char_cnt"] += float(r.get("wrong_pn_char_cnt", 0))
        gt["total_pn_char_cnt"] += float(r.get("pn_char_cnt", 0))
        gt["total_wrong_pn_morph_cnt"] += float(r.get("wrong_pn_morph_cnt", 0))
        gt["total_pn_morph_cnt"] += float(r.get("pn_morph_cnt", 0))

    # 3. 마지막 그룹 저장 (루프가 끝나서 저장되지 못한 마지막 항목 처리)
    if current_stats is not None:
        summary_rows.append(_create_summary_row(current_key, current_stats))

    # 4. 전체 평균(Grand Total) 행 추가
    gt_row = {
        "experiment_type": "TOTAL",
        "file": total_label, 
        "hotwords": "ALL_AGGREGATED",
        "cer": f"{gt['total_wrong_char_cnt'] / max(gt['total_char_cnt'], 1):.4f}",
        "wer": f"{gt['total_wrong_morph_cnt'] / max(gt['total_morph_cnt'], 1):.4f}",
        "pn_recall": f"{gt['pn_recall_sum'] / max(gt['pn_count'], 1):.4f}",
        "pn_cer": f"{gt['total_wrong_pn_char_cnt'] / max(gt['total_pn_char_cnt'], 1):.4f}",
        "pn_wer": f"{gt['total_wrong_pn_morph_cnt'] / max(gt['total_pn_morph_cnt'], 1):.4f}",
        "replog": "Total_Sum",
        "total_wrong_char_cnt": gt["total_wrong_char_cnt"],
        "total_char_cnt": gt["total_char_cnt"],
        "total_wrong_morph_cnt": gt["total_wrong_morph_cnt"],
        "total_morph_cnt": gt["total_morph_cnt"],
        "total_wrong_pn_morph_cnt": gt["total_wrong_pn_morph_cnt"],
        "total_pn_morph_cnt": gt["total_pn_morph_cnt"],
        "total_file_cnt": gt['file_count']
    }
    summary_rows.append(gt_row)
    
    return summary_rows

def _create_summary_row(key, stats):
    """요약 행을 생성하는 헬퍼 함수"""
    top_k, pp_on, strat, bias_cnt, hw, exp_type = key
    return {
        "experiment_type": exp_type,
        "file": "Group_Average",
        "top_k": top_k, "postprocess_on": pp_on,
        "hotwords_strategy": strat, "bias_weight_update_cnt": bias_cnt,
        "hotwords": hw,
        
        "cer": f"{stats['total_wrong_char_cnt'] / max(stats['total_char_cnt'], 1):.4f}",
        "wer": f"{stats['total_wrong_morph_cnt'] / max(stats['total_morph_cnt'], 1):.4f}",
        "pn_recall": f"{stats['pn_recall_sum'] / max(stats['pn_count'], 1):.4f}",
        "pn_cer": f"{stats['total_wrong_pn_char_cnt'] / max(stats['total_pn_char_cnt'], 1):.4f}",
        "pn_wer": f"{stats['total_wrong_pn_morph_cnt'] / max(stats['total_pn_morph_cnt'], 1):.4f}",
        
        "replog": f"Files: {stats['count']}",
        "total_wrong_char_cnt": stats["total_wrong_char_cnt"],
        "total_char_cnt": stats["total_char_cnt"],
        "total_wrong_morph_cnt": stats["total_wrong_morph_cnt"],
        "total_morph_cnt": stats["total_morph_cnt"],
        "total_wrong_pn_morph_cnt": stats["total_wrong_pn_morph_cnt"],
        "total_pn_morph_cnt": stats["total_pn_morph_cnt"],
        "total_file_cnt": stats['count'],
    }
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 비대화형 백엔드 (show() 불필요)
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from config import LABEL_NAMES, OUTPUT_DIR
from feature import FEATURE_COLS


# =========================================================
# SHAP 분석 (XGBoost 전용)
# =========================================================

def run_shap(model, X, split_name="Test"):
    """
    XGBoost 모델에 대해 SHAP 분석 수행

    출력
    ----
    클래스별 summary plot (beeswarm) + bar plot
    → output/ 폴더에 PNG 저장

    Returns
    -------
    explainer   : shap.TreeExplainer
    shap_values : ndarray or list
    """
    print(f"\n{'='*60}")
    print(f"[{split_name}] SHAP 분석 시작")
    print(f"{'='*60}")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_per_class = shap_values
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_per_class = [shap_values[:, :, i]
                              for i in range(shap_values.shape[2])]
        elif shap_values.ndim == 2:
            print("경고: SHAP 값이 2D 배열입니다. 클래스 분리 불가.")
            shap_per_class = [shap_values]
        else:
            print("경고: 예상치 못한 SHAP 배열 형태입니다.")
            return explainer, shap_values
    else:
        print("경고: 예상치 못한 SHAP 타입입니다.")
        return explainer, shap_values

    for class_idx, class_name in enumerate(LABEL_NAMES):
        if class_idx >= len(shap_per_class):
            break
        sv = shap_per_class[class_idx]
        _plot_beeswarm(sv, X, class_name, split_name)
        _plot_bar(sv, X, class_name, split_name)

    print(f"  plot 저장 완료 (output/ 폴더)")
    print(f"[{split_name}] SHAP 분석 완료")
    return explainer, shap_values


# =========================================================
# Beeswarm plot
# =========================================================

def _plot_beeswarm(shap_vals, X, class_name, split_name):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, show=False)
    plt.title(f"[{split_name}] {class_name} - SHAP Beeswarm")
    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"shap_{split_name.lower()}_{class_name.lower()}_beeswarm.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


# =========================================================
# Bar plot
# =========================================================

def _plot_bar(shap_vals, X, class_name, split_name):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.title(f"[{split_name}] {class_name} - SHAP Bar")
    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR,
        f"shap_{split_name.lower()}_{class_name.lower()}_bar.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


# =========================================================
# 포렌식 집중 분석 (Replay / Spoofing 전용) — 글로벌 요약
# =========================================================

def run_shap_forensic(model, X, split_name="Test"):
    """
    Replay(1)과 Spoofing(2) 클래스에 대해서만 SHAP 분석 수행.
    전체 run_shap()과 별개로 호출 가능.
    """
    FORENSIC_CLASSES = {
        LABEL_NAMES.index("Replay"):   "Replay",
        LABEL_NAMES.index("Spoofing"): "Spoofing",
    }

    print(f"\n{'='*60}")
    print(f"[{split_name}] SHAP 포렌식 집중 분석 (Replay / Spoofing)")
    print(f"{'='*60}")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_per_class = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_per_class = [shap_values[:, :, i]
                          for i in range(shap_values.shape[2])]
    else:
        print("경고: shap_values 형태를 처리할 수 없습니다.")
        return

    saved = []
    for class_idx, class_name in FORENSIC_CLASSES.items():
        if class_idx >= len(shap_per_class):
            continue

        sv = shap_per_class[class_idx]
        _plot_beeswarm(sv, X, f"Forensic_{class_name}", split_name)
        _plot_bar(sv, X, f"Forensic_{class_name}", split_name)
        saved.append(class_name)

    print(f"  포렌식 plot 저장 완료: {', '.join(saved)}")


# =========================================================
# 프레임 수준 증거 역추적
# =========================================================

def _trace_evidence(window_df, top_feature_names):
    """
    SHAP 상위 feature를 원본 CAN 프레임 수준 증거로 역추적.

    Parameters
    ----------
    window_df         : 해당 윈도우의 원본 CAN 프레임 DataFrame
                        (Timestamp, Arbitration_ID, DLC, Data, y_msg 포함)
    top_feature_names : SHAP 상위 feature 이름 리스트

    Returns
    -------
    evidence : dict  {feature_name: {증거 항목들}}
    """
    evidence    = {}
    id_series   = window_df["Arbitration_ID"]
    data_series = window_df["Data"].astype(str)
    ts_series   = window_df["Timestamp"].values
    total_msgs  = len(window_df)

    id_counts      = id_series.value_counts()
    id_data_key    = id_series.astype(str) + "|" + data_series
    id_data_counts = id_data_key.value_counts()

    for feat in top_feature_names:
        info = {}

        frame_idx = list(window_df.index)  # 원본 프레임 인덱스

        # ── ID 분포 관련 ───────────────────────────────────────────
        if any(k in feat for k in ["top1_id", "top3_id", "id_entropy",
                                    "unique_id_count"]):
            top5 = id_counts.head(5)
            info["top_ids"] = {
                f"0x{int(aid):03X}" if str(aid).isdigit() else str(aid): int(cnt)
                for aid, cnt in top5.items()
            }
            info["top1_ratio"]      = round(float(top5.iloc[0]) / total_msgs, 4) if len(top5) else 0
            info["unique_id_count"] = int(id_series.nunique())
            info["total_frames"]    = total_msgs
            # 지배적 ID의 프레임 인덱스 (최대 10개)
            if len(top5):
                top1_id = top5.index[0]
                info["frame_indices"] = [
                    int(i) for i in window_df[id_series == top1_id].index[:10]
                ]

        # ── (ID, Payload) 반복 관련 ───────────────────────────────
        elif any(k in feat for k in ["repeat_id_data", "unique_id_data",
                                      "max_same_payload"]):
            top3 = id_data_counts.head(3)
            info["top_repeated_pairs"] = {k: int(v) for k, v in top3.items()}
            info["repeat_ratio"] = round(
                float((id_data_counts > 1).sum()) / len(id_data_counts), 4
            ) if len(id_data_counts) else 0
            # 가장 긴 동일 payload 연속 런
            run_len, cur_len = 0, 1
            vals = data_series.tolist()
            for i in range(1, len(vals)):
                if vals[i] == vals[i - 1]:
                    cur_len += 1
                    run_len = max(run_len, cur_len)
                else:
                    cur_len = 1
            info["max_payload_run"] = run_len
            # 가장 많이 반복된 (ID|payload) 쌍의 프레임 인덱스 (최대 10개)
            if len(top3):
                top_pair = top3.index[0]
                mask = (id_data_key == top_pair)
                info["frame_indices"] = [int(i) for i in window_df[mask].index[:10]]

        # ── payload entropy / 다양성 관련 ─────────────────────────
        elif any(k in feat for k in ["payload_entropy", "payload_byte"]):
            vc = data_series.value_counts()
            info["unique_payloads"]         = int(data_series.nunique())
            info["total_frames"]            = total_msgs
            info["most_common_payload"]     = str(vc.index[0]) if len(vc) else ""
            info["most_common_payload_cnt"] = int(vc.iloc[0]) if len(vc) else 0
            # 가장 많은 payload의 프레임 인덱스 (최대 10개)
            if len(vc):
                mask = (data_series == vc.index[0])
                info["frame_indices"] = [int(i) for i in window_df[mask].index[:10]]

        # ── IAT / 타이밍 관련 ─────────────────────────────────────
        elif any(k in feat for k in ["iat", "burstiness"]):
            if len(ts_series) > 1:
                iats = np.diff(ts_series)
                info["iat_mean_ms"]  = round(float(np.mean(iats)) * 1000, 4)
                info["iat_std_ms"]   = round(float(np.std(iats))  * 1000, 4)
                info["iat_min_ms"]   = round(float(np.min(iats))  * 1000, 4)
                burst_thr            = np.mean(iats) * 0.1
                burst_mask           = iats < burst_thr
                info["burst_frames"] = int(burst_mask.sum())
                # 버스트 발생 프레임 인덱스 (최대 10개) — iats[i]는 frame[i+1] 직전
                burst_positions = np.where(burst_mask)[0] + 1
                info["frame_indices"] = [int(frame_idx[p]) for p in burst_positions[:10]]
            else:
                info["note"] = "프레임 수 부족"

        # ── payload 변화량 관련 ────────────────────────────────────
        elif any(k in feat for k in ["byte_diff", "bit_flip"]):
            if len(data_series) > 1:
                changes = sum(
                    1 for i in range(1, len(data_series))
                    if data_series.iloc[i] != data_series.iloc[i - 1]
                )
                info["payload_change_rate"] = round(changes / (len(data_series) - 1), 4)
            info["note"] = "연속 프레임 간 payload 변화량"

        # ── msg_count ──────────────────────────────────────────────
        elif feat == "msg_count":
            info["msg_count"] = total_msgs

        # ── DLC 관련 ───────────────────────────────────────────────
        elif any(k in feat for k in ["dlc"]):
            if "DLC" in window_df.columns:
                info["mean_dlc"] = round(float(window_df["DLC"].mean()), 4)
                info["std_dlc"]  = round(float(window_df["DLC"].std()),  4)

        # ── ID-level payload summary / Jaccard ──────────────────────
        elif any(k in feat for k in ["id_payload", "changed_id", "jaccard"]):
            info["note"] = "윈도우 레벨 집계 피처 (직접 역추적 불가)"

        else:
            info["note"] = "참조용 피처"

        evidence[feat] = info

    return evidence


# =========================================================
# 포렌식 서술 생성 (숫자 → 조사관 언어)
# =========================================================

def _forensic_narrative(feat_name, actual, normal_mean, evidence_info):
    """
    SHAP 상위 피처 + 프레임 증거를 바탕으로 포렌식 서술 생성.
    '모델이 이 피처를 중요하게 봤다'가 아니라
    '조사관이 왜 이 구간을 공격으로 판단해야 하는가'에 초점.
    """
    direction = "높음" if actual > normal_mean else "낮음"

    if any(k in feat_name for k in ["repeat_id_data", "max_same_payload", "top1_id_data"]):
        pairs = evidence_info.get("top_repeated_pairs", {})
        rr    = evidence_info.get("repeat_ratio", 0)
        mr    = evidence_info.get("max_payload_run", 0)
        if pairs:
            top_pair = next(iter(pairs))
            top_cnt  = pairs[top_pair]
            return (f"동일 (ID|payload) 쌍 '{top_pair}'가 {top_cnt}회 반복 재전송 확인 "
                    f"(반복 비율 {rr:.1%}, 최장 연속 {mr}회)")
        return f"(ID, payload) 반복 비율 {actual:.3f}  정상평균 {normal_mean:.3f} → {direction}"

    elif any(k in feat_name for k in ["iat_std", "iat_mean", "iat_min", "iat_max", "burstiness"]):
        iat_std_ms = evidence_info.get("iat_std_ms", None)
        burst      = evidence_info.get("burst_frames", None)
        if iat_std_ms is not None:
            reg  = "매우 균일 → 기계적 주입 패턴 의심" if iat_std_ms < 1.0 else "불규칙"
            base = f"프레임 간격 표준편차 {iat_std_ms:.3f}ms ({reg})"
            if burst is not None and burst > 0:
                base += f", 집중 버스트 {burst}회 감지"
            return base
        return f"IAT 실측 {actual:.6f}s  정상평균 {normal_mean:.6f}s → {direction}"

    elif any(k in feat_name for k in ["payload_entropy", "payload_byte"]):
        unique     = evidence_info.get("unique_payloads",         None)
        common     = evidence_info.get("most_common_payload",     "")
        common_cnt = evidence_info.get("most_common_payload_cnt", 0)
        if unique is not None:
            return (f"유일 payload {unique}종 — "
                    f"최다 payload '{common}' {common_cnt}회 (단조로운 payload → Replay 특성)")
        return f"payload 엔트로피 {actual:.4f}  정상평균 {normal_mean:.4f} → {direction}"

    elif any(k in feat_name for k in ["top1_id_ratio", "top3_id_ratio", "id_entropy",
                                       "unique_id_count"]):
        top_ids    = evidence_info.get("top_ids", {})
        top1_ratio = evidence_info.get("top1_ratio", None)
        if top_ids:
            top_id  = next(iter(top_ids))
            top_cnt = top_ids[top_id]
            ratio_s = f"{top1_ratio:.1%}" if top1_ratio is not None else ""
            return f"ID {top_id}가 {top_cnt}회로 지배적 점유 ({ratio_s}) → 특정 ID 집중 공격 의심"
        return f"ID 분포 실측 {actual:.4f}  정상평균 {normal_mean:.4f} → {direction}"

    elif any(k in feat_name for k in ["byte_diff", "bit_flip"]):
        cr = evidence_info.get("payload_change_rate", None)
        if cr is not None:
            if cr < 0.1:
                verdict = "payload 거의 불변 → 동일 데이터 재주입 (Replay)"
            else:
                verdict = "payload 반복 변조 감지 → Spoofing 가능성"
            return f"연속 프레임 payload 변화율 {cr:.1%} — {verdict}"
        return f"실측 {actual:.4f}  정상평균 {normal_mean:.4f} → {direction}"

    else:
        return f"실측 {actual:.4f}  정상평균 {normal_mean:.4f} → {direction}"


# =========================================================
# 포렌식 보고서 — 구간 단위 로컬 분석 + 프레임 역추적
# =========================================================

def run_forensic_report(model, feat_df, X, windows,
                        split_name="Test", max_samples_per_class=5):
    """
    공격으로 분류된 윈도우에 대해 포렌식 보고서 생성.

    기존 글로벌 SHAP 요약에서 한 단계 더 나아가:
      1. 시간 구간 특정 (Timestamp start ~ end)
      2. 정상 구간 대비 피처 이상값 (z-score)
      3. SHAP 상위 피처 → 원본 CAN 프레임 수준 증거 역추적
      4. waterfall plot → 구간 판정 근거 시각화

    Parameters
    ----------
    model                : 학습된 XGBClassifier
    feat_df              : build_feature_dataframe() 결과 DataFrame
    X                    : 피처 DataFrame (FEATURE_COLS 순서)
    windows              : generate_sliding_windows() 결과 list
                           windows[i]["data"] 에 원본 CAN 프레임이 있음
    split_name           : 저장 파일명 구분용
    max_samples_per_class: 공격 유형별 최대 출력 윈도우 수

    Returns
    -------
    all_results : list of dict  (LLM 보고서 생성에 활용)
    """
    FORENSIC_CLASSES = {
        LABEL_NAMES.index("Replay"):   "Replay",
        LABEL_NAMES.index("Spoofing"): "Spoofing",
    }

    print(f"\n{'='*60}")
    print(f"[{split_name}] 포렌식 보고서 생성 (구간 + 프레임 역추적)")
    print(f"{'='*60}")

    # 정상 윈도우 기준값
    normal_mask = feat_df["label"] == 0
    normal_X    = X[normal_mask.values]
    normal_mean = normal_X.mean()
    normal_std  = normal_X.std().replace(0, 1e-9)

    explainer    = shap.TreeExplainer(model)
    summary_rows = []
    all_results  = []

    for class_idx, attack_name in FORENSIC_CLASSES.items():
        attack_mask    = feat_df["label"] == class_idx
        orig_indices   = feat_df.index[attack_mask].tolist()   # windows 인덱스
        attack_feat_df = feat_df[attack_mask].reset_index(drop=True)
        attack_X       = X[attack_mask.values].reset_index(drop=True)
        attack_windows = [windows[i] for i in orig_indices]

        if len(attack_feat_df) == 0:
            print(f"\n  [{attack_name}] 해당 공격 윈도우 없음, 건너뜀")
            continue

        print(f"\n{'-'*60}")
        print(f"  [{attack_name}] 탐지된 공격 구간: {len(attack_feat_df)}개")
        print(f"{'-'*60}")

        shap_exp   = explainer(attack_X)
        _sv        = np.abs(shap_exp.values)
        total_shap = _sv.sum(axis=tuple(range(1, _sv.ndim)))
        top_indices = np.argsort(total_shap)[::-1][:max_samples_per_class]

        class_result = {
            "attack_type":  attack_name,
            "window_count": len(attack_feat_df),
            "windows":      [],
        }

        for rank, idx in enumerate(top_indices, 1):
            t_start  = attack_feat_df.loc[idx, "start"]
            t_end    = attack_feat_df.loc[idx, "end"]
            x_row    = attack_X.iloc[idx]
            win_data = attack_windows[idx]["data"]

            # z-score 이상 피처
            z_scores = (x_row - normal_mean) / normal_std
            top5_z   = z_scores.abs().nlargest(5)

            # SHAP 상위 피처
            sv_raw = shap_exp.values
            if sv_raw.ndim == 3:
                sv_row = sv_raw[idx, :, class_idx]
                bv_single = float(shap_exp.base_values[idx, class_idx]) \
                            if shap_exp.base_values.ndim == 2 \
                            else float(shap_exp.base_values[idx])
            else:
                sv_row = sv_raw[idx]
                bv_single = float(shap_exp.base_values[idx]) \
                            if shap_exp.base_values.ndim > 0 \
                            else float(shap_exp.base_values)

            top5_shap_idx = np.argsort(np.abs(sv_row))[::-1][:5]
            top5_shap = [
                {
                    "feature":     FEATURE_COLS[i],
                    "shap_value":  round(float(sv_row[i]), 6),
                    "direction":   "↑" if sv_row[i] > 0 else "↓",
                    "actual":      round(float(x_row.iloc[i]), 6),
                    "normal_mean": round(float(normal_mean.iloc[i]), 6),
                }
                for i in top5_shap_idx
            ]

            # 프레임 수준 역추적
            top_feat_names = [d["feature"] for d in top5_shap]
            evidence = _trace_evidence(win_data, top_feat_names)

            # 콘솔 출력 — 포렌식 요약 (간결)
            n_frames = len(win_data)
            print(f"\n  [{attack_name}] Window #{rank} | "
                  f"T={t_start:.4f}s~{t_end:.4f}s | {n_frames}frames")
            print(f"  {'-'*56}")
            for i, d in enumerate(top5_shap[:3], 1):
                feat = d["feature"]
                z    = float(z_scores[feat]) if feat in z_scores.index else 0.0
                evi  = evidence.get(feat, {})
                narr = _forensic_narrative(feat, d["actual"], d["normal_mean"], evi)
                print(f"    {i}. [{feat}]  z={z:+.1f}  SHAP={d['shap_value']:+.4f}")
                print(f"       => {narr}")

            # waterfall plot
            exp_single = shap.Explanation(
                values=sv_row,
                base_values=bv_single,
                data=shap_exp.data[idx] if shap_exp.data is not None else None,
                feature_names=shap_exp.feature_names,
            )
            plt.figure(figsize=(10, 7))
            shap.plots.waterfall(exp_single, show=False)
            plt.title(
                f"[{split_name}] {attack_name} Window #{rank}\n"
                f"Timestamp {t_start:.4f}s ~ {t_end:.4f}s",
                fontsize=11
            )
            plt.tight_layout()
            save_path = os.path.join(
                OUTPUT_DIR,
                f"forensic_{split_name.lower()}_{attack_name.lower()}"
                f"_rank{rank:02d}_waterfall.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close("all")

            # LLM용 데이터 수집
            window_result = {
                "rank":           rank,
                "t_start":        round(t_start, 4),
                "t_end":          round(t_end, 4),
                "shap_total":     round(float(total_shap[idx]), 4),
                "shap_top5":      top5_shap,
                "z_top5": [
                    {
                        "feature":     feat_name,
                        "z_score":     round(float(z_val), 4),
                        "actual":      round(float(x_row[feat_name]), 6),
                        "normal_mean": round(float(normal_mean[feat_name]), 6),
                        "direction":   "↑" if x_row[feat_name] > normal_mean[feat_name] else "↓",
                    }
                    for feat_name, z_val in top5_z.items()
                ],
                "frame_evidence": evidence,
            }
            class_result["windows"].append(window_result)

            # CSV 행
            row = {
                "split":      split_name,
                "attack":     attack_name,
                "rank":       rank,
                "t_start":    round(t_start, 4),
                "t_end":      round(t_end, 4),
                "shap_total": round(float(total_shap[idx]), 4),
            }
            for feat_name, z_val in top5_z.items():
                row[f"z_{feat_name}"] = round(float(z_val), 4)
            summary_rows.append(row)

        all_results.append(class_result)

    # CSV 저장
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path   = os.path.join(OUTPUT_DIR,
                                  f"forensic_report_{split_name.lower()}.csv")
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n포렌식 보고서 CSV 저장: {csv_path}")

    print(f"\n[{split_name}] 포렌식 보고서 생성 완료")
    return all_results


# =========================================================
# 디버그용 실행
# =========================================================

if __name__ == "__main__":
    from preprocess import load_and_preprocess
    from feature import generate_sliding_windows, build_feature_dataframe, get_XY
    from train import load_model
    from config import PATHS

    model = load_model()

    print("\n=== Test 데이터 로드 ===")
    df_test   = load_and_preprocess(PATHS["test"], split_name="TEST")
    windows   = generate_sliding_windows(df_test)
    feat_test = build_feature_dataframe(windows)
    X_test, _ = get_XY(feat_test)

    run_shap(model, X_test, split_name="Test")
    run_shap_forensic(model, X_test, split_name="Test")
    results = run_forensic_report(model, feat_test, X_test, windows,
                                  split_name="Test", max_samples_per_class=5)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import LABEL_NAMES, OUTPUT_DIR
from train import get_predictions


# =========================================================
# 성능 평가
# =========================================================

def evaluate(model, X, y, split_name=""):
    """
    classification_report + confusion matrix 출력 및 저장

    Parameters
    ----------
    model      : 학습된 XGBClassifier
    X          : 피처 DataFrame
    y          : 정답 레이블 Series
    split_name : 저장/출력 시 구분 이름 (예: "Test")
    """
    y_pred = get_predictions(model, X)

    # 실제 데이터에 존재하는 클래스만 평가
    present_labels = sorted(y.unique())
    target_names   = [LABEL_NAMES[i] for i in present_labels]

    print(f"\n{'='*60}")
    print(f"[{split_name}] Classification Report")
    print(f"{'='*60}")
    report = classification_report(
        y, y_pred,
        labels=present_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # txt 저장
    report_path = os.path.join(OUTPUT_DIR, f"report_{split_name.lower()}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"[{split_name}] Classification Report\n\n")
        f.write(report)
    print(f"리포트 저장: {report_path}")

    # confusion matrix 저장
    cm = confusion_matrix(y, y_pred, labels=present_labels)
    _plot_confusion_matrix(cm, target_names, split_name)

    return y_pred, report


# =========================================================
# Confusion Matrix 시각화
# =========================================================

def _plot_confusion_matrix(cm, target_names, split_name):
    """Count / Normalized 두 버전을 나란히 출력하고 PNG로 저장."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        [f"[{split_name}] Count", f"[{split_name}] Normalized"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{split_name.lower()}.png")
    plt.savefig(save_path, dpi=150)
    plt.close("all")
    print(f"Confusion matrix 저장: {save_path}")


# =========================================================
# 요약 테이블
# =========================================================

def print_summary(results):
    """
    여러 split 결과를 클래스별 F1 + Macro/Weighted F1로 요약.

    Parameters
    ----------
    results : dict  {"Val": (y_true, y_pred), "Test": (y_true, y_pred)}
    """
    rows = []
    for split_name, (y_true, y_pred) in results.items():
        present_labels = sorted(np.unique(y_true))
        target_names   = [LABEL_NAMES[i] for i in present_labels]
        rd = classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            digits=4,
            zero_division=0,
        )
        row = {"Split": split_name}
        for cls in target_names:
            if cls in rd:
                row[f"{cls}_F1"] = round(rd[cls]["f1-score"], 4)
        row["Macro_F1"]    = round(rd["macro avg"]["f1-score"], 4)
        row["Weighted_F1"] = round(rd["weighted avg"]["f1-score"], 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Split")
    print(f"\n{'='*60}")
    print("요약 테이블")
    print(f"{'='*60}")
    print(summary_df.to_string())

    save_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(save_path, encoding="utf-8-sig")
    print(f"\n요약 테이블 저장: {save_path}")
"""
Spoofing vs Flooding 피처 분포 비교 진단
Train / Test 각각에서 두 클래스의 핵심 피처 통계 출력
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import PATHS, LABEL_NAMES
from preprocess import load_and_preprocess
from feature import generate_sliding_windows, build_feature_dataframe, get_XY

COMPARE_COLS = [
    "msg_count", "msg_per_id", "iat_mean", "top1_id_ratio",
    "unique_id_count", "byte0_std", "byte1_std", "payload_entropy",
]


def compare(feat_df, split_name):
    print(f"\n{'='*60}")
    print(f"[{split_name}] Spoofing vs Flooding 피처 비교")
    print(f"{'='*60}")

    flooding_df = feat_df[feat_df["label"] == 1]  # Flooding
    spoofing_df = feat_df[feat_df["label"] == 4]  # Spoofing

    if flooding_df.empty:
        print("  Flooding 윈도우 없음")
        return
    if spoofing_df.empty:
        print("  Spoofing 윈도우 없음")
        return

    print(f"\n  샘플 수: Flooding={len(flooding_df)}, Spoofing={len(spoofing_df)}")
    print(f"\n  {'피처':<25} {'Flooding mean':>14} {'Spoofing mean':>14} {'차이배율':>10}")
    print(f"  {'-'*65}")
    for col in COMPARE_COLS:
        if col not in feat_df.columns:
            continue
        f_mean = flooding_df[col].mean()
        s_mean = spoofing_df[col].mean()
        ratio  = s_mean / f_mean if f_mean != 0 else float("inf")
        flag   = "  ← 거의 같음!" if abs(ratio - 1) < 0.3 else ""
        print(f"  {col:<25} {f_mean:>14.3f} {s_mean:>14.3f} {ratio:>10.2f}x{flag}")


def main():
    print("Train 로드 중...")
    df_train    = load_and_preprocess(PATHS["train"], split_name="TRAIN")
    windows_tr  = generate_sliding_windows(df_train)
    feat_train  = build_feature_dataframe(windows_tr)
    compare(feat_train, "Train")

    print("\nTest 로드 중...")
    df_test     = load_and_preprocess(PATHS["test"], split_name="TEST")
    windows_te  = generate_sliding_windows(df_test)
    feat_test   = build_feature_dataframe(windows_te)
    compare(feat_test, "Test")


if __name__ == "__main__":
    main()

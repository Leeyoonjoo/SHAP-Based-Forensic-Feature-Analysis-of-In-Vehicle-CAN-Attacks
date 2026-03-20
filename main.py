import os
import time
import pandas as pd

from config import PATHS, OUTPUT_DIR, LABEL_NAMES
from preprocess import load_and_preprocess
from feature import generate_sliding_windows, build_feature_dataframe, get_XY
from train import train_xgboost, save_model
from evaluate import evaluate, print_summary
from shap_analysis import run_shap, run_shap_forensic, run_forensic_report
from llm_report import generate_llm_report


# =========================================================
# 전체 파이프라인 (Train + Val + Test)
# early stopping은 Val 데이터로 처리
# =========================================================

def main():
    total_start = time.time()

    # ---------------------------------------------------------
    # STEP 1. 데이터 로드
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1. 데이터 로드")
    print("="*60)

    df_train = load_and_preprocess(PATHS["train"], split_name="TRAIN")
    df_val   = load_and_preprocess(PATHS["val"],   split_name="VAL")
    df_test  = load_and_preprocess(PATHS["test"],  split_name="TEST")

    # ---------------------------------------------------------
    # STEP 2. 슬라이딩 윈도우 생성
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2. 슬라이딩 윈도우 생성")
    print("="*60)

    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        t0 = time.time()
        print(f"\n[{name}] 윈도우 생성 중...")
        if name == "Train":
            windows_train = generate_sliding_windows(df)
            print(f"  완료: {len(windows_train):,}개  ({time.time()-t0:.1f}초)")
        elif name == "Val":
            windows_val = generate_sliding_windows(df)
            print(f"  완료: {len(windows_val):,}개  ({time.time()-t0:.1f}초)")
        else:
            windows_test = generate_sliding_windows(df)
            print(f"  완료: {len(windows_test):,}개  ({time.time()-t0:.1f}초)")

    # ---------------------------------------------------------
    # STEP 3. 피처 추출
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3. 피처 추출 (26개)")
    print("="*60)

    for name, windows in [("Train", windows_train), ("Val", windows_val), ("Test", windows_test)]:
        t0 = time.time()
        print(f"\n[{name}] 피처 추출 중...")
        if name == "Train":
            feat_train = build_feature_dataframe(windows)
            X_train, y_train = get_XY(feat_train)
            print(f"  완료: {feat_train.shape}  ({time.time()-t0:.1f}초)")
        elif name == "Val":
            feat_val = build_feature_dataframe(windows)
            X_val, y_val = get_XY(feat_val)
            print(f"  완료: {feat_val.shape}  ({time.time()-t0:.1f}초)")
        else:
            feat_test = build_feature_dataframe(windows)
            X_test, y_test = get_XY(feat_test)
            print(f"  완료: {feat_test.shape}  ({time.time()-t0:.1f}초)")

    # ---------------------------------------------------------
    # STEP 4. 윈도우 단위 클래스 분포 출력
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4. 윈도우 단위 클래스 분포")
    print("="*60)

    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = pd.Series(y).value_counts().sort_index()
        print(f"\n[{name}]")
        for idx, cnt in dist.items():
            label = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else str(idx)
            ratio = cnt / len(y) * 100
            print(f"  {label:<12} {cnt:>8,}  ({ratio:5.1f}%)")

    # ---------------------------------------------------------
    # STEP 5. XGBoost 학습
    # Val 데이터를 early stopping eval_set으로 사용
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5. XGBoost 학습 (Val early stopping)")
    print("="*60)

    t0    = time.time()
    model = train_xgboost(X_train, y_train, X_val, y_val)
    print(f"\n학습 시간: {time.time()-t0:.1f}초")
    save_model(model)

    # ---------------------------------------------------------
    # STEP 6. 평가 (Val + Test)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6. 평가")
    print("="*60)

    y_pred_val,  _ = evaluate(model, X_val,  y_val,  split_name="Val")
    y_pred_test, _ = evaluate(model, X_test, y_test, split_name="Test")

    print_summary({
        "Val":  (y_val,  y_pred_val),
        "Test": (y_test, y_pred_test),
    })

    # ---------------------------------------------------------
    # STEP 7. SHAP 분석 (Test 세트 기준)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 7. SHAP 분석 (Test 세트 기준)")
    print("="*60)

    run_shap(model, X_test, split_name="Test")
    run_shap_forensic(model, X_test, split_name="Test")

    forensic_results = run_forensic_report(
        model, feat_test, X_test, windows_test,
        split_name="Test", max_samples_per_class=3
    )

    # ---------------------------------------------------------
    # STEP 8. LLM 포렌식 보고서 자동 생성
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 8. LLM 포렌식 보고서 자동 생성")
    print("="*60)

    generate_llm_report(forensic_results, split_name="Test")

    # ---------------------------------------------------------
    # 완료
    # ---------------------------------------------------------
    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"전체 파이프라인 완료  (총 {elapsed/60:.1f}분)")
    print(f"결과 저장 경로: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

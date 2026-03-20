import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (XGB_PARAMS, OUTPUT_DIR, LABEL_NAMES,
                    SPOOFING_WEIGHT_MULTIPLIER, SPOOFING_THRESHOLD)


# =========================================================
# XGBoost 학습
# =========================================================

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    XGBoost 5클래스 분류 학습

    Val 데이터가 주어지면 eval_set으로 사용 (early stopping)
    없으면 Train 내부 stratified 20%를 early stopping용으로 분리

    클래스 불균형 처리
    ------------------
    compute_sample_weight("balanced") 로 기본 가중치 부여 후
    Spoofing 클래스에 SPOOFING_WEIGHT_MULTIPLIER 배 추가 가중치
    """
    spoofing_idx = LABEL_NAMES.index("Spoofing")

    if X_val is not None and y_val is not None:
        X_tr, y_tr = X_train, y_train
        X_es, y_es = X_val,   y_val
        print(f"\n  외부 Val 사용: {len(X_es):,}개")
    else:
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42,
        )
        print(f"\n  내부 Early-stopping val: {len(X_es):,}개 (Train 20% stratified)")

    # 클래스 불균형 보정
    sample_weight = compute_sample_weight("balanced", y_tr)
    sample_weight[y_tr == spoofing_idx] *= SPOOFING_WEIGHT_MULTIPLIER

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_es, y_es)],
        verbose=50,
    )

    print(f"\nXGBoost 학습 완료  (best iteration: {model.best_iteration})")
    print(f"  학습 샘플 수: {len(X_train):,}")
    print(f"  피처 수:      {X_train.shape[1]}")
    _print_class_dist("Train", y_train)

    return model


# =========================================================
# 유틸
# =========================================================

def _print_class_dist(split_name, y):
    import pandas as pd
    dist = pd.Series(y).value_counts().sort_index()
    print(f"\n  [{split_name}] 클래스 분포:")
    for idx, cnt in dist.items():
        name = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else str(idx)
        print(f"    {name:<12} {cnt:>8,}")


def get_predictions(model, X, threshold=SPOOFING_THRESHOLD):
    """
    예측값 반환 (정수 클래스)

    Spoofing 확률이 threshold 이상이면 argmax 결과와 무관하게 Spoofing으로 분류
    → Spoofing recall 개선
    """
    spoofing_idx = LABEL_NAMES.index("Spoofing")
    proba        = model.predict_proba(X)
    y_pred       = np.argmax(proba, axis=1)

    spoofing_mask          = proba[:, spoofing_idx] >= threshold
    y_pred[spoofing_mask]  = spoofing_idx

    return y_pred


# =========================================================
# 모델 저장 / 로드
# =========================================================

def save_model(model, filename="xgb_model.pkl"):
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(model, path)
    print(f"\n모델 저장: {path}")
    return path


def load_model(filename="xgb_model.pkl"):
    path = os.path.join(OUTPUT_DIR, filename)
    model = joblib.load(path)
    print(f"\n모델 로드: {path}")
    return model


# =========================================================
# 디버그용 실행
# =========================================================

if __name__ == "__main__":
    from preprocess import load_and_preprocess
    from feature import generate_sliding_windows, build_feature_dataframe, get_XY
    from config import PATHS

    df_train      = load_and_preprocess(PATHS["train"], split_name="TRAIN")
    windows_train = generate_sliding_windows(df_train)
    feat_train    = build_feature_dataframe(windows_train)
    X_train, y_train = get_XY(feat_train)

    df_val      = load_and_preprocess(PATHS["val"], split_name="VAL")
    windows_val = generate_sliding_windows(df_val)
    feat_val    = build_feature_dataframe(windows_val)
    X_val, y_val = get_XY(feat_val)

    model = train_xgboost(X_train, y_train, X_val, y_val)
    save_model(model)
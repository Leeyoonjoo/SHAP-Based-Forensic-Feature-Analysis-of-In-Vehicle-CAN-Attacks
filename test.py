import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap


# =========================================================
# 1. 기본 설정
# =========================================================

LABEL_MAP = {
    "Normal": 0,
    "Replay": 1,
    "Spoofing": 2
}

LABEL_NAMES = ["Normal", "Replay", "Spoofing"]


# =========================================================
# 2. 전처리 함수
# =========================================================

def normalize_arbitration_id(x):
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    try:
        return int(s, 16)
    except:
        return -1


def parse_data_bytes(data_str):
    if pd.isna(data_str):
        return []
    s = str(data_str).strip()
    parts = s.split()
    try:
        return [int(p, 16) for p in parts]
    except:
        return []


def load_dataset(file_paths, classes=("Normal", "Replay", "Spoofing")):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        if "SubClass" not in df.columns:
            df["SubClass"] = "Normal"
        if "Class" not in df.columns:
            df["Class"] = "Normal"
        df = df[df["SubClass"].isin(classes)].copy()
        df["source_file"] = path
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("Timestamp").reset_index(drop=True)
    data["Arbitration_ID"] = data["Arbitration_ID"].apply(normalize_arbitration_id)
    data["payload_bytes"] = data["Data"].apply(parse_data_bytes)
    data["y_msg"] = data["SubClass"].map(LABEL_MAP)
    return data


# =========================================================
# 3. 윈도우 생성 (속도 최적화)
# =========================================================

def generate_sliding_windows(df, window_size=0.2, stride=0.1, purity_threshold=0.5):
    """
    시간 슬라이딩 윈도우 생성
    인덱스 기반 탐색으로 속도 최적화
    """
    windows = []
    timestamps = df["Timestamp"].values
    n = len(timestamps)
    start_time = timestamps[0]
    end_time = timestamps[-1]
    t = start_time

    left = 0  # 윈도우 시작 인덱스 추적

    while t + window_size <= end_time:
        # 왼쪽 경계: t 이상인 첫 인덱스
        while left < n and timestamps[left] < t:
            left += 1

        # 오른쪽 경계: t + window_size 미만인 마지막 인덱스
        right = left
        while right < n and timestamps[right] < t + window_size:
            right += 1

        if right == left:
            t += stride
            continue

        w = df.iloc[left:right].copy()

        class_ratio = w["y_msg"].value_counts(normalize=True)
        label = int(class_ratio.idxmax())
        purity = float(class_ratio.max())

        if purity >= purity_threshold:
            windows.append({
                "start": t,
                "end": t + window_size,
                "label": label,
                "purity": purity,
                "data": w
            })

        t += stride

    return windows


# =========================================================
# 4. Feature Engineering
# =========================================================

def shannon_entropy(values):
    if len(values) == 0:
        return 0.0
    counts = Counter(values)
    probs = np.array(list(counts.values()), dtype=float) / len(values)
    return -np.sum(probs * np.log2(probs + 1e-12))


def extract_features(window_df, prev_df=None):
    feats = {}

    # 기본 통계
    feats["msg_count"] = len(window_df)
    feats["unique_id_count"] = window_df["Arbitration_ID"].nunique()
    feats["mean_dlc"] = window_df["DLC"].mean()
    feats["std_dlc"] = window_df["DLC"].std() if len(window_df) > 1 else 0.0

    # IAT (Inter-Arrival Time)
    # 공격 메시지 삽입 시 IAT가 비정상적으로 짧아짐
    ts = window_df["Timestamp"].values
    if len(ts) > 1:
        iat = np.diff(ts)
        feats["iat_mean"] = np.mean(iat)
        feats["iat_std"] = np.std(iat)
        feats["iat_min"] = np.min(iat)
        feats["iat_max"] = np.max(iat)
        feats["burstiness"] = (
            (np.std(iat) - np.mean(iat)) /
            (np.std(iat) + np.mean(iat) + 1e-12)
        )
    else:
        feats["iat_mean"] = 0.0
        feats["iat_std"] = 0.0
        feats["iat_min"] = 0.0
        feats["iat_max"] = 0.0
        feats["burstiness"] = 0.0

    # CAN ID 분포
    # ID 번호 자체가 아닌 패턴의 이상 탐지
    id_counts = window_df["Arbitration_ID"].value_counts()
    feats["top1_id_ratio"] = id_counts.iloc[0] / len(window_df)
    feats["top3_id_ratio_sum"] = id_counts.iloc[:3].sum() / len(window_df)
    feats["id_entropy"] = shannon_entropy(window_df["Arbitration_ID"].tolist())

    # ID + Data 반복 (Replay 핵심)
    # Replay: 동일한 (ID, Data) 쌍이 반복됨
    id_data = (
        window_df["Arbitration_ID"].astype(str) + "_" +
        window_df["Data"].astype(str)
    )
    id_data_counts = id_data.value_counts()
    feats["unique_id_data_count"] = id_data.nunique()
    feats["top1_id_data_ratio"] = id_data_counts.iloc[0] / len(window_df)
    feats["repeat_id_data_ratio"] = (
        id_data_counts[id_data_counts >= 2].sum() / len(window_df)
    )

    # Payload 분포
    payload_list = window_df["Data"].astype(str).tolist()
    feats["payload_entropy"] = shannon_entropy(payload_list)

    # 동일 payload 연속 길이 (Replay 핵심)
    max_run = 1
    cur_run = 1
    for i in range(1, len(payload_list)):
        if payload_list[i] == payload_list[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    feats["max_same_payload_run"] = max_run if len(payload_list) > 0 else 0

    # 바이트 변화량
    payloads = window_df["payload_bytes"].tolist()
    byte_diffs = []
    for i in range(1, len(payloads)):
        a = payloads[i - 1]
        b = payloads[i]
        L = min(len(a), len(b))
        if L > 0:
            diff = np.mean([abs(a[j] - b[j]) for j in range(L)])
            byte_diffs.append(diff)
    feats["byte_diff_mean"] = np.mean(byte_diffs) if byte_diffs else 0.0
    feats["byte_diff_std"] = np.std(byte_diffs) if byte_diffs else 0.0

    # DATA_0~7 각 바이트별 평균, 분산 (Spoofing 핵심)
    # Spoofing 시 특정 바이트 값이 정상 범위에서 이탈
    # 예: 정상 C0(192) -> 공격 C1(193)
    for i in range(8):
        col_vals = [b[i] for b in payloads if len(b) > i]
        feats[f"byte{i}_mean"] = np.mean(col_vals) if col_vals else 0.0
        feats[f"byte{i}_std"] = np.std(col_vals) if col_vals else 0.0

    # 직전 윈도우와의 유사도 (Replay 핵심)
    # Replay는 이전 구간과 동일한 메시지가 반복됨
    if prev_df is not None and len(prev_df) > 0:
        prev_set = set(
            (prev_df["Arbitration_ID"].astype(str) + "_" +
             prev_df["Data"].astype(str)).tolist()
        )
        curr_set = set(id_data.tolist())
        union = len(prev_set | curr_set)
        inter = len(prev_set & curr_set)
        feats["prev_window_jaccard"] = inter / union if union > 0 else 0.0
    else:
        feats["prev_window_jaccard"] = 0.0

    return feats


def build_feature_dataframe(windows):
    rows = []
    prev_df = None
    for w in windows:
        feats = extract_features(w["data"], prev_df=prev_df)
        feats["label"] = w["label"]
        feats["purity"] = w["purity"]
        feats["start"] = w["start"]
        feats["end"] = w["end"]
        rows.append(feats)
        prev_df = w["data"]
    return pd.DataFrame(rows)


# =========================================================
# 5. 데이터 분할 (시간순)
# =========================================================

def time_based_split(feature_df, train_ratio=0.7):
    """
    시간순 split: 앞 70% 학습, 뒤 30% 테스트
    학습/테스트 구간이 절대 섞이지 않음
    """
    feature_df = feature_df.sort_values("start").reset_index(drop=True)
    split_idx = int(len(feature_df) * train_ratio)

    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=["label", "purity", "start", "end"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label", "purity", "start", "end"])
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test, train_df, test_df


# =========================================================
# 6. 모델 학습
# =========================================================

def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            multi_class="multinomial"
        ))
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# =========================================================
# 7. 평가
# =========================================================

def get_predictions(model, X_test):
    """
    LR과 XGBoost 모두 정수 클래스 예측값 반환
    XGBoost softprob는 확률 배열을 반환하므로 argmax 처리
    """
    y_pred = model.predict(X_test)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = get_predictions(model, X_test)
    print(f"\n===== {name} =====")
    print(classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        digits=4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred


# =========================================================
# 8. SHAP 분석 (XGBoost에만 적용)
# =========================================================

def run_shap_multiclass(xgb_model, X_test, title_prefix=""):
    """
    공격 유형별 SHAP summary plot 출력
    shap 버전에 따른 반환 형태 차이 처리
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # shap 버전에 따라 list 또는 ndarray로 반환됨
    if isinstance(shap_values, list):
        # 구버전: list of arrays (클래스 수만큼)
        shap_per_class = shap_values

    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # 신버전: (n_samples, n_features, n_classes)
            shap_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        elif shap_values.ndim == 2:
            # 이진 분류 형태로 나오는 경우 (예외 처리)
            print("경고: SHAP 값이 2D 배열입니다. 클래스별 분리 불가.")
            shap_per_class = [shap_values]
        else:
            print("경고: 예상치 못한 SHAP 값 형태입니다.")
            return explainer, shap_values
    else:
        print("경고: 예상치 못한 SHAP 값 타입입니다.")
        return explainer, shap_values

    # 클래스별 plot 출력
    for class_idx, class_name in enumerate(LABEL_NAMES):
        if class_idx >= len(shap_per_class):
            break

        print(f"\n[SHAP Summary] {title_prefix} - {class_name}")

        # Beeswarm plot: 피처값과 기여도 동시 표시
        shap.summary_plot(shap_per_class[class_idx], X_test, show=False)
        plt.title(f"{title_prefix} - {class_name} (Beeswarm)")
        plt.tight_layout()
        plt.savefig(f"shap_{title_prefix}_{class_name}_beeswarm.png", dpi=150)
        plt.show()

        # Bar plot: 피처 중요도 순위
        shap.summary_plot(
            shap_per_class[class_idx], X_test,
            plot_type="bar", show=False
        )
        plt.title(f"{title_prefix} - {class_name} (Bar)")
        plt.tight_layout()
        plt.savefig(f"shap_{title_prefix}_{class_name}_bar.png", dpi=150)
        plt.show()

    return explainer, shap_values


# =========================================================
# 9. 실행 함수
# =========================================================

def run_experiment(
    file_paths,
    title="experiment",
    window_size=0.2,
    stride=0.1,
    purity_threshold=0.5,
    train_ratio=0.7
):
    print(f"\n{'='*60}")
    print(f"실험: {title}")
    print(f"{'='*60}")

    print("\n1) 데이터 로드")
    df = load_dataset(file_paths)
    print(f"전체 메시지 수: {len(df)}")
    print("\n메시지 단위 클래스 분포:")
    print(df["SubClass"].value_counts())

    print("\n2) 슬라이딩 윈도우 생성")
    windows = generate_sliding_windows(
        df,
        window_size=window_size,
        stride=stride,
        purity_threshold=purity_threshold
    )
    print(f"생성된 윈도우 수: {len(windows)}")

    print("\n3) 피처 테이블 생성")
    feature_df = build_feature_dataframe(windows)
    n_features = len(feature_df.columns) - 4  # label, purity, start, end 제외
    print(f"피처 수: {n_features}")
    print("\n윈도우 단위 레이블 분포:")
    label_dist = feature_df["label"].value_counts().sort_index()
    for idx, cnt in label_dist.items():
        print(f"  {LABEL_NAMES[idx]}: {cnt}")

    print("\n4) 시간순 분할 (앞 70% 학습 / 뒤 30% 테스트)")
    X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(
        feature_df, train_ratio=train_ratio
    )
    print(f"학습 세트: {X_train.shape}")
    print(f"테스트 세트: {X_test.shape}")

    print("\n5) 모델 학습")
    lr_model = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    print("\n6) 평가")
    evaluate_model(lr_model, X_test, y_test, name=f"Logistic Regression ({title})")
    evaluate_model(xgb_model, X_test, y_test, name=f"XGBoost ({title})")

    print("\n7) SHAP 분석 (XGBoost)")
    explainer, shap_values = run_shap_multiclass(
        xgb_model, X_test, title_prefix=title
    )

    return {
        "raw_df": df,
        "feature_df": feature_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "explainer": explainer,
        "shap_values": shap_values
    }


# =========================================================
# 10. main
# =========================================================

if __name__ == "__main__":

    # --- Stationary 실험 ---
    stationary_files = [
        "./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_S_0.csv",
        "./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_S_1.csv",
        "./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_S_2.csv"
    ]

    results_s = run_experiment(
        file_paths=stationary_files,
        title="Stationary",
        window_size=0.2,
        stride=0.1,
        purity_threshold=0.5,
        train_ratio=0.7
    )

    # --- Driving 실험 ---
    driving_files = [
        "./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_0.csv",
        "./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv",
        "./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_2.csv"
    ]

    results_d = run_experiment(
        file_paths=driving_files,
        title="Driving",
        window_size=0.2,
        stride=0.1,
        purity_threshold=0.5,
        train_ratio=0.7
    )

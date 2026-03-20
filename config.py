import os

# =========================================================
# 기본 경로 (src/ 기준 실행)
# =========================================================

BASE = os.path.join(os.path.dirname(__file__), "..")

# =========================================================
# 데이터 분할 경로
#
# Train : Pre_train_S_0~2 + Pre_train_D_0~2
#         (S: 시뮬레이션 세션, D: 실제 주행 세션)
#         D_0은 SubClass 컬럼 없음 → preprocess에서 전부 Normal 처리
# Val   : Pre_submit_S + Pre_submit_D  (early stopping용)
# Test  : Fin_host_S 만  (S 세션 독립 평가)
#
# → S/D를 함께 학습하여 주행 패턴 다양성 확보
# → Test는 S 세션으로 고정하여 독립 평가 유지
# =========================================================

PATHS = {
    "train": [
        os.path.join(BASE, "0_Preliminary", "0_Training", "Pre_train_S_0.csv"),
        os.path.join(BASE, "0_Preliminary", "0_Training", "Pre_train_S_1.csv"),
        os.path.join(BASE, "0_Preliminary", "0_Training", "Pre_train_S_2.csv"),
        os.path.join(BASE, "0_Preliminary", "0_Training", "Pre_train_D_0.csv"),
        os.path.join(BASE, "0_Preliminary", "0_Training", "Pre_train_D_1.csv"),
        os.path.join(BASE, "0_Preliminary", "0_Training", "Pre_train_D_2.csv"),
    ],
    "val": [
        os.path.join(BASE, "0_Preliminary", "1_Submission", "Pre_submit_S.csv"),
        os.path.join(BASE, "0_Preliminary", "1_Submission", "Pre_submit_D.csv"),
    ],
    "test": [
        os.path.join(BASE, "1_Final", "Fin_host_S.csv"),
    ],
}

# =========================================================
# 라벨 정의
#
# Flooding / Fuzzing은 LABEL_MAP에 포함하지 않는다.
# preprocess.py에서 map 실패 행을 자동 제거하므로
# 별도 필터링 코드 없이 Flooding/Fuzzing이 걸러진다.
# =========================================================

LABEL_MAP = {
    "Normal":   0,
    "Replay":   1,
    "Spoofing": 2,
}

LABEL_NAMES = ["Normal", "Replay", "Spoofing"]
NUM_CLASSES  = len(LABEL_NAMES)  # 3

# =========================================================
# 슬라이딩 윈도우 파라미터
# =========================================================

WINDOW_SIZE      = 0.2   # 초 단위 윈도우 크기
STRIDE           = 0.1   # 초 단위 슬라이딩 간격
SESSION_GAP      = 1.0   # 이 값 이상의 타임스탬프 간격은 세션 경계로 처리
ATTACK_THRESHOLD = 0.01  # 윈도우 내 공격 메시지 비율이 이 이상이면 공격 라벨 부여

# =========================================================
# XGBoost 하이퍼파라미터
# =========================================================

XGB_PARAMS = {
    "n_estimators":      1000,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "objective":         "multi:softprob",
    "num_class":         NUM_CLASSES,
    "eval_metric":       "mlogloss",
    "early_stopping_rounds": 30,
    "random_state":      42,
}

# Spoofing 클래스 추가 가중치 (클래스 불균형 보정)
SPOOFING_WEIGHT_MULTIPLIER = 3.0

# Spoofing 확률이 이 값 이상이면 argmax 결과와 무관하게 Spoofing으로 분류
SPOOFING_THRESHOLD = 0.35

# =========================================================
# 출력 경로
# =========================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
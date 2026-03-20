import numpy as np
import pandas as pd
from collections import Counter

from config import WINDOW_SIZE, STRIDE, SESSION_GAP, ATTACK_THRESHOLD, LABEL_NAMES


# =========================================================
# 유틸
# =========================================================

def shannon_entropy(values):
    """Shannon Entropy 계산. 값이 없으면 0 반환."""
    if len(values) == 0:
        return 0.0
    counts = Counter(values)
    probs  = np.array(list(counts.values()), dtype=float) / len(values)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# =========================================================
# 슬라이딩 윈도우 생성
# =========================================================

def generate_sliding_windows(df,
                              window_size=WINDOW_SIZE,
                              stride=STRIDE,
                              session_gap=SESSION_GAP,
                              attack_threshold=ATTACK_THRESHOLD):
    """
    시간 기반 슬라이딩 윈도우 생성

    라벨링 규칙
    -----------
    - Replay 비율 >= attack_threshold AND Replay >= Spoofing → Replay (1)
    - Spoofing 비율 >= attack_threshold                      → Spoofing (2)
    - 그 외                                                  → Normal (0)

    최적화
    ------
    np.searchsorted 로 경계 인덱스를 벡터 단위로 계산 (O(N log N))
    세션 gap > session_gap 이면 별개 세션으로 분리하여
    세션 경계를 넘는 윈도우 생성을 방지한다.
    """
    timestamps = df["Timestamp"].values
    y_msg      = df["y_msg"].values
    n          = len(timestamps)

    # 세션 경계 분리
    gaps   = np.diff(timestamps)
    breaks = np.where(gaps > session_gap)[0]

    seg_starts = np.concatenate([[0],      breaks + 1])
    seg_ends   = np.concatenate([breaks,   [n - 1]])

    t_start_list = []
    for s0, s1 in zip(seg_starts, seg_ends):
        t0 = timestamps[s0]
        t1 = timestamps[s1]
        if t1 - t0 >= window_size:
            t_start_list.append(
                np.arange(t0, t1 - window_size + 1e-9, stride)
            )

    if not t_start_list:
        return []

    t_starts = np.concatenate(t_start_list)

    lefts  = np.searchsorted(timestamps, t_starts,               side="left")
    rights = np.searchsorted(timestamps, t_starts + window_size, side="left")

    valid    = rights > lefts
    t_starts = t_starts[valid]
    lefts    = lefts[valid]
    rights   = rights[valid]

    windows = []
    for t, left, right in zip(t_starts, lefts, rights):
        total = right - left
        y_w   = y_msg[left:right]

        replay_cnt   = int((y_w == 1).sum())
        spoofing_cnt = int((y_w == 2).sum())
        replay_ratio   = replay_cnt   / total
        spoofing_ratio = spoofing_cnt / total

        if replay_ratio >= attack_threshold and replay_cnt >= spoofing_cnt:
            label  = 1
            purity = replay_ratio
        elif spoofing_ratio >= attack_threshold:
            label  = 2
            purity = spoofing_ratio
        else:
            label  = 0
            purity = 1.0 - replay_ratio - spoofing_ratio

        windows.append({
            "start":  t,
            "end":    t + window_size,
            "label":  label,
            "purity": float(purity),
            "data":   df.iloc[left:right],
        })

    return windows


# =========================================================
# 단일 윈도우 피처 추출 (26개)
# =========================================================

def extract_features(window_df, prev_df=None):
    """
    단일 윈도우에서 피처 26개 추출

    피처 구성
    ---------
    1. 기본 통계 (4)
       msg_count, unique_id_count, mean_dlc, std_dlc

    2. 시간 기반 IAT (5)
       iat_mean, iat_std, iat_min, iat_max, burstiness

    3. Arbitration ID 분포 (3)
       top1_id_ratio, top3_id_ratio_sum, id_entropy

    4. ID+Data 반복 / payload 다양성 (6)
       unique_id_data_count, top1_id_data_ratio, repeat_id_data_ratio
       payload_entropy, payload_byte_entropy, max_same_payload_run

    5. payload 변화량 (3)
       byte_diff_mean, byte_diff_std, bit_flip_mean

    6. ID-level payload summary (4)
       id_payload_diff_mean_mean, id_payload_diff_std_mean
       id_bitflip_mean_mean, changed_id_ratio

    7. 이전 윈도우 유사도 (1)
       prev_window_jaccard

    총 26개
    """
    feats       = {}
    payloads    = window_df["payload_bytes"].tolist()
    payload_str = window_df["Data"].astype(str).tolist()
    n_msg       = len(window_df)

    # ---------------------------------------------------------
    # 1. 기본 통계
    # ---------------------------------------------------------
    feats["msg_count"]       = n_msg
    feats["unique_id_count"] = window_df["Arbitration_ID"].nunique()
    feats["mean_dlc"]        = window_df["DLC"].mean()
    feats["std_dlc"]         = window_df["DLC"].std() if n_msg > 1 else 0.0

    # ---------------------------------------------------------
    # 2. IAT (Inter-Arrival Time)
    # Replay   : 동일 주기 반복 → IAT 분산 낮음
    # Spoofing : 삽입 타이밍 어긋나면 burstiness 증가
    # ---------------------------------------------------------
    ts = window_df["Timestamp"].values
    if len(ts) > 1:
        iat      = np.diff(ts)
        iat_mean = float(np.mean(iat))
        iat_std  = float(np.std(iat))
        feats["iat_mean"]   = iat_mean
        feats["iat_std"]    = iat_std
        feats["iat_min"]    = float(np.min(iat))
        feats["iat_max"]    = float(np.max(iat))
        feats["burstiness"] = (
            (iat_std - iat_mean) / (iat_std + iat_mean + 1e-12)
        )
    else:
        for k in ["iat_mean", "iat_std", "iat_min", "iat_max", "burstiness"]:
            feats[k] = 0.0

    # ---------------------------------------------------------
    # 3. Arbitration ID 분포
    # Replay : 특정 ID 집중 → top1_id_ratio 높음
    # ---------------------------------------------------------
    id_counts = window_df["Arbitration_ID"].value_counts()
    feats["top1_id_ratio"]     = float(id_counts.iloc[0] / n_msg)
    feats["top3_id_ratio_sum"] = float(id_counts.iloc[:3].sum() / n_msg)
    feats["id_entropy"]        = shannon_entropy(window_df["Arbitration_ID"].tolist())

    # ---------------------------------------------------------
    # 4. ID+Data 반복 / payload 다양성
    # Replay           : 동일 (ID, Data) 반복 → repeat_id_data_ratio 높음
    # payload_byte_entropy : 바이트 값 분포 entropy
    #   Fuzzing이면 바이트 값이 매우 다양 → 높음
    #   Replay이면 동일 바이트 반복 → 낮음
    # ---------------------------------------------------------
    id_data = (
        window_df["Arbitration_ID"].astype(str) + "_" +
        window_df["Data"].astype(str)
    )
    id_data_counts = id_data.value_counts()

    feats["unique_id_data_count"] = int(id_data.nunique())
    feats["top1_id_data_ratio"]   = float(id_data_counts.iloc[0] / n_msg)
    feats["repeat_id_data_ratio"] = float(
        id_data_counts[id_data_counts >= 2].sum() / n_msg
    )
    feats["payload_entropy"]      = shannon_entropy(payload_str)

    all_bytes = [b for p in payloads for b in p]
    feats["payload_byte_entropy"] = shannon_entropy(all_bytes)

    # 동일 payload 최장 연속 길이
    # Replay : 같은 패킷 연달아 밀어 넣으면 높음
    max_run = cur_run = 1
    for i in range(1, len(payload_str)):
        if payload_str[i] == payload_str[i - 1]:
            cur_run += 1
            max_run  = max(max_run, cur_run)
        else:
            cur_run = 1
    feats["max_same_payload_run"] = max_run if payload_str else 0

    # ---------------------------------------------------------
    # 5. payload 변화량 (numpy 벡터화)
    # byte_diff_mean : Replay면 ≈ 0, Spoofing이면 특정 값 조작으로 증가
    # bit_flip_mean  : 바이트 단위가 아닌 비트 단위 변화
    #                  Spoofing 1비트 조작(C0→C1)을 더 정밀하게 포착
    # ---------------------------------------------------------
    if n_msg > 1:
        max_len = max((len(p) for p in payloads), default=0)
    else:
        max_len = 0

    if max_len > 0 and n_msg > 1:
        arr = np.zeros((n_msg, max_len), dtype=np.uint8)
        for i, p in enumerate(payloads):
            if p:
                arr[i, :len(p)] = p

        diff      = np.abs(arr[1:].astype(np.int16) - arr[:-1].astype(np.int16))
        pair_mean = diff.mean(axis=1)
        feats["byte_diff_mean"] = float(pair_mean.mean())
        feats["byte_diff_std"]  = float(pair_mean.std())

        xor  = arr[1:] ^ arr[:-1]
        bits = np.unpackbits(xor.ravel()).reshape(n_msg - 1, -1).sum(axis=1)
        feats["bit_flip_mean"] = float(bits.mean() / max_len)
    else:
        feats["byte_diff_mean"] = 0.0
        feats["byte_diff_std"]  = 0.0
        feats["bit_flip_mean"]  = 0.0

    # ---------------------------------------------------------
    # 6. ID-level payload summary
    # 같은 Arbitration_ID 그룹 내부에서 payload 변화를 요약
    # Spoofing : 특정 ID 내부에서만 값이 바뀜 → id_payload_diff_mean_mean 증가
    # Replay   : 동일 ID에서 변화가 거의 없음 → id_payload_diff_std_mean 낮음
    # ---------------------------------------------------------
    id_diff_means  = []
    id_diff_stds   = []
    id_bflip_means = []
    changed_ids    = 0

    for _, grp in window_df.groupby("Arbitration_ID", sort=False):
        grp_payloads = grp["payload_bytes"].tolist()
        g  = len(grp_payloads)
        if g < 2:
            continue
        ml = max((len(p) for p in grp_payloads), default=0)
        if ml == 0:
            continue

        ga = np.zeros((g, ml), dtype=np.uint8)
        for i, p in enumerate(grp_payloads):
            if p:
                ga[i, :len(p)] = p

        gdiff = np.abs(ga[1:].astype(np.int16) - ga[:-1].astype(np.int16))
        gpm   = gdiff.mean(axis=1)

        id_diff_means.append(float(gpm.mean()))
        id_diff_stds.append(float(gpm.std()) if len(gpm) >= 2 else 0.0)

        gxor  = ga[1:] ^ ga[:-1]
        gbits = np.unpackbits(gxor.ravel()).reshape(g - 1, -1).sum(axis=1)
        id_bflip_means.append(float(gbits.mean() / ml))

        if float(gpm.mean()) > 0:
            changed_ids += 1

    n_uid = feats["unique_id_count"] if feats["unique_id_count"] > 0 else 1
    feats["id_payload_diff_mean_mean"] = float(np.mean(id_diff_means))  if id_diff_means  else 0.0
    feats["id_payload_diff_std_mean"]  = float(np.mean(id_diff_stds))   if id_diff_stds   else 0.0
    feats["id_bitflip_mean_mean"]      = float(np.mean(id_bflip_means)) if id_bflip_means else 0.0
    feats["changed_id_ratio"]          = changed_ids / n_uid

    # ---------------------------------------------------------
    # 7. 이전 윈도우 Jaccard 유사도
    # Replay : 이전 구간과 동일한 (ID, Data) 반복 → jaccard 높음
    # ---------------------------------------------------------
    if prev_df is not None and len(prev_df) > 0:
        prev_set = set(
            (prev_df["Arbitration_ID"].astype(str) + "_" +
             prev_df["Data"].astype(str)).tolist()
        )
        curr_set = set(id_data.tolist())
        union    = len(prev_set | curr_set)
        inter    = len(prev_set & curr_set)
        feats["prev_window_jaccard"] = inter / union if union > 0 else 0.0
    else:
        feats["prev_window_jaccard"] = 0.0

    return feats


# =========================================================
# 윈도우 목록 → Feature DataFrame
# =========================================================

def build_feature_dataframe(windows):
    """windows 리스트에서 피처 테이블을 생성한다."""
    rows    = []
    prev_df = None

    for w in windows:
        feats            = extract_features(w["data"], prev_df=prev_df)
        feats["label"]   = w["label"]
        feats["purity"]  = w["purity"]
        feats["start"]   = w["start"]
        feats["end"]     = w["end"]
        rows.append(feats)
        prev_df = w["data"]

    return pd.DataFrame(rows)


# =========================================================
# 피처 컬럼명 (26개)
# =========================================================

FEATURE_COLS = (
    # 1. 기본 통계 (4)
    ["msg_count", "unique_id_count", "mean_dlc", "std_dlc"]
    # 2. IAT (5)
    + ["iat_mean", "iat_std", "iat_min", "iat_max", "burstiness"]
    # 3. ID 분포 (3)
    + ["top1_id_ratio", "top3_id_ratio_sum", "id_entropy"]
    # 4. ID+Data 반복 / payload 다양성 (6)
    + ["unique_id_data_count", "top1_id_data_ratio", "repeat_id_data_ratio",
       "payload_entropy", "payload_byte_entropy", "max_same_payload_run"]
    # 5. payload 변화량 (3)
    + ["byte_diff_mean", "byte_diff_std", "bit_flip_mean"]
    # 6. ID-level payload summary (4)
    + ["id_payload_diff_mean_mean", "id_payload_diff_std_mean",
       "id_bitflip_mean_mean", "changed_id_ratio"]
    # 7. 이전 윈도우 유사도 (1)
    + ["prev_window_jaccard"]
)  # 총 26개


def get_XY(feature_df):
    """피처 DataFrame에서 X, y 분리."""
    X = feature_df[FEATURE_COLS]
    y = feature_df["label"]
    return X, y


# =========================================================
# 디버그용 실행
# =========================================================

if __name__ == "__main__":
    from preprocess import load_and_preprocess
    from config import PATHS

    df = load_and_preprocess(PATHS["train"], split_name="TRAIN")

    print("\n슬라이딩 윈도우 생성 중...")
    windows = generate_sliding_windows(df)
    print(f"윈도우 수: {len(windows):,}")

    print("\n피처 테이블 생성 중...")
    feat_df = build_feature_dataframe(windows)
    print(f"피처 테이블: {feat_df.shape}")
    print(f"피처 수: {len(FEATURE_COLS)}개")

    print("\n레이블 분포:")
    for idx, cnt in feat_df["label"].value_counts().sort_index().items():
        print(f"  {LABEL_NAMES[idx]:<12} {cnt:>8,}")
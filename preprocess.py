import pandas as pd
import numpy as np

from config import LABEL_MAP, PATHS


# =========================================================
# Arbitration ID 정규화
# =========================================================

def normalize_arbitration_id(x):
    """
    10진수 문자열 → int 그대로 변환
    16진수 문자열 → int(x, 16) 변환
    파싱 실패 시 -1 반환
    """
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    try:
        return int(s, 16)
    except ValueError:
        return -1


# =========================================================
# DATA 바이트 파싱
# =========================================================

def parse_data_bytes(data_str):
    """
    "C0 03 00 00 00 00 00 00" → [192, 3, 0, 0, 0, 0, 0, 0]
    파싱 실패 시 빈 리스트 반환
    """
    if pd.isna(data_str):
        return []
    parts = str(data_str).strip().split()
    try:
        return [int(p, 16) for p in parts]
    except ValueError:
        return []


# =========================================================
# 단일 파일 로드
# =========================================================

def load_single(path):
    """
    CSV 파일 1개 로드
    SubClass 컬럼이 없으면 (Pre_train_S_0 등) 전부 Normal로 채운다
    """
    df = pd.read_csv(path)

    if "SubClass" not in df.columns:
        df["SubClass"] = "Normal"
    if "Class" not in df.columns:
        df["Class"] = "Normal"

    df["SubClass"] = df["SubClass"].fillna("Normal")
    return df


# =========================================================
# 복수 파일 로드 + 전처리
# =========================================================

def load_and_preprocess(file_paths, split_name=""):
    """
    여러 CSV 파일을 로드하여 하나의 DataFrame으로 반환

    처리 순서
    ---------
    1. 파일별 로드 후 concat
    2. Timestamp 오름차순 정렬
    3. Arbitration_ID 정규화 (hex → int)
    4. Data 바이트 파싱
    5. y_msg 라벨 컬럼 생성
       - LABEL_MAP에 없는 SubClass(Flooding, Fuzzing)는 자동 제거
    """
    dfs = []
    for path in file_paths:
        df = load_single(path)
        dfs.append(df)
        print(f"  로드: {path}  ({len(df):,}행)")

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("Timestamp").reset_index(drop=True)

    data["Arbitration_ID"] = data["Arbitration_ID"].apply(normalize_arbitration_id)
    data["payload_bytes"]  = data["Data"].apply(parse_data_bytes)

    # LABEL_MAP에 없는 클래스(Flooding, Fuzzing) → NaN → 제거
    data["y_msg"] = data["SubClass"].map(LABEL_MAP)
    removed = data["y_msg"].isna().sum()
    if removed > 0:
        print(f"  [제거] LABEL_MAP 미포함 행 {removed:,}개 (Flooding/Fuzzing 등)")
        data = data[data["y_msg"].notna()].copy()

    data["y_msg"] = data["y_msg"].astype(int)

    print(f"\n[{split_name}] 전처리 완료: 총 {len(data):,}행")
    _print_class_distribution(data, split_name)

    return data


# =========================================================
# 클래스 분포 출력
# =========================================================

def _print_class_distribution(df, title=""):
    dist  = df["SubClass"].value_counts().sort_values(ascending=False)
    total = len(df)
    print(f"\n  클래스 분포 ({title}):")
    for cls, cnt in dist.items():
        ratio = cnt / total * 100
        print(f"    {cls:<12} {cnt:>10,}  ({ratio:5.1f}%)")


# =========================================================
# 디버그용 실행
# =========================================================

if __name__ == "__main__":
    for split, paths in PATHS.items():
        print(f"\n{'='*50}  {split.upper()}")
        df = load_and_preprocess(paths, split_name=split.upper())
        print(f"  shape: {df.shape}")
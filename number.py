import os
import pandas as pd
from collections import defaultdict

# =========================================================
# 경로 설정
# =========================================================

BASE = ".."

FILES = {
    "Pre_train_D_0": f"{BASE}/0_Preliminary/0_Training/Pre_train_D_0.csv",
    "Pre_train_D_1": f"{BASE}/0_Preliminary/0_Training/Pre_train_D_1.csv",
    "Pre_train_D_2": f"{BASE}/0_Preliminary/0_Training/Pre_train_D_2.csv",
    "Pre_train_S_0": f"{BASE}/0_Preliminary/0_Training/Pre_train_S_0.csv",
    "Pre_train_S_1": f"{BASE}/0_Preliminary/0_Training/Pre_train_S_1.csv",
    "Pre_train_S_2": f"{BASE}/0_Preliminary/0_Training/Pre_train_S_2.csv",
    "Pre_submit_D":  f"{BASE}/0_Preliminary/1_Submission/Pre_submit_D.csv",
    "Pre_submit_S":  f"{BASE}/0_Preliminary/1_Submission/Pre_submit_S.csv",
    "Fin_host_S":    f"{BASE}/1_Final/Fin_host_session_submit_S.csv",
}

ROLE = {
    "Pre_train_D_0": ("Driving",     "Train"),
    "Pre_train_D_1": ("Driving",     "Train"),
    "Pre_train_D_2": ("Driving",     "Train"),
    "Pre_train_S_0": ("Stationary",  "Train"),
    "Pre_train_S_1": ("Stationary",  "Train"),
    "Pre_train_S_2": ("Stationary",  "Train"),
    "Pre_submit_D":  ("Driving",     "Validation"),
    "Pre_submit_S":  ("Stationary",  "Validation"),
    "Fin_host_S":    ("Stationary",  "Test"),
}

# =========================================================
# 단일 파일 분석
# =========================================================

def analyze_file(name, path):
    if not os.path.exists(path):
        print(f"  [파일 없음] {path}")
        return None

    df = pd.read_csv(path)
    total = len(df)
    has_subclass = "SubClass" in df.columns

    # SubClass 없으면 전부 Normal로 간주
    if not has_subclass:
        class_dist = {"Normal": total}
        subclass_dist = {"Normal": total}
    else:
        df["SubClass"] = df["SubClass"].fillna("Normal")
        class_dist = df["Class"].value_counts().to_dict() if "Class" in df.columns else {}
        subclass_dist = df["SubClass"].value_counts().to_dict()

    return {
        "name": name,
        "path": path,
        "condition": ROLE[name][0],
        "split":     ROLE[name][1],
        "total":     total,
        "subclass":  subclass_dist,
    }


# =========================================================
# 전체 분석 및 출력
# =========================================================

def main():
    results = []
    for name, path in FILES.items():
        print(f"로드 중: {name} ...")
        r = analyze_file(name, path)
        if r:
            results.append(r)

    print("\n" + "=" * 80)
    print("파일별 상세")
    print("=" * 80)
    for r in results:
        print(f"\n[{r['split']}] {r['name']}  ({r['condition']})")
        print(f"  전체 행 수: {r['total']:,}")
        print(f"  SubClass 분포:")
        for cls, cnt in sorted(r["subclass"].items(), key=lambda x: -x[1]):
            ratio = cnt / r["total"] * 100
            print(f"    {cls:<12} {cnt:>10,}  ({ratio:5.1f}%)")

    # -------------------------------------------------------
    # Split × Condition 단위 집계
    # -------------------------------------------------------
    print("\n" + "=" * 80)
    print("Split × Condition 집계")
    print("=" * 80)

    groups = defaultdict(lambda: defaultdict(int))
    for r in results:
        key = (r["split"], r["condition"])
        groups[key]["total"] += r["total"]
        for cls, cnt in r["subclass"].items():
            groups[key][cls] += cnt

    for (split, cond), stats in sorted(groups.items()):
        total = stats["total"]
        print(f"\n[{split}] {cond}  →  총 {total:,}행")
        for cls, cnt in sorted(stats.items(), key=lambda x: -x[1]):
            if cls == "total":
                continue
            ratio = cnt / total * 100
            print(f"  {cls:<12} {cnt:>10,}  ({ratio:5.1f}%)")

    # -------------------------------------------------------
    # 전체 요약 테이블
    # -------------------------------------------------------
    print("\n" + "=" * 80)
    print("요약 테이블")
    print("=" * 80)
    header = f"{'파일명':<22} {'조건':<12} {'역할':<12} {'Normal':>10} {'Attack':>10} {'Total':>10}"
    print(header)
    print("-" * 80)

    grand_total = grand_normal = grand_attack = 0
    for r in results:
        normal = r["subclass"].get("Normal", 0)
        attack = r["total"] - normal
        grand_total  += r["total"]
        grand_normal += normal
        grand_attack += attack
        print(
            f"{r['name']:<22} {r['condition']:<12} {r['split']:<12}"
            f" {normal:>10,} {attack:>10,} {r['total']:>10,}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<22} {'':<12} {'':<12}"
        f" {grand_normal:>10,} {grand_attack:>10,} {grand_total:>10,}"
    )


if __name__ == "__main__":
    main()
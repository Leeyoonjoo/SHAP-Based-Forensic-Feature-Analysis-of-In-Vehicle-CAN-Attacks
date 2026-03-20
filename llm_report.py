"""
llm_report.py  (Ollama 로컬 LLM 버전)
───────────────────────────────────────
run_forensic_report() 의 구조화 결과를 Ollama에 전달하여
조사관용 포렌식 보고서(마크다운)를 자동 생성한다.

사전 준비
─────────
1. Ollama 설치 : https://ollama.com/download
2. 모델 다운로드: ollama pull llama3.2
3. Ollama 실행  : 설치 후 보통 자동으로 백그라운드에서 실행됨
                  안 되면 터미널에서 'ollama serve' 실행

모델 변경
─────────
OLLAMA_MODEL 상수를 바꾸면 된다.
  "llama3.2"      → 3B, 빠름, 권장
  "llama3.1:8b"   → 8B, 좋은 품질
  "gemma2:9b"     → 구글, 한국어 준수
  "qwen2.5:7b"    → 한국어 성능 좋음
  "mistral"       → 7B, 영어 강점
"""

# import os
# import json
# import requests
# from config import OUTPUT_DIR

# OLLAMA_URL   = "http://localhost:11434/api/chat"
# OLLAMA_MODEL = "qwen2.5:3b"  # ← 원하는 모델로 변경


# # =========================================================
# # 증거 데이터 → 프롬프트 텍스트 변환
# # =========================================================

# def _summarize_evidence(evi):
#     """frame_evidence 딕셔너리를 한 줄 핵심 요약으로 변환."""
#     if "top_repeated_pairs" in evi:
#         top = next(iter(evi["top_repeated_pairs"]), "")
#         cnt = evi["top_repeated_pairs"].get(top, 0) if top else 0
#         return f"반복쌍 '{top}' {cnt}회, 반복비율={evi.get('repeat_ratio', 0):.1%}"
#     if "top_ids" in evi:
#         top = next(iter(evi["top_ids"]), "")
#         cnt = evi["top_ids"].get(top, 0) if top else 0
#         return f"지배ID {top} {cnt}회, top1비율={evi.get('top1_ratio', 0):.1%}"
#     if "iat_std_ms" in evi:
#         return f"IAT표준편차={evi['iat_std_ms']:.3f}ms, 버스트={evi.get('burst_frames', 0)}회"
#     if "unique_payloads" in evi:
#         return (f"고유payload={evi['unique_payloads']}종, "
#                 f"최다='{evi.get('most_common_payload', '')}' "
#                 f"{evi.get('most_common_payload_cnt', 0)}회")
#     if "payload_change_rate" in evi:
#         return f"payload변화율={evi['payload_change_rate']:.1%}"
#     note = evi.get("note", "")
#     return note[:60] if note else "-"


# def _format_for_prompt(all_results):
#     lines = []

#     for result in all_results:
#         attack = result["attack_type"]
#         lines.append(f"## {attack} 공격 "
#                      f"(총 {result['window_count']}개 구간 탐지, "
#                      f"대표 {len(result['windows'])}개 분석)")

#         for w in result["windows"]:
#             lines.append(f"\n### 구간 #{w['rank']} "
#                          f"({w['t_start']}s~{w['t_end']}s, SHAP합계={w['shap_total']})")
#             lines.append("**핵심 피처 (SHAP 상위 3개):**")
#             for feat in w["shap_top5"][:3]:
#                 evi     = w["frame_evidence"].get(feat["feature"], {})
#                 evi_str = _summarize_evidence(evi)
#                 lines.append(
#                     f"  - {feat['feature']}: SHAP={feat['shap_value']:+.4f}, "
#                     f"실측={feat['actual']:.4f} (정상평균={feat['normal_mean']:.4f}) | {evi_str}"
#                 )

#         lines.append("")

#     return "\n".join(lines)


# # =========================================================
# # LLM 보고서 생성
# # =========================================================

# SYSTEM_PROMPT = """당신은 차량 사이버보안 분석가입니다. IDS 분석 결과를 바탕으로 조사 보고서를 작성합니다.

# [문체 규칙 — 반드시 지켜야 함]
# - 단어나 표현에 쌍따옴표를 쓰지 않는다. (예: "관찰됨" → 관찰됨, "높은 값" → 높은 값)
# - ~됩니다, ~것입니다, ~하였습니다 체를 쓰지 않는다. 서술형으로 간결하게 쓴다.
# - 불필요한 소제목, 번호 매기기, 빈 줄 남발을 피한다.
# - 수치는 문장 안에 자연스럽게 녹인다. (예: "SHAP 값은 +0.42로" 가 아니라 "SHAP +0.42는")
# - 각 공격 구간은 3~4문장 단락으로 쓴다. 증거 → 이유 순서로.

# [보고서 구조]
# # CAN 버스 침입 분석 보고서
# ## 분석 개요
# (데이터셋, 탐지 공격 유형, 총 탐지 구간 수를 두 문장으로)
# ## Replay Attack 분석
# (각 구간을 단락 형식으로, ### 구간 #N 제목 사용)
# ## Spoofing Attack 분석
# (동일 형식)
# ## 종합 판단"""


# def generate_llm_report(all_results, split_name="Test"):
#     """
#     포렌식 분석 결과를 Ollama 로컬 LLM에 전달하여 조사관용 보고서 생성.

#     Parameters
#     ----------
#     all_results : list of dict
#         run_forensic_report() 의 반환값
#     split_name  : str
#         저장 파일명 구분용

#     Returns
#     -------
#     report_path : str  저장된 마크다운 파일 경로  (실패 시 None)
#     """
#     print(f"\n{'='*60}")
#     print(f"[{split_name}] LLM 포렌식 보고서 생성 중... (모델: {OLLAMA_MODEL})")
#     print(f"{'='*60}")

#     if not all_results:
#         print("  경고: 분석 결과가 없어 보고서를 생성할 수 없습니다.")
#         return None

#     evidence_text = _format_for_prompt(all_results)

#     user_prompt = f"""아래 데이터를 바탕으로 CAN 버스 침입 분석 보고서를 한국어로 작성해라.

# 분석 대상: {split_name} 세트 | 도구: XGBoost IDS + SHAP

# 피처 의미:
# repeat_id_data_ratio: 같은 (ID, payload) 쌍이 반복된 비율. 높으면 동일 메시지 재주입.
# top1_id_ratio: 가장 많이 등장한 ID의 점유율. 높으면 특정 ID 집중.
# payload_entropy: payload 다양성. 낮으면 단조로운 패턴(Replay 특성).
# iat_std: 프레임 전송 간격의 표준편차. 낮으면 기계적으로 균일한 주입.
# max_same_payload_run: 동일 payload가 연속된 최대 길이.

# --- 분석 데이터 ---
# {evidence_text}
# ---

# 작성 방식:
# - 각 구간은 단락 하나로. 번호 목록 쓰지 말고 문장으로 이어서 써라.
# - 수치를 직접 언급하되 자연스럽게 녹여라. (예: SHAP +0.38은 정상 대비 replay_id_data_ratio가 2.3배 높은 것에서 비롯됐다)
# - 왜 이 수치 패턴이 해당 공격 유형의 흔적인지 CAN 버스 동작 원리로 설명해라.
# - 단어에 쌍따옴표 붙이지 마라. ~됩니다 체 쓰지 마라."""

#     # 프롬프트 길이 확인 (디버그)
#     total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
#     print(f"  프롬프트 길이: {total_chars}자 (~{total_chars//4} 토큰 추정)")

#     # system 메시지를 user 메시지 앞에 합침 (일부 모델은 system role 미지원)
#     combined_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

#     payload = {
#         "model": OLLAMA_MODEL,
#         "messages": [
#             {"role": "user", "content": combined_prompt},
#         ],
#         "stream": True,
#         "options": {
#             "num_ctx": 4096,      # 컨텍스트 창 제한 (메모리 절약)
#             "num_predict": 1024,  # 최대 출력 토큰
#         },
#     }

#     try:
#         print(f"  Ollama 스트리밍 호출 중 (모델: {OLLAMA_MODEL}) ", end="", flush=True)
#         resp = requests.post(OLLAMA_URL, json=payload,
#                              timeout=(10, 300), stream=True)
#         if not resp.ok:
#             print(f"\n  [오류 상세] status={resp.status_code}, body={resp.text[:500]}")
#         resp.raise_for_status()

#         report_text = ""
#         token_count = 0
#         for line in resp.iter_lines():
#             if not line:
#                 continue
#             try:
#                 chunk = json.loads(line.decode("utf-8"))
#             except (json.JSONDecodeError, UnicodeDecodeError):
#                 continue
#             token = chunk.get("message", {}).get("content", "")
#             report_text += token
#             token_count += 1
#             if token_count % 100 == 0:
#                 print(".", end="", flush=True)
#             if chunk.get("done"):
#                 break
#         print(f" ({token_count} tokens)")

#     except requests.exceptions.ConnectionError:
#         print("\n  [오류] Ollama에 연결할 수 없습니다.")
#         print("  => Ollama가 실행 중인지 확인하세요: ollama serve")
#         print(f"  => 모델이 설치됐는지 확인하세요: ollama pull {OLLAMA_MODEL}")
#         return None
#     except requests.exceptions.Timeout:
#         print("\n  [오류] Ollama 응답 시간 초과")
#         return None
#     except Exception as e:
#         print(f"\n  [오류] LLM 보고서 생성 실패: {e}")
#         return None

#     report_path = os.path.join(
#         OUTPUT_DIR,
#         f"forensic_report_llm_{split_name.lower()}.md"
#     )
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(report_text)

#     print(f"  LLM 보고서 저장: {report_path}")
#     print(f"[{split_name}] LLM 보고서 생성 완료")
#     return report_path

import os
import json
import requests
from config import OUTPUT_DIR

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"   # 필요하면 qwen2.5:3b, llama3.1:8b 등으로 변경


# =========================================================
# 유틸
# =========================================================

def _fmt_float(v, digits=4):
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def _top1_item(d):
    if not isinstance(d, dict) or not d:
        return None, None
    k = next(iter(d))
    return k, d[k]


# =========================================================
# 증거 요약 생성
# =========================================================

def _summarize_feature_evidence(feature_name, feat_info, evidence_info):
    """
    각 SHAP 상위 피처에 대해 LLM에 넘길 조사관용 증거 문장 생성
    """
    actual = feat_info.get("actual")
    normal = feat_info.get("normal_mean")
    shap_v = feat_info.get("shap_value")

    actual_s = _fmt_float(actual, 6)
    normal_s = _fmt_float(normal, 6)
    shap_s = f"{float(shap_v):+.4f}" if shap_v is not None else "N/A"

    lines = [
        f"- 피처: {feature_name}",
        f"  SHAP: {shap_s}, 실측: {actual_s}, 정상평균: {normal_s}",
    ]

    # 1) ID 분포 관련
    if any(k in feature_name for k in ["top1_id", "top3_id", "id_entropy", "unique_id_count"]):
        top_ids = _safe_get(evidence_info, "top_ids", {})
        top1_ratio = _safe_get(evidence_info, "top1_ratio", 0)
        unique_id_count = _safe_get(evidence_info, "unique_id_count", 0)
        total_frames = _safe_get(evidence_info, "total_frames", 0)
        top_id, top_cnt = _top1_item(top_ids)

        if top_id is not None:
            lines.append(
                f"  증거: 지배적 ID는 {top_id}, {top_cnt}회 출현, "
                f"top1 점유율 {top1_ratio:.1%}, 고유 ID 수 {unique_id_count}, 총 프레임 {total_frames}"
            )
        else:
            lines.append(
                f"  증거: 고유 ID 수 {unique_id_count}, 총 프레임 {total_frames}"
            )

    # 2) 반복 전송 관련
    elif any(k in feature_name for k in ["repeat_id_data", "unique_id_data", "max_same_payload"]):
        top_pairs = _safe_get(evidence_info, "top_repeated_pairs", {})
        repeat_ratio = _safe_get(evidence_info, "repeat_ratio", 0)
        max_payload_run = _safe_get(evidence_info, "max_payload_run", 0)
        top_pair, top_pair_cnt = _top1_item(top_pairs)

        if top_pair is not None:
            lines.append(
                f"  증거: 대표 반복 (ID|payload) 쌍은 {top_pair}, {top_pair_cnt}회 반복, "
                f"반복 비율 {repeat_ratio:.1%}, 최장 동일 payload 연속 길이 {max_payload_run}"
            )
        else:
            lines.append(
                f"  증거: 반복 비율 {repeat_ratio:.1%}, 최장 동일 payload 연속 길이 {max_payload_run}"
            )

    # 3) payload 다양성 / entropy 관련
    elif any(k in feature_name for k in ["payload_entropy", "payload_byte"]):
        unique_payloads = _safe_get(evidence_info, "unique_payloads", 0)
        most_common_payload = _safe_get(evidence_info, "most_common_payload", "")
        most_common_payload_cnt = _safe_get(evidence_info, "most_common_payload_cnt", 0)
        total_frames = _safe_get(evidence_info, "total_frames", 0)

        lines.append(
            f"  증거: 고유 payload 수 {unique_payloads}, 최다 payload {most_common_payload}, "
            f"{most_common_payload_cnt}회 출현, 총 프레임 {total_frames}"
        )

    # 4) IAT / burstiness 관련
    elif any(k in feature_name for k in ["iat", "burstiness"]):
        iat_mean_ms = _safe_get(evidence_info, "iat_mean_ms", None)
        iat_std_ms = _safe_get(evidence_info, "iat_std_ms", None)
        iat_min_ms = _safe_get(evidence_info, "iat_min_ms", None)
        burst_frames = _safe_get(evidence_info, "burst_frames", 0)

        if iat_mean_ms is not None or iat_std_ms is not None:
            lines.append(
                f"  증거: IAT 평균 {iat_mean_ms if iat_mean_ms is not None else 'N/A'}ms, "
                f"IAT 표준편차 {iat_std_ms if iat_std_ms is not None else 'N/A'}ms, "
                f"IAT 최소 {iat_min_ms if iat_min_ms is not None else 'N/A'}ms, "
                f"버스트 프레임 {burst_frames}"
            )
        else:
            note = _safe_get(evidence_info, "note", "프레임 수 부족")
            lines.append(f"  증거: {note}")

    # 5) payload 변화량
    elif any(k in feature_name for k in ["byte_diff", "bit_flip"]):
        payload_change_rate = _safe_get(evidence_info, "payload_change_rate", None)
        note = _safe_get(evidence_info, "note", "")

        if payload_change_rate is not None:
            lines.append(
                f"  증거: 연속 프레임 간 payload 변화율 {payload_change_rate:.1%}, {note}"
            )
        else:
            lines.append(f"  증거: {note}")

    # 6) msg_count
    elif feature_name == "msg_count":
        msg_count = _safe_get(evidence_info, "msg_count", None)
        if msg_count is not None:
            lines.append(f"  증거: 윈도우 내 총 메시지 수 {msg_count}")

    # 7) DLC
    elif "dlc" in feature_name:
        mean_dlc = _safe_get(evidence_info, "mean_dlc", None)
        std_dlc = _safe_get(evidence_info, "std_dlc", None)
        if mean_dlc is not None:
            lines.append(f"  증거: 평균 DLC {mean_dlc}, DLC 표준편차 {std_dlc}")

    # 8) 기타
    else:
        note = _safe_get(evidence_info, "note", "참조용 피처")
        lines.append(f"  증거: {note}")

    return "\n".join(lines)


def _summarize_z_scores(z_top5):
    if not z_top5:
        return "- 없음"

    out = []
    for z in z_top5:
        feat = z.get("feature", "")
        z_score = z.get("z_score", 0)
        actual = z.get("actual", 0)
        normal_mean = z.get("normal_mean", 0)
        direction = z.get("direction", "-")
        out.append(
            f"- {feat}: z={float(z_score):+.2f}, 실측={_fmt_float(actual, 6)}, "
            f"정상평균={_fmt_float(normal_mean, 6)}, 방향={direction}"
        )
    return "\n".join(out)


def _build_window_block(attack_type, window_info):
    rank = window_info["rank"]
    t_start = window_info["t_start"]
    t_end = window_info["t_end"]
    shap_total = window_info["shap_total"]
    shap_top5 = window_info.get("shap_top5", [])
    z_top5 = window_info.get("z_top5", [])
    frame_evidence = window_info.get("frame_evidence", {})

    lines = []
    lines.append(f"### 구간 #{rank}")
    lines.append(f"- 공격 유형: {attack_type}")
    lines.append(f"- 시간 범위: {t_start}s ~ {t_end}s")
    lines.append(f"- SHAP 총합: {shap_total}")
    lines.append(f"- 정상 대비 이상 피처(z-score 상위):")
    lines.append(_summarize_z_scores(z_top5))
    lines.append(f"- 주요 포렌식 증거(SHAP 상위 피처 기준):")

    for feat in shap_top5:
        feat_name = feat["feature"]
        evi = frame_evidence.get(feat_name, {})
        lines.append(_summarize_feature_evidence(feat_name, feat, evi))

    return "\n".join(lines)


def _format_for_prompt(all_results):
    """
    LLM에 넘길 조사관용 구조화 텍스트 생성
    """
    lines = []
    total_windows = sum(r.get("window_count", 0) for r in all_results)

    lines.append("[전체 분석 요약]")
    lines.append(f"- 전체 탐지 구간 수: {total_windows}")

    for result in all_results:
        attack_type = result["attack_type"]
        window_count = result["window_count"]
        windows = result.get("windows", [])

        lines.append("")
        lines.append(f"[{attack_type} 공격 요약]")
        lines.append(f"- 탐지 구간 수: {window_count}")
        lines.append(f"- 대표 분석 구간 수: {len(windows)}")

        for w in windows:
            lines.append("")
            lines.append(_build_window_block(attack_type, w))

    return "\n".join(lines)


# =========================================================
# 시스템 프롬프트
# =========================================================

SYSTEM_PROMPT = """
당신은 차량 디지털 포렌식 분석가다.
출력은 단순 XAI 설명문이 아니라, 조사관이 읽는 CAN 버스 침입 분석 보고서여야 한다.

반드시 지킬 규칙:
1. 각 공격 구간은 시간 범위, 의심 ID 또는 지배적 ID, 반복된 ID|payload 또는 대표 payload, 반복 횟수 또는 점유율, IAT 또는 timing 특성, 정상 대비 편차, 공격 판단 이유를 포함한다.
2. 피처 이름 자체를 나열하는 데 그치지 말고, 실제 관측된 CAN 프레임 현상으로 풀어서 설명한다.
3. 공격 구간에 대해 정상적, 안정적, 이상적, 자연스럽다 같은 표현을 쓰지 않는다.
4. Replay Attack은 동일 메시지 재전송, 반복 payload, 낮은 timing 변동성, 기계적 주입 패턴 관점에서 설명한다.
5. Spoofing Attack은 특정 ID 편중, 정상 ID를 가장한 값 주입, payload 변화 또는 조작 흔적 관점에서 설명한다.
6. 단정이 어려우면 가능성, 정황, 추정으로 표현한다.
7. 각 구간 마지막 문장은 조사관 판단으로 끝낸다.
8. SHAP 값은 보조 근거다. 원본 프레임 증거와 정상 대비 편차를 우선 서술한다.
9. 공격 구간 하나당 4~6문장 정도로 쓴다.
10. 보고서 마지막에는 분석 한계와 추가 확인 필요 사항을 반드시 쓴다.

보고서 형식:
# CAN 버스 침입 분석 보고서
## 분석 개요
## 분석 기준 및 한계
## Replay Attack 분석
### 구간 #1
...
## Spoofing Attack 분석
### 구간 #1
...
## 종합 판단
## 추가 확인 권고 사항
""".strip()


# =========================================================
# LLM 보고서 생성
# =========================================================

def generate_llm_report(all_results, split_name="Test"):
    """
    run_forensic_report()의 결과를 바탕으로
    Ollama 로컬 LLM에게 포렌식 보고서 생성을 요청한다.

    Parameters
    ----------
    all_results : list of dict
        shap_analysis.run_forensic_report() 반환값
    split_name : str
        파일명 구분용

    Returns
    -------
    report_path : str or None
    """
    print(f"\n{'=' * 70}")
    print(f"[{split_name}] LLM 포렌식 보고서 생성 시작 (모델: {OLLAMA_MODEL})")
    print(f"{'=' * 70}")

    if not all_results:
        print("  [경고] all_results가 비어 있어 보고서를 생성할 수 없습니다.")
        return None

    evidence_text = _format_for_prompt(all_results)

    user_prompt = f"""
아래는 {split_name} 세트에서 탐지된 CAN 버스 공격 구간의 포렌식 증거다.
이를 바탕으로 조사관용 한국어 보고서를 작성하라.

분석 도구:
- 탐지 모델: XGBoost IDS
- 설명 방법: SHAP
- 보고 목적: 공격 구간의 프레임 수준 흔적을 근거로 Replay Attack 및 Spoofing Attack의 정황을 정리

중요한 작성 규칙:
- 모델 해석 설명문처럼 쓰지 말고, 포렌식 조사 문서처럼 써라.
- 피처명만 반복하지 말고, 실제로 어떤 ID, payload, timing, 반복 특성이 관찰되었는지 중심으로 쓴다.
- 공격 구간에 대해 정상적, 안정적, 이상적이라는 표현은 금지한다.
- 각 구간마다 왜 Replay 또는 Spoofing으로 볼 수 있는지 논리적으로 써라.
- 마지막에는 본 분석이 윈도우 기반 추정이며 ECU 단위 확정에는 추가 로그와 명세 대조가 필요하다고 명시하라.

아래 증거를 이용하라.

{evidence_text}
""".strip()

    combined_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    total_chars = len(combined_prompt)

    print(f"  프롬프트 길이: {total_chars}자")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": combined_prompt}
        ],
        "stream": True,
        "options": {
            "num_ctx": 8192,
            "num_predict": 1800,
            "temperature": 0.2,
        },
    }

    try:
        print(f"  Ollama 호출 중... ", end="", flush=True)
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(30, 900),
            stream=True
        )

        if not resp.ok:
            print(f"\n  [오류 상세] status={resp.status_code}, body={resp.text[:500]}")
        resp.raise_for_status()

        report_text = ""
        chunk_count = 0

        for line in resp.iter_lines():
            if not line:
                continue

            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            token = chunk.get("message", {}).get("content", "")
            report_text += token
            chunk_count += 1

            if chunk_count % 80 == 0:
                print(".", end="", flush=True)

            if chunk.get("done"):
                break

        print(f" 완료 ({chunk_count} chunks)")

    except requests.exceptions.ConnectionError:
        print("\n  [오류] Ollama에 연결할 수 없습니다.")
        print("  - ollama serve 실행 여부 확인")
        print(f"  - 모델 설치 여부 확인: ollama pull {OLLAMA_MODEL}")
        return None

    except requests.exceptions.Timeout:
        print("\n  [오류] Ollama 응답 시간이 초과되었습니다.")
        return None

    except Exception as e:
        print(f"\n  [오류] LLM 보고서 생성 실패: {e}")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(
        OUTPUT_DIR,
        f"forensic_report_llm_{split_name.lower()}.md"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  보고서 저장 완료: {report_path}")
    print(f"[{split_name}] LLM 포렌식 보고서 생성 완료")
    return report_path


# =========================================================
# 단독 테스트용
# =========================================================

if __name__ == "__main__":
    print("이 파일은 단독 실행보다 import 후 generate_llm_report(all_results)로 사용하는 것을 권장함.")
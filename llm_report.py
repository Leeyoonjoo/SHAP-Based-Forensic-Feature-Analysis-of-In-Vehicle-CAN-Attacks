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



import os
import re
import json
import requests
from config import OUTPUT_DIR

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"   # 필요하면 qwen2.5:3b, gemma2:9b 등으로 변경

# =========================================================
# 피처 의미 사전 (LLM 프롬프트에 삽입)
# =========================================================
FEATURE_SEMANTICS = {
    "id_payload_diff_std_mean":  "ID별 payload 변화량의 표준편차 평균. 높을수록 동일 ID가 다양한 payload 패턴으로 반복 전송됨 → Replay 핵심 지표",
    "id_payload_diff_mean_mean": "ID별 연속 payload 변화량 평균. 높으면 동일 ID가 payload를 바꿔가며 반복 주입됨",
    "repeat_id_data_ratio":      "동일 (ID, payload) 쌍 반복 비율. 높으면 동일 메시지가 그대로 재주입됨 → Replay 직접 증거",
    "iat_mean":                  "프레임 간 평균 전송 간격(ms). 낮을수록 프레임이 빠르게 밀집 주입됨",
    "iat_std":                   "프레임 간 전송 간격 표준편차. 낮으면 주입 간격이 기계적으로 균일함",
    "msg_count":                 "윈도우 내 총 프레임 수. 정상 대비 많으면 외부 프레임 주입 의심",
    "top1_id_ratio":             "가장 빈번한 ID의 점유율. 높으면 특정 ID가 비정상적으로 집중 전송됨 → Spoofing 핵심 지표",
    "top3_id_ratio_sum":         "상위 3개 ID 합산 점유율. 소수 ID 집중 전송 여부 지표",
    "mean_dlc":                  "평균 데이터 길이(bytes). 비정상값은 payload 구조 변조 의심",
    "payload_entropy":           "payload 다양성 지수. 낮으면 단조로운 payload 반복 → Replay 특성",
}

# 문장 스타일 예시 (few-shot)
_REPLAY_EXAMPLE = (
    "IAT 평균이 0.35ms로 정상(0.42ms)보다 낮고, 0.2초 윈도우 내 프레임이 568개로 정상 대비 18% 많다. "
    "CAN 버스에서 이처럼 고빈도·균일 간격으로 프레임이 밀집되는 것은 사전 캡처된 메시지를 기계적으로 재주입하는 Replay 패턴과 일치한다. "
    "조사관 판단: 해당 구간은 Replay Attack 정황이 명확하며, 동일 ID·payload 반복 여부를 원본 로그에서 대조해야 한다."
)
_SPOOFING_EXAMPLE = (
    "ID 0x233가 517프레임 중 37회(7.2%)를 점유해 정상 top1 점유율(4.2%) 대비 1.7배 높다. "
    "CAN 버스에서 특정 ID가 이렇게 편중되는 것은 해당 ID를 가장한 메시지가 반복 주입된 Spoofing 흔적으로 볼 수 있다. "
    "조사관 판단: 0x233 ID의 정상 전송 주기 및 ECU 명세를 대조하여 위조 여부를 확인해야 한다."
)


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

    for feat in shap_top5[:3]:
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

_IAT_FEATURES = {"iat_mean", "iat_std", "iat_min", "iat_max", "burstiness"}

def _build_window_prompt_item(attack_type, w):
    """구간 하나를 LLM에게 넘길 짧은 텍스트로 변환. IAT 계열 피처는 ms로 변환."""
    lines = [f"[{attack_type} 구간 #{w['rank']} | T={w['t_start']}s~{w['t_end']}s]"]
    for feat in w["shap_top5"][:3]:
        name   = feat["feature"]
        actual = feat.get("actual", 0)
        normal = feat.get("normal_mean", 0)
        shap_v = feat.get("shap_value", 0)
        evi    = w["frame_evidence"].get(name, {})

        # IAT 계열은 초→ms 변환
        if name in _IAT_FEATURES:
            actual_s = f"{actual * 1000:.4f}ms"
            normal_s = f"{normal * 1000:.4f}ms"
        else:
            actual_s = f"{actual:.4f}"
            normal_s = f"{normal:.4f}"

        evi_line = _summarize_feature_evidence(name, feat, evi)
        evi_oneline = evi_line.replace("\n", " | ")
        lines.append(
            f"  피처={name} | SHAP={shap_v:+.4f} | 실측={actual_s} | 정상평균={normal_s} | {evi_oneline}"
        )
    return "\n".join(lines)


def _call_ollama(user_prompt, assistant_prefix=""):
    """Ollama 스트리밍 호출. 성공 시 생성 텍스트 반환, 실패 시 None."""
    messages = [{"role": "user", "content": user_prompt}]
    if assistant_prefix:
        messages.append({"role": "assistant", "content": assistant_prefix})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "num_ctx": 2048,
            "num_predict": 300,
            "temperature": 0.3,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=(30, 600), stream=True)
        if not resp.ok:
            print(f"\n  [오류] status={resp.status_code}: {resp.text[:200]}")
            return None
        resp.raise_for_status()

        text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            text += chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                break
        return text.strip()

    except requests.exceptions.ConnectionError:
        print("\n  [오류] Ollama 연결 실패 — ollama serve 실행 확인")
        return None
    except requests.exceptions.Timeout:
        print("\n  [오류] Ollama 응답 시간 초과")
        return None
    except Exception as e:
        print(f"\n  [오류] {e}")
        return None


def generate_llm_report(all_results, split_name="Test"):
    """
    Python이 실제 수치로 보고서 뼈대를 만들고,
    LLM은 구간당 짧은 한국어 분석 문장만 생성한다.
    """
    print(f"\n{'=' * 70}")
    print(f"[{split_name}] LLM 포렌식 보고서 생성 (모델: {OLLAMA_MODEL})")
    print(f"{'=' * 70}")

    if not all_results:
        print("  [경고] all_results가 비어 있습니다.")
        return None

    total_windows = sum(r.get("window_count", 0) for r in all_results)
    attack_types  = [r["attack_type"] for r in all_results]

    # ── 보고서 뼈대 (Python이 실제 수치로 직접 작성) ──────────────────
    report_lines = [
        "# CAN 버스 침입 분석 보고서",
        "",
        "## 분석 개요",
        f"분석 대상: {split_name} 세트 | 탐지 도구: XGBoost IDS + SHAP",
        f"탐지된 공격 유형: {', '.join(attack_types)} | 총 탐지 구간: {total_windows}개",
        "",
        "## 분석 기준 및 한계",
        "- 0.2초 슬라이딩 윈도우 단위로 피처를 추출하여 XGBoost로 분류하였다.",
        "- SHAP 값은 모델 판단 근거를 설명하는 보조 지표이며 절대적 증거가 아니다.",
        "- ECU 단위 확정 판단에는 원본 로그와 ECU 명세 대조가 추가로 필요하다.",
        "",
    ]

    # ── 공격 유형별 섹션 ───────────────────────────────────────────────
    for result in all_results:
        attack = result["attack_type"]
        wcount = result["window_count"]
        windows = result.get("windows", [])

        report_lines.append(f"## {attack} Attack 분석")
        report_lines.append(f"총 {wcount}개 구간 탐지, 대표 {len(windows)}개 구간 분석.")
        report_lines.append("")

        for w in windows:
            rank     = w["rank"]
            t_start  = w["t_start"]
            t_end    = w["t_end"]
            shap_tot = w["shap_total"]

            report_lines.append(f"### 구간 #{rank}")
            report_lines.append(f"- 시간: {t_start}s ~ {t_end}s")
            report_lines.append(f"- SHAP 총합: {shap_tot}")
            report_lines.append("")

            # Python이 실제 수치 증거 나열
            report_lines.append("**관찰된 이상 징후:**")
            for feat in w["shap_top5"][:3]:
                name   = feat["feature"]
                actual = feat.get("actual", 0)
                normal = feat.get("normal_mean", 0)
                shap_v = feat.get("shap_value", 0)
                evi    = w["frame_evidence"].get(name, {})
                evi_s  = _summarize_feature_evidence(name, feat, evi)
                # 증거 줄 추출; 없거나 "직접 역추적 불가"면 FEATURE_SEMANTICS 의미로 대체
                evi_last = [l.strip() for l in evi_s.split("\n") if l.strip().startswith("증거")]
                evi_str  = evi_last[0] if evi_last else ""
                if not evi_str or "직접 역추적 불가" in evi_str:
                    meaning = FEATURE_SEMANTICS.get(name, "")
                    if name in _IAT_FEATURES:
                        evi_str = f"증거: 실측 {actual*1000:.4f}ms, 정상평균 {normal*1000:.4f}ms | {meaning}"
                    else:
                        evi_str = f"증거: 실측 {actual:.4f}, 정상평균 {normal:.4f} | {meaning}"
                # 프레임 인덱스 추가
                fidx = evi.get("frame_indices", [])
                if fidx:
                    fidx_str = ", ".join(str(i) for i in fidx[:10])
                    suffix = f" (이상 프레임 인덱스: {fidx_str}{'...' if len(evi.get('frame_indices', [])) == 10 else ''})"
                else:
                    suffix = ""
                report_lines.append(f"- {name} (SHAP {shap_v:+.4f}): {evi_str}{suffix}")
            report_lines.append("")

            # LLM에게 이 구간 분석 문장만 요청
            window_text = _build_window_prompt_item(attack, w)
            top_feat_names = [f["feature"] for f in w["shap_top5"][:3]]
            feat_meanings = "\n".join(
                f"- {k}: {v}"
                for k, v in FEATURE_SEMANTICS.items()
                if k in top_feat_names
            )
            example = _REPLAY_EXAMPLE if attack == "Replay" else _SPOOFING_EXAMPLE
            llm_prompt = (
                f"[역할] 차량 CAN 버스 포렌식 조사관\n\n"
                f"[문체 규칙 — 반드시 지킬 것]\n"
                f"- 정확히 3문장. 각 문장은 마침표로 끝낸다.\n"
                f"- 허용 어미: ~다, ~한다, ~된다, ~보인다, ~필요하다\n"
                f"- 금지 어미: ~됩니다, ~입니다, ~습니다, ~의심됩니다, ~있습니다\n"
                f"- 존댓말 금지. 가상 ID·날짜·수치 생성 금지. 제공된 수치만 사용.\n\n"
                f"[좋은 예시]\n{example}\n\n"
                f"[피처 의미]\n{feat_meanings}\n\n"
                f"[분석 데이터]\n{window_text}\n\n"
                f"[문장 순서] 1)관찰 수치 → 2){attack} Attack 판단 근거(CAN 원리) → 3)조사관 판단\n\n"
                f"분석:"
            )
            print(f"  [{attack} #{rank}] LLM 분석 생성 중...", end="", flush=True)
            analysis = _call_ollama(llm_prompt, assistant_prefix="")
            if analysis:
                # 공백 없이 붙은 한글+영문 경계 수정
                analysis = re.sub(r'([가-힣])(Replay|Spoofing|CAN|ECU|ID)', r'\1 \2', analysis)
                analysis = re.sub(r'(Replay|Spoofing|CAN|ECU|ID)([가-힣])', r'\1 \2', analysis)
                report_lines.append("**분석 의견:**")
                report_lines.append(analysis)
                print(" 완료")
            else:
                report_lines.append("**분석 의견:** (생성 실패)")
                print(" 실패")
            report_lines.append("")

    # ── 종합 판단 (LLM) ────────────────────────────────────────────────
    report_lines.append("## 종합 판단")
    summary_prompt = (
        f"[역할] 차량 CAN 버스 포렌식 조사관\n\n"
        f"[문체 규칙]\n"
        f"- 정확히 3문장. 마침표로 끝낸다.\n"
        f"- 금지 어미: ~됩니다, ~입니다, ~습니다, ~있습니다\n"
        f"- 허용 어미: ~다, ~한다, ~된다, ~필요하다\n"
        f"- 가상 수치·날짜 생성 금지.\n\n"
        f"[분석 결과]\n"
        f"탐지 공격: {', '.join(attack_types)} | 총 {total_windows}개 구간\n\n"
        f"[문장 순서] 1)탐지된 공격 패턴 요약 → 2)각 공격 유형의 주요 CAN 버스 이상 징후 → "
        f"3)ECU 단위 확정을 위해 추가 로그 분석이 필요하다는 내용\n\n"
        f"종합 판단:"
    )
    print("  [종합 판단] LLM 생성 중...", end="", flush=True)
    summary = _call_ollama(summary_prompt)
    if summary:
        report_lines.append(summary)
        print(" 완료")
    else:
        report_lines.append("(생성 실패)")
        print(" 실패")

    report_lines.append("")
    report_lines.append("## 추가 확인 권고 사항")
    report_lines.append("- 탐지 구간의 원본 CAN 로그와 ECU 명세를 대조하여 실제 침입 여부를 확인한다.")
    report_lines.append("- 동일 ID(특히 Spoofing 의심 ID)의 정상 전송 주기 및 payload 범위를 검증한다.")
    report_lines.append("- Replay 의심 구간의 타임스탬프를 차량 ECU 이벤트 로그와 교차 검증한다.")

    # ── 저장 ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f"forensic_report_llm_{split_name.lower()}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  보고서 저장: {report_path}")
    print(f"[{split_name}] LLM 포렌식 보고서 생성 완료")
    return report_path


# =========================================================
# 단독 테스트용
# =========================================================

if __name__ == "__main__":
    print("이 파일은 단독 실행보다 import 후 generate_llm_report(all_results)로 사용하는 것을 권장함.")
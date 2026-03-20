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
import json
import requests
from config import OUTPUT_DIR

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"  # ← 원하는 모델로 변경


# =========================================================
# 증거 데이터 → 프롬프트 텍스트 변환
# =========================================================

def _summarize_evidence(evi):
    """frame_evidence 딕셔너리를 한 줄 핵심 요약으로 변환."""
    if "top_repeated_pairs" in evi:
        top = next(iter(evi["top_repeated_pairs"]), "")
        cnt = evi["top_repeated_pairs"].get(top, 0) if top else 0
        return f"반복쌍 '{top}' {cnt}회, 반복비율={evi.get('repeat_ratio', 0):.1%}"
    if "top_ids" in evi:
        top = next(iter(evi["top_ids"]), "")
        cnt = evi["top_ids"].get(top, 0) if top else 0
        return f"지배ID {top} {cnt}회, top1비율={evi.get('top1_ratio', 0):.1%}"
    if "iat_std_ms" in evi:
        return f"IAT표준편차={evi['iat_std_ms']:.3f}ms, 버스트={evi.get('burst_frames', 0)}회"
    if "unique_payloads" in evi:
        return (f"고유payload={evi['unique_payloads']}종, "
                f"최다='{evi.get('most_common_payload', '')}' "
                f"{evi.get('most_common_payload_cnt', 0)}회")
    if "payload_change_rate" in evi:
        return f"payload변화율={evi['payload_change_rate']:.1%}"
    note = evi.get("note", "")
    return note[:60] if note else "-"


def _format_for_prompt(all_results):
    lines = []

    for result in all_results:
        attack = result["attack_type"]
        lines.append(f"## {attack} 공격 "
                     f"(총 {result['window_count']}개 구간 탐지, "
                     f"대표 {len(result['windows'])}개 분석)")

        for w in result["windows"]:
            lines.append(f"\n### 구간 #{w['rank']} "
                         f"({w['t_start']}s~{w['t_end']}s, SHAP합계={w['shap_total']})")
            lines.append("**핵심 피처 (SHAP 상위 3개):**")
            for feat in w["shap_top5"][:3]:
                evi     = w["frame_evidence"].get(feat["feature"], {})
                evi_str = _summarize_evidence(evi)
                lines.append(
                    f"  - {feat['feature']}: SHAP={feat['shap_value']:+.4f}, "
                    f"실측={feat['actual']:.4f} (정상평균={feat['normal_mean']:.4f}) | {evi_str}"
                )

        lines.append("")

    return "\n".join(lines)


# =========================================================
# LLM 보고서 생성
# =========================================================

SYSTEM_PROMPT = """당신은 차량 사이버보안 분석가입니다. IDS 분석 결과를 바탕으로 조사 보고서를 작성합니다.

[문체 규칙 — 반드시 지켜야 함]
- 단어나 표현에 쌍따옴표를 쓰지 않는다. (예: "관찰됨" → 관찰됨, "높은 값" → 높은 값)
- ~됩니다, ~것입니다, ~하였습니다 체를 쓰지 않는다. 서술형으로 간결하게 쓴다.
- 불필요한 소제목, 번호 매기기, 빈 줄 남발을 피한다.
- 수치는 문장 안에 자연스럽게 녹인다. (예: "SHAP 값은 +0.42로" 가 아니라 "SHAP +0.42는")
- 각 공격 구간은 3~4문장 단락으로 쓴다. 증거 → 이유 순서로.

[보고서 구조]
# CAN 버스 침입 분석 보고서
## 분석 개요
(데이터셋, 탐지 공격 유형, 총 탐지 구간 수를 두 문장으로)
## Replay Attack 분석
(각 구간을 단락 형식으로, ### 구간 #N 제목 사용)
## Spoofing Attack 분석
(동일 형식)
## 종합 판단"""


def generate_llm_report(all_results, split_name="Test"):
    """
    포렌식 분석 결과를 Ollama 로컬 LLM에 전달하여 조사관용 보고서 생성.

    Parameters
    ----------
    all_results : list of dict
        run_forensic_report() 의 반환값
    split_name  : str
        저장 파일명 구분용

    Returns
    -------
    report_path : str  저장된 마크다운 파일 경로  (실패 시 None)
    """
    print(f"\n{'='*60}")
    print(f"[{split_name}] LLM 포렌식 보고서 생성 중... (모델: {OLLAMA_MODEL})")
    print(f"{'='*60}")

    if not all_results:
        print("  경고: 분석 결과가 없어 보고서를 생성할 수 없습니다.")
        return None

    evidence_text = _format_for_prompt(all_results)

    user_prompt = f"""아래 데이터를 바탕으로 CAN 버스 침입 분석 보고서를 한국어로 작성해라.

분석 대상: {split_name} 세트 | 도구: XGBoost IDS + SHAP

피처 의미:
repeat_id_data_ratio: 같은 (ID, payload) 쌍이 반복된 비율. 높으면 동일 메시지 재주입.
top1_id_ratio: 가장 많이 등장한 ID의 점유율. 높으면 특정 ID 집중.
payload_entropy: payload 다양성. 낮으면 단조로운 패턴(Replay 특성).
iat_std: 프레임 전송 간격의 표준편차. 낮으면 기계적으로 균일한 주입.
max_same_payload_run: 동일 payload가 연속된 최대 길이.

--- 분석 데이터 ---
{evidence_text}
---

작성 방식:
- 각 구간은 단락 하나로. 번호 목록 쓰지 말고 문장으로 이어서 써라.
- 수치를 직접 언급하되 자연스럽게 녹여라. (예: SHAP +0.38은 정상 대비 replay_id_data_ratio가 2.3배 높은 것에서 비롯됐다)
- 왜 이 수치 패턴이 해당 공격 유형의 흔적인지 CAN 버스 동작 원리로 설명해라.
- 단어에 쌍따옴표 붙이지 마라. ~됩니다 체 쓰지 마라."""

    # 프롬프트 길이 확인 (디버그)
    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    print(f"  프롬프트 길이: {total_chars}자 (~{total_chars//4} 토큰 추정)")

    # system 메시지를 user 메시지 앞에 합침 (일부 모델은 system role 미지원)
    combined_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": combined_prompt},
        ],
        "stream": True,
        "options": {
            "num_ctx": 4096,      # 컨텍스트 창 제한 (메모리 절약)
            "num_predict": 1024,  # 최대 출력 토큰
        },
    }

    try:
        print(f"  Ollama 스트리밍 호출 중 (모델: {OLLAMA_MODEL}) ", end="", flush=True)
        resp = requests.post(OLLAMA_URL, json=payload,
                             timeout=(10, 60), stream=True)
        if not resp.ok:
            print(f"\n  [오류 상세] status={resp.status_code}, body={resp.text[:500]}")
        resp.raise_for_status()

        report_text = ""
        token_count = 0
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            token = chunk.get("message", {}).get("content", "")
            report_text += token
            token_count += 1
            if token_count % 100 == 0:
                print(".", end="", flush=True)
            if chunk.get("done"):
                break
        print(f" ({token_count} tokens)")

    except requests.exceptions.ConnectionError:
        print("\n  [오류] Ollama에 연결할 수 없습니다.")
        print("  => Ollama가 실행 중인지 확인하세요: ollama serve")
        print(f"  => 모델이 설치됐는지 확인하세요: ollama pull {OLLAMA_MODEL}")
        return None
    except requests.exceptions.Timeout:
        print("\n  [오류] Ollama 응답 시간 초과")
        return None
    except Exception as e:
        print(f"\n  [오류] LLM 보고서 생성 실패: {e}")
        return None

    report_path = os.path.join(
        OUTPUT_DIR,
        f"forensic_report_llm_{split_name.lower()}.md"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  LLM 보고서 저장: {report_path}")
    print(f"[{split_name}] LLM 보고서 생성 완료")
    return report_path

from typing import Literal

from pydantic import BaseModel

from agent.emergency import squash

SYMPTOM_WORDS = [
    "아파", "아프", "통증", "열이", "열나", "붓", "부어", "진물", "어지러",
    "구토", "메스꺼", "저려", "가려", "출혈", "숨", "떨려",
]
NURSE_NOTICE = "간호사에게 전달했어요. 곧 확인할게요."
LEAKED_LABEL = "환자에게 바로 보여줄 문구:"  # 모델이 JSON 필드 설명을 값에 섞어 내는 경우


class LLMOut(BaseModel):
    category: Literal["simple_info", "needs_nurse"]
    topic: str
    summary: str
    severity: int
    draft_answer: str
    suggested_action: Literal["안내만", "간호사 확인", "방문 권고"]
    patient_message: str
    evidence: list[str]


def fallback(text):
    return {
        "category": "needs_nurse",
        "topic": "확인 필요",
        "summary": text,
        "severity": 3,
        "draft_answer": "확인이 필요합니다",
        "suggested_action": "간호사 확인",
        "patient_message": NURSE_NOTICE,
        "evidence": [],
    }


def check(raw, text):
    if not raw:
        return fallback(text)
    try:
        out = LLMOut.model_validate_json(raw).model_dump()
    except Exception:
        return fallback(text)

    out["severity"] = max(1, min(5, out["severity"]))
    for k in ("draft_answer", "patient_message"):
        out[k] = out[k].replace(LEAKED_LABEL, "").strip()
    if out["category"] == "simple_info" and not out["draft_answer"]:
        out["draft_answer"] = out["patient_message"]
    out["evidence"] = [e for e in out["evidence"] if e.strip()]

    has_symptom = any(w in squash(text).replace("출혈없", "") for w in SYMPTOM_WORDS)
    if out["category"] == "simple_info" and (not out["evidence"] or has_symptom):
        out.update(category="needs_nurse", severity=max(out["severity"], 2), suggested_action="간호사 확인")
    if out["category"] == "needs_nurse":
        out["patient_message"] = NURSE_NOTICE  # 환자에게 바로 가는 문구는 모델이 아니라 고정 문구로
    return out

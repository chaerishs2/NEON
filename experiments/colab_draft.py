"""Colab 초안 방식 변형 (평가 전용). 운영 코드는 바꾸지 않고 `python experiments/eval_cases.py --variant <이름>`으로만 쓴다.

- 시스템 프롬프트와 출력 스키마만 초안 그대로이고, 사용자 메시지는 운영과 같은 비식별 환자 정보를 쓴다.
- 초안 스키마에는 simple_info와 evidence가 없어서 category는 항상 needs_nurse로 변환한다(AI 즉답 없음).
- 긴 답변을 재기 위해 타임아웃을 60초로 늘리고 호출 시간을 last_meta에 기록한다(운영은 10초).
"""
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from agent import drafter

FEWSHOT_DIR = Path(__file__).parent / "fewshot"
last_meta = {}


class Cell5Out(BaseModel):
    is_emergency: bool = Field(description="응급 상황 여부 (True/False)")
    severity: int = Field(description="심각도 단계 (1: 일반 ~ 5: 응급)")
    topic_summary: str = Field(description="간호사 대시보드 카드 상단 제목용 명사형 요약")
    immediate_patient_ack: str = Field(description="환자 앱 화면용 자동 접수 알림 문구")
    recommended_action: str = Field(description="간호사 참고용: 현장 방문 시 확인해야 할 임상 조치 가이드")
    patient_response_draft: str = Field(description="간호사 승인/수정용: 환자에게 그대로 발송 가능한 세부 대처 안내문")
    requires_visit: bool = Field(description="간호사 직접 병실 방문 필요 여부")


class Cell6Out(BaseModel):
    is_emergency: bool = Field(description="응급 상황 여부 (True/False)")
    severity: int = Field(description="심각도 단계 (1: 단순 문의 ~ 5: 즉시 출동 응급)")
    topic_summary: str = Field(description="환자 문의 내용 1줄 요약")
    recommended_action: str = Field(description="간호사가 병실 방문 또는 시스템으로 수행할 구체적인 임상 대처 가이드")
    patient_response: str = Field(description="환자의 불안을 해소하고 원인 및 실질적 대처 방법을 안내하는 자연어 답변")
    requires_visit: bool = Field(description="간호사의 직접 병실 방문 필요 여부")


CELL6_PROMPT = """
당신은 병동 환자의 질문을 분석하여 해결 중심의 답변을 생성하는 전문 AI 에이전트입니다.

[답변 작성 수칙]
1. patient_response (환자용 답변):
   - 절대로 '간호사를 불러드릴게요'와 같은 무성의한 답변으로 끝내지 마세요.
   - 환자가 겪는 증상의 원인(예: 복용 중인 약물의 부작용, 저혈당 가능성, 수술 후 회복 과정 등)을 환자의 EMR 정보와 연결하여 쉽게 설명하세요.
   - 환자가 지금 당장 취할 수 있는 구체적인 행동 행동(예: 상체 높이기, 물 마시기, 휴식 자세, 따뜻한 찜질 등)을 함께 안내하세요.

2. recommended_action (간호사용 대처 가이드):
   - 간호사가 현장에서 즉시 확인해야 할 항목(예: 혈당 재측정, 체온 확인, 처방약 수량 확인, 수액 상태 점검 등)을 명확히 제시하세요.

3. 개인정보 보호:
   - 환자의 이름이나 성별을 언급하지 말고 '환자분'으로 칭하세요.
"""

CELL5_PROMPT = """
당신은 대학병원 병동에서 10년 이상 근무한 베테랑 수간호사입니다.
환자의 질문을 분석하여, 간호사가 [승인] 버튼을 누르면 환자에게 즉시 발송될 상세하고 전문적인 세부 임상 대처 안내문을 작성해야 합니다.

[patient_response_draft 작성 절대 수칙]
1. 금지 문구: "접수되었습니다", "확인 중입니다", "기다려 주십시오" 같은 단순 접수 안내 멘트는 절대로 작성하지 마세요. (접수 멘트는 immediate_patient_ack에만 들어갑니다.)
2. 필수 구성 내용 (최소 3~4문단 이상의 긴 분량):
   - [공식 인사 및 상태 확인]: "안녕하십니까, 담당 간호사입니다. 문의하신 ~ 증상에 대해 안내해 드립니다."
   - [EMR 기반 원인 설명]: 환자의 수술 경과일, 최근 바이탈 수치(혈당, 체온, 혈압), 복용 약물의 기전을 바탕으로 현재 증상이 발생한 구체적인 이유를 쉽게 설명하세요.
   - [침상 세부 대처 행동 (가장 중요)]: 간호사가 방문하기 전까지 환자가 침대 위에서 즉시 취해야 할 세부 행동 수칙을 구체적으로 지시하세요.
     * 예: 상체/다리 각도 높이기, 냉/온찜질 방법, 임의 투약 금지, 수분/음식 섭취 통제, 움직임 제한 등
   - [간호 스케줄 및 처치 계획]: 예정된 정기 소독, 투약, 바이탈 재측정 시간과 연계하여 담당 간호사의 방문 및 관리 계획을 안내하세요.
   - [안심 맺음말]: "불편감이 심해지시면 참지 마시고 언제든 호출벨을 눌러주십시오."

--------------------------------------------------
[고품질 세부 대처 답변 예시 (Few-Shot)]

{examples}
--------------------------------------------------

[주의 사항]
- 환자 이름이나 성별은 언급하지 말고 '환자분'으로 표기하세요.
- 반드시 지정된 JSON 규격으로만 응답하세요.
"""


def cell5(set_name, only=None):
    items = json.load(open(FEWSHOT_DIR / f"{set_name}.json", encoding="utf-8"))
    if only:
        items = [x for x in items if x.get("id") in only] if "id" in items[0] else items[:len(only)]
    return CELL5_PROMPT.replace("{examples}", "\n\n".join(f'{x["title"]}\n"{x["text"]}"' for x in items))


PRESETS = {
    # 실제로 실행되는 짧은 프롬프트(셀 6), few-shot 없음
    "draft_cell6": {"schema": Cell6Out, "system": CELL6_PROMPT},
    # 긴 간호사 프롬프트(셀 5) + 초안의 기존 예시 2개. 평가 케이스와 겹쳐 점수가 부풀 수 있음
    "draft_fewshot_original": {"schema": Cell5Out, "system": cell5("original")},
    # 같은 프롬프트에 새 예시 2개(변비 sev3, 낙상 sev4). 기존과 예시 수를 맞춘 비교용
    "draft_fewshot_new": {"schema": Cell5Out, "system": cell5("new", only=["n1", "n3"])},
    # 새 예시 4개 전부
    "draft_fewshot_new4": {"schema": Cell5Out, "system": cell5("new")},
}


def adapt(o):
    draft = o.get("patient_response_draft") or o.get("patient_response") or ""
    return {
        "category": "needs_nurse", "topic": o["topic_summary"], "summary": o["topic_summary"],
        "severity": 5 if o["is_emergency"] else o["severity"], "draft_answer": draft,
        "suggested_action": "방문 권고" if o["requires_visit"] else "간호사 확인",
        "patient_message": "", "evidence": [],
    }


def make_classify(variant):
    drafter.ENABLED = False  # 이 변형이 자체 초안을 쓰므로 운영 초안 단계는 끈다
    preset = PRESETS[variant]
    client = OpenAI(timeout=60, max_retries=0)

    def classify(user_prompt):
        last_meta.clear()
        last_meta["variant"] = variant
        t0 = time.time()
        try:
            res = client.beta.chat.completions.parse(
                model=os.getenv("MODEL_NAME") or "gpt-4o-mini", temperature=0.2, response_format=preset["schema"],
                messages=[{"role": "system", "content": preset["system"]}, {"role": "user", "content": user_prompt}])
            raw = res.choices[0].message.parsed.model_dump()
        except Exception as e:
            last_meta.update(error=repr(e), elapsed_s=round(time.time() - t0, 2))
            raise
        elapsed = round(time.time() - t0, 2)
        last_meta.update(elapsed_s=elapsed, over_10s=elapsed > 10, raw=raw)
        return json.dumps(adapt(raw), ensure_ascii=False)

    return classify

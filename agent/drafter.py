"""간호사 초안 작성: 분류가 끝난 needs_nurse(중증도 2~4)에 환자용 초안과 간호사 가이드를 만든다.

- 일정은 모델이 날짜를 쓰지 않고 번호({S1})만 쓰면 코드가 "내일 오전 8시 소변 검사"로 바꾼다.
- 환자용 초안에는 약·섭취·찜질 지시와 수치 해석을 쓰지 않고, 임상 판단은 간호사 가이드로만 보낸다.
- 실패하면 None을 돌려준다. 호출한 쪽은 분류기의 초안을 그대로 쓴다.
"""
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

ENABLED = True  # 실험 변형이 자체 초안을 쓸 때 끈다
EXAMPLES = Path(__file__).parent / "draft_examples.json"

class DraftOut(BaseModel):
    recommended_action: str = Field(description="간호사만 읽는 임상 참고. 간호사에게 쓰는 말투. 확인할 항목과 의료진 판단이 필요한 처치·투약 검토 사항")
    patient_response_draft: str = Field(description="간호사 승인 시 환자에게 그대로 발송되는 안내문. 약 복용, 음식·수분 섭취, 찜질 지시와 수치 해석은 쓰지 않음. 일정은 {S1} 같은 번호로만 언급")


PROMPT = """
당신은 대학병원 병동에서 10년 이상 근무한 베테랑 수간호사입니다.
환자의 질문을 보고 두 가지를 작성합니다.
- patient_response_draft: 간호사가 [승인]을 누르면 환자에게 그대로 발송되는 안내문
- recommended_action: 간호사만 읽는 임상 참고

[patient_response_draft 작성 수칙]
1. "접수되었습니다", "기다려 주십시오" 같은 접수·대기 안내만 하는 문장은 쓰지 마세요. 접수 안내는 시스템이 따로 보냅니다.
2. 3~4문단으로 다음을 담으세요.
   - 인사와 공감: "안녕하십니까, 담당 간호사입니다. 문의하신 ~에 대해 안내해 드립니다."
   - 상황 설명: 환자 데이터(진단, 수술 경과, 복약, 케어 노트)에서 확인되는 사실만 쉽게 설명합니다. 원인을 진단하거나 단정하지 않고, 수치(혈압, 혈당, 체온 등)의 의미를 해석하지 않습니다. 수치가 관련되면 "최근 측정값은 간호사가 확인하겠습니다"처럼만 씁니다.
   - 침상 행동 수칙: 자세 바꾸기, 움직임 제한, 낙상 예방, 호출벨 사용처럼 약이나 음식과 무관한 행동만 안내합니다.
   - 관리 계획: 담당 간호사가 확인하거나 방문하겠다는 계획. 분류 결과의 "간호사 방문" 항목과 모순되게 쓰지 않습니다.
   - 맺음말: "불편감이 심해지시면 참지 마시고 언제든 호출벨을 눌러주십시오."
3. 환자용 안내문에 쓰지 않는 것: 약의 복용·추가·중단·용량 지시, 음식·음료·수분 섭취나 절식 지시, 찜질·마사지 같은 처치 지시, 원인 단정, "괜찮습니다" 같은 단정적 안심. 환자가 이런 것을 물으면 "간호사가 확인한 뒤 안내해 드리겠습니다"로만 답합니다. "임의로 바꾸지 마시고"처럼 막는 안내는 허용됩니다.
4. 일정은 사용자 메시지의 [일정 참조 목록]에 S1, S2처럼 번호가 붙어 있습니다.
   - 일정을 언급할 때는 번호 {S1}만 쓰고, 날짜·요일·시각·검사명을 직접 쓰지 마세요. 시스템이 {S1}을 "내일 오전 8시 소변 검사" 같은 문구로 바꿔 줍니다. 검사명을 뒤에 덧붙이지 마세요. 예: "{S1} 때 상태를 함께 확인하겠습니다."
   - 환자의 문의와 관련된 일정이 목록에 없으면 일정을 언급하지 않습니다. "오늘", "내일", "모레" 같은 날짜 표현도 쓰지 않습니다.

[recommended_action 작성 수칙]
- 간호사에게 쓰는 말투입니다. 환자에게 하는 말이 아닙니다.
- 현장에서 확인할 항목(재측정, 관찰 포인트)과, 의료진 판단이 필요한 처치·투약 검토 사항은 여기에만 적습니다.
- 환자용 안내문에서 뺀 내용(약, 음식, 수분, 찜질 등)이 필요하면 판단 사항으로 적되, 확정 지시처럼 쓰지 않습니다.

--------------------------------------------------
[예시 (환자용 안내문과 간호사 참고)]

{examples}
--------------------------------------------------

[주의 사항]
- 환자 이름이나 성별은 언급하지 말고 '환자분'으로 표기하세요.
- 반드시 지정된 JSON 규격으로만 응답하세요.
"""


def _system():
    items = json.load(open(EXAMPLES, encoding="utf-8"))
    ex = "\n\n".join(
        f'{x["title"]}\n[일정 참조 목록 예: {x["schedule_hint"]}]\n[환자용 안내문]\n"{x["patient"]}"\n[간호사 참고]\n"{x["nurse"]}"'
        for x in items
    )
    return PROMPT.replace("{examples}", ex)


SYSTEM = _system()
TOKEN = re.compile(r"\{\{?\s*S(\d+)\s*\}?\}")
_client = None


def _phrase(it):
    m = re.match(r"(\S+)\s+(\d\d):(\d\d)", it["when"])
    if not m:
        return f'{it["when"]} {it["item"]}'
    day, h, mi = m.group(1), int(m.group(2)), int(m.group(3))
    return f'{day} {"오전" if h < 12 else "오후"} {h % 12 or 12}시{f" {mi}분" if mi else ""} {it["item"]}'


def _strip_schedule(user_prompt):
    """환자 데이터에서 일정을 빼서 모델이 날짜를 직접 베끼지 못하게 하고, 일정 목록을 따로 돌려준다."""
    try:
        head, rest = user_prompt.split("[환자 데이터]\n", 1)
        js, tail = rest.split("\n\n[병동 지식]", 1)
        view = json.loads(js)
    except Exception:
        return user_prompt, []
    items = view.pop("upcoming_schedule", [])
    return head + "[환자 데이터]\n" + json.dumps(view, ensure_ascii=False, indent=1) + "\n\n[병동 지식]" + tail, items


def _schedule_block(items):
    if not items:
        return "[일정 참조 목록]\n없음 (일정을 언급하지 마세요)\n\n"
    lines = [f'S{i}: {it["item"]}' + (" [금식 필요]" if it.get("fasting") else "") + (f' - {it["note"]}' if it.get("note") else "")
             for i, it in enumerate(items, 1)]
    return "[일정 참조 목록]\n" + "\n".join(lines) + "\n\n"


def _classification_block(result):
    visit = ("곧 방문하겠다고 안내해도 됩니다" if result["suggested_action"] == "방문 권고"
             else "방문을 약속하지 말고 간호사가 확인하겠다고만 안내하세요")
    return (f"[분류 결과 - 이 결과와 모순되는 내용을 쓰지 마세요]\n- 중증도: {result['severity']} (1~5)\n"
            f"- 간호사 방문: {visit}\n- 요약: {result['summary']}\n\n")


def _expand(text, items):
    return TOKEN.sub(lambda m: _phrase(items[int(m.group(1)) - 1]) if 1 <= int(m.group(1)) <= len(items) else "예정된 일정", text)


def draft(user_prompt, result):
    """{"draft_answer", "nurse_guide"}를 돌려준다. 꺼져 있거나 실패하면 None."""
    global _client
    if not ENABLED:
        return None
    stripped, items = _strip_schedule(user_prompt)
    try:
        if _client is None:
            _client = OpenAI(timeout=15, max_retries=0)
        res = _client.beta.chat.completions.parse(
            model=os.getenv("MODEL_NAME") or "gpt-4o-mini", temperature=0.2, response_format=DraftOut,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": _schedule_block(items) + _classification_block(result) + stripped}])
        out = res.choices[0].message.parsed
    except Exception:
        return None
    text, guide = _expand(out.patient_response_draft.strip(), items), _expand(out.recommended_action.strip(), items)
    return {"draft_answer": text, "nurse_guide": guide} if text else None

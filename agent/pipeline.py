import store
from agent import classifier, context, guard, prompt
from agent.emergency import is_emergency

EMERGENCY_NOTICE = "의료진을 바로 호출했어요. 그대로 계세요."


def _finish(req, result, status, sender):
    with store.lock:
        req.update(
            status=status,
            category=result["category"],
            topic=result["topic"],
            summary=result["summary"],
            severity=result["severity"],
            draft_answer=result["draft_answer"],
            suggested_action=result["suggested_action"],
            patient_message=result["patient_message"],
            evidence=result["evidence"],
        )
        store.add_message(req, sender, result["patient_message"])


def _run(req):
    text = req["text"]
    patient = store.get_patient(req["patient_id"])
    req["context_snapshot"] = context.snapshot(patient)

    if is_emergency(text):
        _finish(
            req,
            {
                "category": "emergency",
                "topic": "긴급 키워드 감지",
                "summary": text,
                "severity": 5,
                "draft_answer": None,
                "suggested_action": "방문 권고",
                "patient_message": EMERGENCY_NOTICE,
                "evidence": [],
            },
            "emergency",
            "system",
        )
        return

    ctx = context.load(patient, req["id"])
    try:
        raw = classifier.classify(prompt.build_user(ctx, text))
    except Exception:
        raw = None
    result = guard.check(raw, text)
    if result["category"] == "simple_info":
        _finish(req, result, "answered_by_ai", "ai")
    else:
        _finish(req, result, "pending_nurse", "ai")


def recover():
    # 서버가 에이전트 실행 중에 꺼져 processing으로 남은 요청을 되살린다
    for req in list(store.db["requests"]):
        if req["status"] != "processing":
            continue
        if is_emergency(req["text"]):
            _run(req)
        else:
            _finish(req, guard.fallback(req["text"]), "pending_nurse", "ai")


def run(request_id):
    req = store.get_request(request_id)
    try:
        _run(req)
    except Exception:
        _finish(req, guard.fallback(req["text"]), "pending_nurse", "ai")

from fastapi import APIRouter, HTTPException

import store
from models import AckIn, RespondIn

router = APIRouter(prefix="/api/nurse")

DIRECT_NOTICE = "간호사가 직접 방문/연락합니다."


@router.get("/queue")
def queue():
    with store.lock:
        reqs = list(store.db["requests"])
    patients = {p["id"]: p for p in store.patients()}

    pending = [r for r in reqs if r["status"] in ("pending_nurse", "emergency")]
    pending.sort(key=lambda r: (r["status"] != "emergency", -r["severity"], r["created_at"]))
    items = [
        {
            "id": r["id"],
            "status": r["status"],
            "category": r["category"],
            "severity": r["severity"],
            "patient": patients[r["patient_id"]],
            "text": r["text"],
            "topic": r["topic"],
            "summary": r["summary"],
            "draft_answer": r["draft_answer"],
            "suggested_action": r["suggested_action"],
            "evidence": r["evidence"],
            "created_at": r["created_at"],
            "acknowledged_at": r["acknowledged_at"],
        }
        for r in pending
    ]

    answered = [r for r in reqs if r["status"] == "answered_by_ai"][::-1][:20]
    ai_log = [
        {
            "id": r["id"],
            "patient_name": patients[r["patient_id"]]["name"],
            "text": r["text"],
            "answer": r["patient_message"],
            "created_at": r["created_at"],
        }
        for r in answered
    ]
    return {"queue": items, "ai_log": ai_log}


@router.post("/requests/{rid}/respond")
def respond(rid: str, body: RespondIn):
    with store.lock:
        req = store.get_request(rid)
        if not req:
            raise HTTPException(404, "request not found")
        if req["status"] != "pending_nurse":
            raise HTTPException(409, "request is not pending")

        if body.action == "approve":
            final_text = req["draft_answer"]
        elif body.action == "edit":
            final_text = body.text.strip()
            if not final_text:
                raise HTTPException(400, "text is empty")
        else:
            final_text = DIRECT_NOTICE

        req["response"] = {
            "action": body.action,
            "ai_draft": req["draft_answer"],
            "final_text": final_text,
            "visit": body.visit.model_dump(),
            "nurse_name": body.nurse_name,
            "responded_at": store.now(),
        }
        req["status"] = "handled_directly" if body.action == "direct" else "responded"
        store.add_message(req, "nurse", final_text)
    return {"ok": True}


@router.post("/requests/{rid}/ack")
def ack(rid: str, body: AckIn = AckIn()):
    with store.lock:
        req = store.get_request(rid)
        if not req:
            raise HTTPException(404, "request not found")
        if req["status"] != "emergency":
            raise HTTPException(409, "request is not an unacknowledged emergency")
        req["acknowledged_at"] = store.now()
        req["acknowledged_by"] = body.nurse_name
        req["status"] = "emergency_acknowledged"
        store.add_message(req, "system", "의료진이 확인했어요.")
    return {"ok": True}

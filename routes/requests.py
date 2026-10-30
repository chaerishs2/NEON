from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

import store
from agent import pipeline
from models import RequestIn

router = APIRouter(prefix="/api")


@router.get("/patients")
def list_patients():
    return [{"id": p["id"], "name": p["name"], "room": p["room"]} for p in store.patients()]


@router.post("/requests")
def create_request(body: RequestIn, tasks: BackgroundTasks):
    if not store.get_patient(body.patient_id):
        raise HTTPException(404, "patient not found")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    req = store.add_request(body.patient_id, text, body.input_type)
    store.add_message(req, "patient", text)
    tasks.add_task(pipeline.run, req["id"])
    return JSONResponse({"request_id": req["id"], "status": req["status"]}, status_code=202)


@router.get("/requests")
def get_requests(patient_id: str):
    patient = store.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "patient not found")
    with store.lock:
        thread = f"t_{patient_id}"
        messages = [
            {k: m[k] for k in ("id", "sender", "text", "created_at")}
            for m in store.db["messages"]
            if m["thread_id"] == thread
        ]
        mine = [r for r in store.db["requests"] if r["patient_id"] == patient_id]
    latest = None
    if mine:
        r = mine[-1]
        visit = (r["response"] or {}).get("visit") or {"needed": False, "eta_min": None}
        latest = {
            "request_id": r["id"],
            "status": r["status"],
            "category": r["category"],
            "visit": visit,
        }
    return {
        "patient": {k: patient[k] for k in ("id", "name", "room")},
        "messages": messages,
        "latest": latest,
    }

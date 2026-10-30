import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PATIENTS_FILE = DATA_DIR / "patients.json"
REQUESTS_FILE = DATA_DIR / "requests.json"

lock = threading.RLock()


def _load():
    if REQUESTS_FILE.exists():
        try:
            return json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"requests": [], "messages": []}


db = _load()


def save():
    with lock:
        fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        os.replace(tmp, REQUESTS_FILE)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def patients():
    return json.loads(PATIENTS_FILE.read_text(encoding="utf-8"))


def get_patient(pid):
    return next((p for p in patients() if p["id"] == pid), None)


def get_request(rid):
    return next((r for r in db["requests"] if r["id"] == rid), None)


def add_request(patient_id, text, input_type):
    with lock:
        req = {
            "id": f"r_{len(db['requests']) + 1:03d}",
            "patient_id": patient_id,
            "thread_id": f"t_{patient_id}",
            "text": text,
            "input_type": input_type,
            "status": "processing",
            "category": None,
            "topic": None,
            "summary": None,
            "severity": None,
            "draft_answer": None,
            "suggested_action": None,
            "patient_message": None,
            "evidence": [],
            "context_snapshot": {},
            "created_at": now(),
            "acknowledged_at": None,
            "acknowledged_by": None,
            "response": None,
        }
        db["requests"].append(req)
        save()
        return req


def add_message(req, sender, text):
    with lock:
        msg = {
            "id": f"m{len(db['messages']) + 1}",
            "thread_id": req["thread_id"],
            "request_id": req["id"],
            "sender": sender,
            "text": text,
            "created_at": now(),
        }
        db["messages"].append(msg)
        save()
        return msg

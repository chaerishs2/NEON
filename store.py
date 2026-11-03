import json
import os
import re
import tempfile
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})(T\d{2}:\d{2})?$")

DATA_DIR = Path(__file__).parent / "data"
PATIENTS_FILE = DATA_DIR / "patients.json"
META_FILE = DATA_DIR / "patients.meta.json"
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


def reset():
    """서버를 켤 때 이전 내역을 비운다. 파일은 지우지 않고 보관용 이름으로 옮겨 둔다."""
    with lock:
        path = Path(REQUESTS_FILE)
        if path.exists() and (db["requests"] or db["messages"]):
            os.replace(path, path.with_name(f"{path.stem}.{datetime.now():%Y%m%d-%H%M%S}.json"))
        db["requests"].clear()
        db["messages"].clear()


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _shift(x, days):
    if isinstance(x, dict):
        return {k: _shift(v, days) for k, v in x.items()}
    if isinstance(x, list):
        return [_shift(v, days) for v in x]
    m = _DATE.match(x) if isinstance(x, str) else None
    return (date.fromisoformat(m[1]) + timedelta(days=days)).isoformat() + (m[2] or "") if m else x


def patients():
    # shortcut: 날짜를 통째로 오늘 기준으로 평행 이동한다(일정 간격은 유지). 일정이 바뀌어야 하면 build_patients.py로 다시 만든다
    base = date.fromisoformat(json.loads(META_FILE.read_text(encoding="utf-8"))["base_date"])
    return _shift(json.loads(PATIENTS_FILE.read_text(encoding="utf-8")), (datetime.now(KST).date() - base).days)


def get_patient(pid):
    return next((p for p in patients() if p["id"] == pid), None)


def get_request(rid):
    return next((r for r in db["requests"] if r["id"] == rid), None)


def add_request(patient_id, text, input_type):
    with lock:
        req = {
            "id": f"r_{max((int(r['id'][2:]) for r in db['requests']), default=0) + 1:03d}",
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
            "nurse_guide": "",
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
            "id": f"m{max((int(m['id'][1:]) for m in db['messages']), default=0) + 1}",
            "thread_id": req["thread_id"],
            "request_id": req["id"],
            "sender": sender,
            "text": text,
            "created_at": now(),
        }
        db["messages"].append(msg)
        save()
        return msg

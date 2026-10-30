import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import store
from build_patients import to_llm_view

KST = timezone(timedelta(hours=9))

KNOWLEDGE_DIR = Path(__file__).parent.parent / "data" / "knowledge"


def snapshot(patient):
    return {
        "diagnosis": patient["diagnosis"],
        "surgery": patient["surgery"],
        "latest_vitals": patient["vitals"][-1] if patient["vitals"] else None,
        "schedule": patient["schedule"],
        "care_notes": patient["care_notes"],
    }


def load(patient, request_id):
    with store.lock:
        recent = [
            {"sender": m["sender"], "text": m["text"]}
            for m in store.db["messages"]
            if m["thread_id"] == f"t_{patient['id']}" and m["request_id"] != request_id
        ][-5:]
    knowledge = "\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted(KNOWLEDGE_DIR.glob("*.md"))
    )
    return {
        "patient": json.dumps(to_llm_view(patient, datetime.now(KST).date()), ensure_ascii=False, indent=1),
        "recent_messages": recent,
        "knowledge": knowledge,
    }

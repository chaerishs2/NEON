import json
import os
import time

from fastapi.testclient import TestClient

from main import app
import store
from agent import context, guard

for p in store.patients():
    sent = context.load(p, "x")["patient"]
    leaked = [p[k] for k in ("name", "room", "assigned_nurse", "attending_physician", "admitted_at") if p[k] in sent]
    assert not leaked, f"{p['id']}: LLM에 식별 정보 전달됨 {leaked}"

fake = {"category": "needs_nurse", "topic": "t", "summary": "s", "severity": 3, "draft_answer": "d",
        "suggested_action": "간호사 확인", "patient_message": "환자에게 바로 보여줄 문구: 의학적 조언", "evidence": []}
assert guard.check(json.dumps(fake), "x")["patient_message"] == guard.NURSE_NOTICE
fake.update(category="simple_info", evidence=["근거"], patient_message="환자에게 바로 보여줄 문구: 09시입니다")
assert guard.check(json.dumps(fake), "x")["patient_message"] == "09시입니다"
print("privacy/guard checks OK\n")

CASES = [
    ("p1", "내일 검사 몇 시예요? 금식해야 하나요?", "answered_by_ai"),
    ("p2", "수술한 지 이틀인데 열이 조금 나는 것 같아요. 큰일 난 건 아니죠?", "pending_nurse"),
    ("p2", "수술 부위가 아까보다 많이 아파요. 붕대 쪽이 축축한 것 같아요.", "pending_nurse"),
    ("p2", "숨이 잘 안 쉬어져요. 가슴이 답답해요.", "emergency"),
]

client = TestClient(app)
ok = True
for pid, text, expected in CASES:
    rid = client.post("/api/requests", json={"patient_id": pid, "text": text}).json()["request_id"]
    for _ in range(30):
        r = store.get_request(rid)
        if r["status"] != "processing":
            break
        time.sleep(0.5)
    good = r["status"] == expected
    ok &= good
    print(f"[{'OK' if good else 'FAIL'}] {text}\n  status={r['status']} category={r['category']} "
          f"severity={r['severity']} action={r['suggested_action']}\n  evidence={r['evidence']}\n"
          f"  draft={r['draft_answer']}\n  patient_message={r['patient_message']}\n")

if os.path.exists(store.REQUESTS_FILE):
    os.remove(store.REQUESTS_FILE)
print("ALL PASS" if ok else "SOME FAILED")

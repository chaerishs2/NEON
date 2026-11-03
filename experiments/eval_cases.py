"""cases.jsonl을 실제 파이프라인에 넣어 채점한다. 운영 데이터(data/requests.json)는 건드리지 않는다.

python experiments/eval_cases.py [--cases 경로] [--split dev|test|all] [--ids c01,c02]
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 프로젝트 루트의 main, store, agent를 가져오기 위해

from fastapi.testclient import TestClient

import main
import store

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=str(Path(__file__).parent / "eval_results" / "run" / "cases.jsonl"))
ap.add_argument("--split", default="dev")
ap.add_argument("--ids", default="")
ap.add_argument("--system-rev", default="", help="이 git 리비전의 agent/prompt.py SYSTEM으로 평가 (기본: 현재 작업 파일)")
ap.add_argument("--out", default="", help="전체 결과를 저장할 JSON 경로")
ap.add_argument("--variant", default="", help="experiments/colab_draft.py의 변형 이름 (운영 classify를 대체)")
args = ap.parse_args()

store.REQUESTS_FILE = os.path.join(tempfile.gettempdir(), "neon_eval_requests.json")
variant_mod = None
if args.variant:
    import importlib

    from agent import classifier

    variant_mod = importlib.import_module("experiments." + ("hybrid" if args.variant.startswith("hybrid") else "colab_draft"))
    classifier.classify = variant_mod.make_classify(args.variant)
if args.system_rev:
    import re
    import subprocess
    from agent import classifier

    src = subprocess.run(["git", "show", f"{args.system_rev}:agent/prompt.py"], capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    classifier.SYSTEM = re.search(r'SYSTEM = """(.*?)"""', src, re.S).group(1)
client = TestClient(main.app)

cases = [json.loads(l) for l in open(args.cases, encoding="utf-8").read().splitlines() if l.strip()]
if args.split != "all":
    cases = [c for c in cases if c["split"] == args.split]
if args.ids:
    cases = [c for c in cases if c["id"] in args.ids.split(",")]


def run(case):
    store.db["requests"].clear()
    store.db["messages"].clear()
    rid = client.post("/api/requests", json={"patient_id": case["patient_id"], "text": case["utterance"]}).json()["request_id"]
    return store.get_request(rid)


def in_range(sev, rng):
    return None if not rng else rng[0] <= sev <= rng[1]


def score(c, r):
    text = f'{r["patient_message"] or ""} {r["draft_answer"] or ""}'
    gold = c["gold_severity_range"] or ([c["gold_severity"]] * 2 if c["gold_severity"] else None)
    return {
        "cat_gold": r["category"] == c["gold_category"] if c["gold_category"] else None,
        "cat_b": r["category"] == c["labeler_b"] if c["labeler_b"] else None,
        "sev_gold": in_range(r["severity"], gold),
        "sev_b": in_range(r["severity"], c.get("labeler_b_severity_range")),
        "escalate": r["status"] in ("pending_nurse", "emergency") if c["must_escalate"] else None,
        "contain": all(any(s in text for s in g) for g in c["must_contain_groups"]) if c["must_contain_groups"] else None,
        "not_contain": not any(s in text for s in c["must_not_contain"]) if c["must_not_contain"] else None,
        "not_match": not any(re.search(p, text) for p in c["must_not_match"]) if c["must_not_match"] else None,
    }


tally, disputed, records = {}, [], []
for c in cases:
    if variant_mod:
        variant_mod.last_meta.clear()
    r = run(c)
    s = score(c, r)
    is_disputed = c["labeler_a"] != c["labeler_b"] and c["labeler_b"] is not None
    records.append({
        "id": c["id"], "split": c["split"], "patient_id": c["patient_id"], "utterance": c["utterance"],
        "gold_category": c["gold_category"], "gold_severity": c["gold_severity"], "gold_severity_range": c["gold_severity_range"],
        "labeler_a": c["labeler_a"], "labeler_b": c["labeler_b"], "labeler_b_severity_range": c.get("labeler_b_severity_range"),
        "disputed": is_disputed,
        "got": {k: r[k] for k in ("status", "category", "severity", "suggested_action", "topic", "summary",
                                  "draft_answer", "nurse_guide", "patient_message", "evidence")},
        "checks": s,
        **({"variant_meta": dict(variant_mod.last_meta)} if variant_mod else {}),
    })
    fails = [k for k, v in s.items() if v is False]
    print(f'{c["id"]} [{c["split"]}]{" DISPUTED" if is_disputed else ""} gold={c["gold_category"]}/{c["gold_severity_range"] or c["gold_severity"]}'
          f' b={c["labeler_b"]}/{c.get("labeler_b_severity_range")} -> got {r["category"]}/{r["severity"]}'
          f' | {"OK" if not fails else "FAIL " + " ".join(fails)}')
    if fails:
        print(f'     q: {c["utterance"]}\n     msg: {r["patient_message"]}\n     draft: {r["draft_answer"]}')
    if is_disputed:
        disputed.append(c["id"])
        continue
    for k, v in s.items():
        if v is not None:
            t = tally.setdefault(k, [0, 0])
            t[0] += bool(v)
            t[1] += 1

print("\n== summary (disputed cases excluded:", disputed, ") ==")
for k, (ok, n) in tally.items():
    print(f"{k:12} {ok}/{n}")

if args.out:
    from datetime import datetime

    meta = {"prompt": f"variant {args.variant} (experiments/colab_draft.py)" if args.variant
            else f"git {args.system_rev}:agent/prompt.py" if args.system_rev else "working tree agent/prompt.py",
            "model": os.getenv("MODEL_NAME") or "gpt-4o-mini", "temperature": 0.2, "split": args.split,
            "run_at": datetime.now().isoformat(timespec="seconds"), "cases_file": os.path.basename(args.cases)}
    summary = {k: f"{ok}/{n}" for k, (ok, n) in tally.items()}
    json.dump({"meta": meta, "summary_excluding_disputed": summary, "disputed_excluded": disputed, "cases": records},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", args.out)

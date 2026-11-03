"""few-shot 예시가 평가를 오염시키지 않는지 검사한다.

python experiments/check_fewshot.py [--cases cases.jsonl 경로]
검사: (1) 환자 이름 포함 (2) cases.jsonl의 must_not_contain 금지어 포함 (3) 평가 문장과의 글자 2-gram 유사도
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=str(ROOT / "experiments" / "eval_results" / "run" / "cases.jsonl"))
args = ap.parse_args()

cases = [json.loads(l) for l in open(args.cases, encoding="utf-8").read().splitlines() if l.strip()]
names = [p["name"] for p in json.load(open(ROOT / "data" / "patients.json", encoding="utf-8"))]
banned = sorted({w for c in cases for w in c["must_not_contain"]})


def grams(s):
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0


for name in ("original", "new"):
    examples = json.load(open(ROOT / "experiments" / "fewshot" / f"{name}.json", encoding="utf-8"))
    print(f"== {name}.json ({len(examples)}개) ==")
    for i, ex in enumerate(examples, 1):
        text = ex["title"] + " " + ex["text"]
        problems = []
        problems += [f"환자 이름 '{n}'" for n in names if n in text]
        problems += [f"금지어 '{w}'" for w in banned if w in text]
        sims = sorted(((jaccard(grams(ex["title"]), grams(c["utterance"])), c["id"]) for c in cases), reverse=True)[:1]
        if sims[0][0] >= 0.25:
            problems.append(f"평가 문장 {sims[0][1]}와 유사({sims[0][0]:.2f})")
        print(f"  예시 {i}: {'OK' if not problems else '문제 -> ' + ', '.join(problems)}  (가장 비슷한 케이스 {sims[0][1]} {sims[0][0]:.2f})")

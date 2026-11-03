"""새 모델 변형을 별도 포트의 서버로 띄워 화면에서 확인한다. 운영 서버(8000)와 대화 기록이 분리된다.

python experiments/serve_variant.py [변형 이름] [포트]   (기본: draft_fewshot_new 8001)
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

variant = sys.argv[1] if len(sys.argv) > 1 else "draft_fewshot_new"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8001

import store

store.REQUESTS_FILE = ROOT / "data" / f"requests_{variant}.json"
if store.REQUESTS_FILE.exists():
    store.db.update(json.loads(store.REQUESTS_FILE.read_text(encoding="utf-8")))
else:
    store.db.update({"requests": [], "messages": []})

import uvicorn

import importlib

from agent import classifier

variant_mod = importlib.import_module("experiments." + ("hybrid" if variant.startswith("hybrid") else "colab_draft"))
classifier.classify = variant_mod.make_classify(variant)
import main  # noqa: E402  (store/classifier를 바꾼 뒤에 불러와서 복구 로직도 변형 데이터를 쓴다)

print(f"[variant server] classifier = {variant}, data = {store.REQUESTS_FILE.name}, port = {port}", flush=True)
uvicorn.run(main.app, port=port)

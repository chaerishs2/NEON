from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from routes import nurse as nurse_routes  # noqa: E402
from routes import requests as requests_routes  # noqa: E402

from agent import pipeline  # noqa: E402

pipeline.recover()
app = FastAPI(title="NEON")
app.include_router(requests_routes.router)
app.include_router(nurse_routes.router)


@app.get("/api/health")
def health():
    return {"ok": True}


# 라우터는 이 위에 등록하고, StaticFiles는 항상 마지막에 마운트한다
app.mount("/", StaticFiles(directory="static", html=True), name="static")

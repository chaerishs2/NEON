from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import store  # noqa: E402
from routes import nurse as nurse_routes  # noqa: E402
from routes import requests as requests_routes  # noqa: E402

from agent import pipeline  # noqa: E402


@asynccontextmanager
async def lifespan(app):
    # 켤 때마다 빈 상태로 시작한다. NEON_RESTORE=1이면 이전 요청을 복원하고 멈춘 요청을 되살린다
    if os.getenv("NEON_RESTORE") == "1":
        pipeline.recover()
    else:
        store.reset()
    yield


app = FastAPI(title="NEON", lifespan=lifespan)
app.include_router(requests_routes.router)
app.include_router(nurse_routes.router)


@app.get("/api/health")
def health():
    return {"ok": True}


# 라우터는 이 위에 등록하고, StaticFiles는 항상 마지막에 마운트한다
class FreshStatic(StaticFiles):
    # 화면 파일을 고치면 브라우저가 옛 파일을 재사용하지 않고 매번 변경 여부를 확인하게 한다
    async def get_response(self, path, scope):
        res = await super().get_response(path, scope)
        res.headers["Cache-Control"] = "no-cache"
        return res


app.mount("/", FreshStatic(directory=Path(__file__).parent / "static", html=True), name="static")

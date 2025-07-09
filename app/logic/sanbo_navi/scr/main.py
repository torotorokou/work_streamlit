import os, sys
from pathlib import Path
from dotenv import load_dotenv

# config/.env を読み込み
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "config" / ".env")

# PYTHONPATH を取得して sys.path に追加
py_path = os.getenv("PYTHONPATH")
if py_path:
    full_path = (Path(__file__).resolve() / py_path).resolve()
    if str(full_path) not in sys.path:
        sys.path.append(str(full_path))

# --- FastAPI 本体
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scr.api import endpoints

app = FastAPI()

# --- CORS設定（React用）
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- APIルーター登録
app.include_router(endpoints.router, prefix="/api")

# --- ルート確認
@app.get("/")
async def root():
    return {"message": "Welcome to the Sanbo Navi API"}

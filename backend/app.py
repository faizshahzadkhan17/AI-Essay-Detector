"""FastAPI app: the analysis pipeline, lightweight auth + essay history,
and the static frontend. Run with: uvicorn app:app --reload
"""

import re
from pathlib import Path
from typing import Optional

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
from analyze import analyze_essay

db.init_db()

app = FastAPI(title="AI Essay Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
SESSION_COOKIE = "session_token"
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


# ---------------- schemas ----------------

class AnalyzeRequest(BaseModel):
    text: str


class AuthRequest(BaseModel):
    username: str
    password: str


# ---------------- auth helpers ----------------

def _current_user_id(session_token: Optional[str]) -> Optional[int]:
    if not session_token:
        return None
    return db.get_user_id_for_session(session_token)


def _require_user_id(session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)) -> int:
    user_id = _current_user_id(session_token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user_id


def _set_session_cookie(response: Response, token: str):
    response.set_cookie(
        key=SESSION_COOKIE, value=token, httponly=True, samesite="lax",
        max_age=60 * 60 * 24 * 30, path="/",
    )


# ---------------- analyze (auto-saves to history when logged in) ----------------

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Essay text is empty.")
    if len(text) > 20000:
        raise HTTPException(status_code=400, detail="Essay is too long (max 20,000 characters).")

    result = analyze_essay(text)

    user_id = _current_user_id(session_token)
    if user_id is not None:
        try:
            result["saved_to_history"] = True
            db.save_analysis(user_id, text, result)
        except Exception:
            result["saved_to_history"] = False
    return result


# ---------------- auth ----------------

@app.post("/api/auth/signup")
def signup(req: AuthRequest, response: Response):
    username = req.username.strip()
    if not USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="Username must be 3-32 characters: letters, numbers, _ . -")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    try:
        user_id = db.create_user(username, req.password)
    except db.UsernameTakenError:
        raise HTTPException(status_code=409, detail="That username is already taken.")
    token = db.create_session(user_id)
    _set_session_cookie(response, token)
    return {"username": username}


@app.post("/api/auth/login")
def login(req: AuthRequest, response: Response):
    try:
        user_id = db.verify_user(req.username, req.password)
    except db.InvalidCredentialsError:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = db.create_session(user_id)
    _set_session_cookie(response, token)
    return {"username": db.get_username(user_id)}


@app.post("/api/auth/logout")
def logout(response: Response, session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    if session_token:
        db.delete_session(session_token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    user_id = _current_user_id(session_token)
    if user_id is None:
        return JSONResponse(status_code=401, content={"detail": "Not logged in."})
    return {"username": db.get_username(user_id)}


# ---------------- history ----------------

@app.get("/api/history")
def history(session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    uid = _require_user_id(session_token)
    return {"analyses": db.list_analyses(uid)}


@app.get("/api/history/{analysis_id}")
def history_detail(analysis_id: int, session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    uid = _require_user_id(session_token)
    result = db.get_analysis(analysis_id, uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return result


# ---------------- frontend ----------------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

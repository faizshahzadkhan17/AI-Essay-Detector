"""
SQLite-backed users, sessions, and essay-analysis history. Deliberately
minimal: stdlib only (sqlite3, hashlib, secrets), no ORM. This is a
local single-instance dev tool, not a hardened multi-tenant service --
no rate limiting, no CSRF token, no email verification, no password
reset. Session cookies are HttpOnly + SameSite=Lax but not Secure
(there's no HTTPS in local dev). Good enough for "log in and see your
past essays", not good enough to deploy publicly as-is.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DATA_DIR

DB_PATH = DATA_DIR / "app.db"

PBKDF2_ITERATIONS = 200_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                essay_text TEXT NOT NULL,
                result_json TEXT NOT NULL,
                n_sentences INTEGER,
                n_flagged INTEGER,
                mean_prob_ai REAL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id, created_at DESC)")


# ---------------- password hashing ----------------

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS).hex()


# ---------------- users ----------------

class UsernameTakenError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def create_user(username: str, password: str) -> int:
    username = username.strip()
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, pw_hash, salt, _now()),
            )
        except sqlite3.IntegrityError:
            raise UsernameTakenError(username)
        return cur.lastrowid


def verify_user(username: str, password: str) -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        raise InvalidCredentialsError()
    if not secrets.compare_digest(_hash_password(password, row["salt"]), row["password_hash"]):
        raise InvalidCredentialsError()
    return row["id"]


def get_username(user_id: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["username"] if row else None


# ---------------- sessions ----------------

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, _now()))
    return token


def get_user_id_for_session(token: str) -> int | None:
    if not token:
        return None
    with get_conn() as conn:
        row = conn.execute("SELECT user_id FROM sessions WHERE token = ?", (token,)).fetchone()
    return row["user_id"] if row else None


def delete_session(token: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ---------------- analysis history ----------------

def save_analysis(user_id: int, essay_text: str, result: dict) -> int:
    summary = result.get("doc_summary", {})
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO analyses (user_id, essay_text, result_json, n_sentences, n_flagged, mean_prob_ai, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, essay_text, json.dumps(result),
                summary.get("n_sentences"), summary.get("n_flagged"), summary.get("mean_prob_ai"),
                _now(),
            ),
        )
        return cur.lastrowid


def list_analyses(user_id: int, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, essay_text, n_sentences, n_flagged, mean_prob_ai, created_at
               FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        preview = r["essay_text"].strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        out.append({
            "id": r["id"], "preview": preview, "n_sentences": r["n_sentences"],
            "n_flagged": r["n_flagged"], "mean_prob_ai": r["mean_prob_ai"], "created_at": r["created_at"],
        })
    return out


def get_analysis(analysis_id: int, user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT essay_text, result_json, created_at FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
    if row is None:
        return None
    result = json.loads(row["result_json"])
    result["essay_text"] = row["essay_text"]
    result["created_at"] = row["created_at"]
    return result

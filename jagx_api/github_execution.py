"""GitHub App connection and repository execution API for JagX AI.

Uses GitHub App user authorization (no PAT in the browser), encrypted server-side
token storage, Supabase-user ownership, installation-scoped repository access,
and optimistic concurrency for file writes.
Required env: GITHUB_APP_CLIENT_ID, GITHUB_APP_CLIENT_SECRET,
GITHUB_APP_SLUG, GITHUB_OAUTH_REDIRECT_URI, GITHUB_TOKEN_ENCRYPTION_KEY,
SUPABASE_URL, SUPABASE_ANON_KEY.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sqlite3
import time
from typing import Optional
from urllib.parse import urlencode

import requests
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/github", tags=["github-execution"])
GITHUB_API = "https://api.github.com"
DB_PATH = os.environ.get("GITHUB_CONNECTION_DB", "github_connections.sqlite3")


def _db():
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS github_oauth_states (
        state TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at INTEGER NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS github_user_tokens (
        user_id TEXT PRIMARY KEY, encrypted_token TEXT NOT NULL,
        github_login TEXT NOT NULL, connected_at INTEGER NOT NULL,
        encrypted_refresh_token TEXT, expires_at INTEGER
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(github_user_tokens)").fetchall()}
    if "encrypted_refresh_token" not in columns:
        db.execute("ALTER TABLE github_user_tokens ADD COLUMN encrypted_refresh_token TEXT")
    if "expires_at" not in columns:
        db.execute("ALTER TABLE github_user_tokens ADD COLUMN expires_at INTEGER")
    db.commit()
    try:
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass
    return db


def _fernet() -> Fernet:
    key = os.environ.get("GITHUB_TOKEN_ENCRYPTION_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="GitHub connection encryption is not configured")
    try:
        return Fernet(key.encode())
    except Exception:
        raise HTTPException(status_code=503, detail="GitHub connection encryption key is invalid")


def _config():
    client_id = os.environ.get("GITHUB_APP_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GITHUB_APP_CLIENT_SECRET", "").strip()
    redirect_uri = os.environ.get("GITHUB_OAUTH_REDIRECT_URI", "").strip()
    if not client_id or not client_secret or not redirect_uri.startswith("https://"):
        raise HTTPException(status_code=503, detail="GitHub App OAuth is not configured")
    _fernet()
    return client_id, client_secret, redirect_uri


def _user(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to JagX AI first")
    token = authorization.split(" ", 1)[1].strip()
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_PUBLISHABLE_KEY") or "").strip()
    if not url or not key:
        raise HTTPException(status_code=503, detail="Account validation is not configured")
    try:
        res = requests.get(url + "/auth/v1/user",
            headers={"apikey": key, "Authorization": "Bearer " + token}, timeout=8)
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Could not validate your JagX session")
    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="Your JagX session is invalid or expired")
    user_id = (res.json() or {}).get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Could not identify the signed-in user")
    return str(user_id)


def _token(user_id: str) -> str:
    cipher = _fernet()
    with _db() as db:
        row = db.execute("SELECT encrypted_token,encrypted_refresh_token,expires_at FROM github_user_tokens WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=409, detail="Connect your GitHub account first")
    try:
        access = cipher.decrypt(row["encrypted_token"].encode()).decode()
        expires_at = row["expires_at"]
        if not expires_at or int(expires_at) > int(time.time()) + 60:
            return access
        refresh_cipher = row["encrypted_refresh_token"]
        if not refresh_cipher:
            raise HTTPException(status_code=401, detail="GitHub session expired; reconnect GitHub")
        client_id, client_secret, _redirect_uri = _config()
        refresh = cipher.decrypt(refresh_cipher.encode()).decode()
        response = requests.post("https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            json={"client_id": client_id, "client_secret": client_secret,
                  "grant_type": "refresh_token", "refresh_token": refresh}, timeout=12)
        data = response.json() if response.ok else {}
        new_access = data.get("access_token")
        if not new_access:
            raise HTTPException(status_code=401, detail="GitHub session expired; reconnect GitHub")
        new_refresh = data.get("refresh_token") or refresh
        new_expiry = int(time.time()) + int(data.get("expires_in") or 28800)
        with _db() as db:
            db.execute("UPDATE github_user_tokens SET encrypted_token=?,encrypted_refresh_token=?,expires_at=? WHERE user_id=?",
                (cipher.encrypt(new_access.encode()).decode(), cipher.encrypt(new_refresh.encode()).decode(),
                 new_expiry, user_id))
            db.commit()
        return new_access
    except InvalidToken:
        raise HTTPException(status_code=503, detail="Stored GitHub connection cannot be decrypted; reconnect GitHub")
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Could not refresh GitHub session")


def _gh(token: str, method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update({"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"})
    try:
        res = requests.request(method, GITHUB_API + path, headers=headers, timeout=15, **kwargs)
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="GitHub is temporarily unreachable")
    if res.status_code >= 400:
        if res.status_code in (401, 403):
            raise HTTPException(status_code=403, detail="GitHub denied this operation. Check app installation and repository permissions.")
        if res.status_code == 404:
            raise HTTPException(status_code=404, detail="Repository or file was not found or is not accessible to this connection")
        if res.status_code == 409:
            raise HTTPException(status_code=409, detail="GitHub reported a repository conflict")
        raise HTTPException(status_code=502, detail="GitHub could not complete the requested operation")
    if res.status_code == 204 or not res.content:
        return {}
    return res.json()


def _installed_repositories(token: str) -> list[dict]:
    data = _gh(token, "GET", "/user/installations?per_page=100")
    repos = []
    for installation in data.get("installations", []):
        iid = installation.get("id")
        if not iid:
            continue
        page = 1
        while page <= 10:
            batch = _gh(token, "GET", f"/user/installations/{iid}/repositories?per_page=100&page={page}")
            items = batch.get("repositories", [])
            repos.extend(items)
            if len(items) < 100:
                break
            page += 1
    # Return only minimum metadata required by the UI.
    unique = {}
    for repo in repos:
        full_name = repo.get("full_name")
        if full_name:
            unique[full_name.lower()] = {
                "full_name": full_name,
                "default_branch": repo.get("default_branch") or "main",
                "private": bool(repo.get("private")),
                "html_url": repo.get("html_url"),
            }
    return sorted(unique.values(), key=lambda item: item["full_name"].lower())


def _require_installed_repo(token: str, owner: str, repo: str) -> str:
    if not owner or not repo or any(x in (owner + repo) for x in ("..", "/", "\\", "\x00")):
        raise HTTPException(status_code=400, detail="Invalid repository name")
    full_name = owner + "/" + repo
    allowed = {item["full_name"].lower() for item in _installed_repositories(token)}
    if full_name.lower() not in allowed:
        raise HTTPException(status_code=403, detail="Install the JagX AI GitHub App on this repository before accessing it")
    return full_name


@router.get("/connect")
def connect(authorization: Optional[str] = Header(default=None)):
    user_id = _user(authorization)
    client_id, _secret, redirect_uri = _config()
    state = secrets.token_urlsafe(32)
    now = int(time.time())
    with _db() as db:
        db.execute("DELETE FROM github_oauth_states WHERE expires_at < ?", (now,))
        db.execute("INSERT INTO github_oauth_states(state,user_id,expires_at) VALUES(?,?,?)",
                   (state, user_id, now + 600))
        db.commit()
    params = urlencode({"client_id": client_id, "redirect_uri": redirect_uri, "state": state})
    slug = os.environ.get("GITHUB_APP_SLUG", "").strip()
    install_url = f"https://github.com/apps/{slug}/installations/new" if slug else None
    return {"authorization_url": "https://github.com/login/oauth/authorize?" + params,
            "installation_url": install_url, "expires_in": 600,
            "note": "Install the JagX AI GitHub App only on repositories you want JagX to access."}


@router.get("/callback")
def callback(code: str = "", state: str = "", error: str = ""):
    frontend = (os.environ.get("JAGX_FRONTEND_URL") or "https://jagxai.name.ng").strip().rstrip("/")
    if error:
        return RedirectResponse(frontend + "/#/github?github_error=authorization_denied", status_code=303)
    if not code or not state:
        return RedirectResponse(frontend + "/#/github?github_error=invalid_callback", status_code=303)
    now = int(time.time())
    with _db() as db:
        row = db.execute("SELECT user_id,expires_at FROM github_oauth_states WHERE state=?", (state,)).fetchone()
        db.execute("DELETE FROM github_oauth_states WHERE state=?", (state,))
        db.commit()
    if not row or int(row["expires_at"]) < now:
        return RedirectResponse(frontend + "/#/github?github_error=state_expired", status_code=303)
    client_id, client_secret, redirect_uri = _config()
    try:
        exchange = requests.post("https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            json={"client_id": client_id, "client_secret": client_secret, "code": code,
                  "redirect_uri": redirect_uri}, timeout=12)
        data = exchange.json() if exchange.ok else {}
    except (requests.RequestException, ValueError):
        return RedirectResponse(frontend + "/#/github?github_error=exchange_failed", status_code=303)
    access_token = data.get("access_token")
    if not access_token:
        return RedirectResponse(frontend + "/#/github?github_error=exchange_failed", status_code=303)
    try:
        account = _gh(access_token, "GET", "/user")
        # A GitHub App user token is limited by the App's granted permissions and installation.
        cipher = _fernet()
        encrypted = cipher.encrypt(access_token.encode()).decode()
        refresh_token = data.get("refresh_token")
        encrypted_refresh = cipher.encrypt(refresh_token.encode()).decode() if refresh_token else None
        expires_in = data.get("expires_in")
        expires_at = now + int(expires_in) if expires_in else None
        with _db() as db:
            db.execute("""INSERT INTO github_user_tokens(user_id,encrypted_token,github_login,connected_at,encrypted_refresh_token,expires_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                encrypted_token=excluded.encrypted_token, github_login=excluded.github_login,
                connected_at=excluded.connected_at, encrypted_refresh_token=excluded.encrypted_refresh_token,
                expires_at=excluded.expires_at""",
                (row["user_id"], encrypted, str(account.get("login") or ""), now, encrypted_refresh, expires_at))
            db.commit()
    except HTTPException:
        return RedirectResponse(frontend + "/#/github?github_error=storage_not_ready", status_code=303)
    return RedirectResponse(frontend + "/#/github?github_connected=1", status_code=303)


@router.get("/status")
def status(authorization: Optional[str] = Header(default=None)):
    user_id = _user(authorization)
    with _db() as db:
        row = db.execute("SELECT github_login,connected_at FROM github_user_tokens WHERE user_id=?", (user_id,)).fetchone()
    return {"connected": bool(row), "login": row["github_login"] if row else None,
            "connected_at": row["connected_at"] if row else None}


@router.delete("/disconnect")
def disconnect(authorization: Optional[str] = Header(default=None)):
    user_id = _user(authorization)
    with _db() as db:
        db.execute("DELETE FROM github_user_tokens WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM github_oauth_states WHERE user_id=?", (user_id,))
        db.commit()
    return {"ok": True, "connected": False}


@router.get("/repos")
def repositories(authorization: Optional[str] = Header(default=None)):
    user_id = _user(authorization)
    return {"repositories": _installed_repositories(_token(user_id))}


@router.get("/repo/{owner}/{repo}/contents")
def contents(owner: str, repo: str, path: str = Query(default=""), ref: str = Query(default="main"),
             authorization: Optional[str] = Header(default=None)):
    user_id = _user(authorization)
    token = _token(user_id)
    _require_installed_repo(token, owner, repo)
    encoded_path = "/".join(__import__("urllib.parse", fromlist=["quote"]).quote(part, safe="") for part in path.split("/") if part)
    endpoint = f"/repos/{owner}/{repo}/contents/{encoded_path}"
    from urllib.parse import quote
    endpoint = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/contents" + ("/" + encoded_path if encoded_path else "")
    data = _gh(token, "GET", endpoint + "?ref=" + __import__("urllib.parse", fromlist=["quote"]).quote(ref, safe=""))
    if isinstance(data, list):
        return {"items": [{"name": x.get("name"), "path": x.get("path"), "type": x.get("type"),
                           "size": x.get("size", 0)} for x in data]}
    if data.get("type") != "file" or not data.get("content"):
        raise HTTPException(status_code=415, detail="Select a text file no larger than GitHub's API limit")
    try:
        raw = base64.b64decode(data["content"])
        text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=415, detail="This file is not valid UTF-8 text")
    return {"name": data.get("name"), "path": data.get("path"), "sha": data.get("sha"),
            "content": text, "size": data.get("size", len(raw))}


class CommitRequest(BaseModel):
    branch: str = Field(default="main", min_length=1, max_length=120)
    message: str = Field(..., min_length=3, max_length=180)
    content: str = Field(..., max_length=500000)
    expected_sha: Optional[str] = Field(default=None, max_length=100)


@router.put("/repo/{owner}/{repo}/contents/{path:path}")
def commit_file(owner: str, repo: str, path: str, body: CommitRequest,
                authorization: Optional[str] = Header(default=None)):
    user_id = _user(authorization)
    token = _token(user_id)
    _require_installed_repo(token, owner, repo)
    if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
        raise HTTPException(status_code=400, detail="Invalid file path")
    from urllib.parse import quote
    endpoint = f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/contents/" + "/".join(quote(p, safe="") for p in path.split("/"))
    current_sha = None
    try:
        current = _gh(token, "GET", endpoint + "?ref=" + quote(body.branch, safe=""))
        current_sha = current.get("sha")
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
    if body.expected_sha and current_sha != body.expected_sha:
        raise HTTPException(status_code=409, detail="This file changed on GitHub. Reload it and review the latest version before committing.")
    if current_sha and not body.expected_sha:
        raise HTTPException(status_code=409, detail="File already exists. Reload it before updating to prevent overwriting newer work.")
    if not current_sha and body.expected_sha:
        raise HTTPException(status_code=409, detail="File was removed on GitHub. Reload before continuing.")
    payload = {"message": body.message, "content": base64.b64encode(body.content.encode()).decode(),
               "branch": body.branch}
    if current_sha:
        payload["sha"] = current_sha
    result = _gh(token, "PUT", endpoint, json=payload)
    commit = result.get("commit") or {}
    return {"ok": True, "path": path, "branch": body.branch, "commit_sha": commit.get("sha"),
            "html_url": commit.get("html_url"), "message": "GitHub accepted the commit"}

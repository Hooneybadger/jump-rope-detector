from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import AuthSession, User


SESSION_COOKIE = "jr_session"
_password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session(db: Session, user: User, request: Request) -> tuple[AuthSession, str]:
    settings = get_settings()
    raw_token = secrets.token_urlsafe(48)
    session = AuthSession(
        token_hash=hash_token(raw_token),
        csrf_token=secrets.token_urlsafe(32),
        user_id=user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=settings.session_hours),
        user_agent=(request.headers.get("user-agent") or "")[:255],
        ip_address=request.client.host if request.client else None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session, raw_token


def set_session_cookie(response: Response, raw_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        SESSION_COOKIE,
        raw_token,
        max_age=settings.session_hours * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")


def resolve_session(db: Session, raw_token: str | None) -> AuthSession | None:
    if not raw_token:
        return None
    session = db.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == hash_token(raw_token),
            AuthSession.expires_at > datetime.now(timezone.utc),
        )
    )
    if session is None or not session.user.is_active:
        return None
    return session


def require_user(request: Request, db: Session) -> tuple[User, AuthSession]:
    session = resolve_session(db, request.cookies.get(SESSION_COOKIE))
    if session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "로그인이 필요합니다.")
    return session.user, session


def require_csrf(request: Request, session: AuthSession) -> None:
    supplied = request.headers.get("x-csrf-token", "")
    if not supplied or not hmac.compare_digest(supplied, session.csrf_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "요청 검증 토큰이 올바르지 않습니다.")


class LoginRateLimiter:
    def __init__(self, limit: int = 10, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            attempts = self._attempts[key]
            while attempts and attempts[0] <= now - self.window:
                attempts.popleft()
            if len(attempts) >= self.limit:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "잠시 후 다시 시도해 주세요.")
            attempts.append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


login_limiter = LoginRateLimiter()


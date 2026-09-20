from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .database import Base, SessionLocal, engine, get_db
from .models import AuditLog, AuthSession, PasswordResetToken, User, Workout, utcnow
from .notifications import send_email
from .pdf import workout_pdf
from .schemas import (
    BulkDeleteInput,
    LoginInput,
    PasswordResetConfirm,
    PasswordResetRequest,
    SignupInput,
    UserCreate,
    UserUpdate,
)
from .security import (
    SESSION_COOKIE,
    clear_session_cookie,
    hash_token,
    hash_password,
    login_limiter,
    new_session,
    require_csrf,
    require_user,
    resolve_session,
    set_session_cookie,
    verify_password,
)


settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
VALID_MODES = {"basic", "alternating", "double"}
MODE_NAMES = {"basic": "모아뛰기", "alternating": "번갈아뛰기", "double": "이중뛰기"}
STATUS_NAMES = {"completed": "완료", "interrupted": "중단", "running": "측정 중"}


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _user_json(user: User, csrf_token: str | None = None) -> dict:
    data = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "displayName": user.display_name,
        "role": user.role,
        "active": user.is_active,
        "permissions": {
            "basic": user.role == "admin" or user.can_basic,
            "alternating": user.role == "admin" or user.can_alternating,
            "double": user.role == "admin" or user.can_double,
            "history": user.role == "admin" or user.can_view_history,
        },
    }
    if csrf_token:
        data["csrfToken"] = csrf_token
    return data


def _audit(db: Session, request: Request | None, actor_id: int | None, action: str,
           target_type: str | None = None, target_id: str | None = None,
           detail: str | None = None) -> None:
    db.add(AuditLog(
        actor_user_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
        ip_address=request.client.host if request and request.client else None,
    ))


def _workout_json(workout: Workout, display_name: str) -> dict:
    return {
        "id": workout.id,
        "user": display_name,
        "mode": workout.mode,
        "count": workout.count,
        "duration": workout.duration_seconds,
        "targetDuration": workout.target_duration_seconds,
        "status": workout.status,
        "startedAt": workout.started_at.isoformat(),
        "endedAt": workout.ended_at.isoformat() if workout.ended_at else None,
    }


def _accessible_workout(workout_id: int, request: Request, db: Session) -> tuple[Workout, User, AuthSession]:
    actor, auth_session = require_user(request, db)
    workout = db.get(Workout, workout_id)
    if workout is None or (actor.role != "admin" and workout.user_id != actor.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "측정 기록을 찾을 수 없습니다.")
    if actor.role != "admin" and not actor.can_view_history:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "기록 조회 권한이 필요합니다.")
    return workout, actor, auth_session


def _seed_admin() -> None:
    if settings.environment == "production" and not settings.admin_password:
        raise RuntimeError("ADMIN_PASSWORD is required in production")
    if not settings.admin_password:
        return
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.username == settings.admin_username.lower()))
        if admin is None:
            admin = User(
                username=settings.admin_username.lower(),
                display_name=settings.admin_display_name,
                password_hash=hash_password(settings.admin_password),
                role="admin",
                can_basic=True,
                can_alternating=True,
                can_double=True,
                can_view_history=True,
            )
            db.add(admin)
            db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.environment != "production":
        Base.metadata.create_all(bind=engine)
    _seed_admin()
    yield


app = FastAPI(title=settings.app_name, docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.hosts)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
        "style-src 'self'; script-src 'self'; connect-src 'self' ws: wss:; "
        "font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(payload: LoginInput, request: Request, response: Response, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "unknown"
    login_limiter.check(ip)
    username = payload.username.strip().lower()
    user = db.scalar(select(User).where(User.username == username))
    now = datetime.now(timezone.utc)
    if user and _aware(user.locked_until) and _aware(user.locked_until) > now:
        raise HTTPException(status.HTTP_423_LOCKED, "로그인 실패가 누적되어 계정이 잠겼습니다.")
    valid = bool(user and user.is_active and verify_password(user.password_hash, payload.password))
    if not valid:
        if user:
            user.failed_login_count += 1
            if user.failed_login_count >= 5:
                user.locked_until = now + timedelta(minutes=15)
                user.failed_login_count = 0
            _audit(db, request, user.id, "auth.login_failed", "user", str(user.id))
            db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "아이디 또는 비밀번호가 올바르지 않습니다.")
    user.failed_login_count = 0
    user.locked_until = None
    auth_session, raw_token = new_session(db, user, request)
    _audit(db, request, user.id, "auth.login", "user", str(user.id))
    db.commit()
    login_limiter.reset(ip)
    set_session_cookie(response, raw_token)
    return _user_json(user, auth_session.csrf_token)


@app.post("/api/auth/signup", status_code=201)
def signup(payload: SignupInput, request: Request, response: Response, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "unknown"
    login_limiter.check(f"signup:{ip}")
    user = User(
        username=payload.username.strip().lower(),
        email=payload.email.strip().lower(),
        display_name=payload.display_name.strip(),
        password_hash=hash_password(payload.password),
        role="member",
        can_basic=True,
        can_alternating=True,
        can_double=True,
        can_view_history=True,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 사용 중인 아이디 또는 이메일입니다.") from exc
    auth_session, raw_token = new_session(db, user, request)
    _audit(db, request, user.id, "auth.signup", "user", str(user.id))
    db.commit()
    set_session_cookie(response, raw_token)
    return _user_json(user, auth_session.csrf_token)


@app.post("/api/auth/password-reset/request", status_code=202)
def request_password_reset(payload: PasswordResetRequest, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "unknown"
    login_limiter.check(f"reset:{ip}")
    user = db.scalar(select(User).where(User.email == payload.email.strip().lower(), User.is_active.is_(True)))
    response = {"message": "가입된 이메일이면 비밀번호 재설정 안내를 보냈습니다."}
    if user is None:
        return response
    raw_token = secrets.token_urlsafe(48)
    reset = PasswordResetToken(
        token_hash=hash_token(raw_token),
        user_id=user.id,
        expires_at=utcnow() + timedelta(minutes=30),
    )
    db.add(reset)
    _audit(db, request, user.id, "auth.password_reset_requested", "user", str(user.id))
    db.commit()
    reset_url = f"{settings.public_base_url.rstrip('/')}?reset={raw_token}"
    if settings.smtp_host and settings.smtp_from:
        try:
            send_email(
                user.email or payload.email,
                "헤아리오 비밀번호 재설정",
                f"30분 안에 아래 주소에서 새 비밀번호를 설정하세요.\n\n{reset_url}",
            )
        except Exception:
            # Account discovery protection: keep the public response identical.
            pass
    if settings.environment in {"development", "test"}:
        response["developmentToken"] = raw_token
    return response


@app.post("/api/auth/password-reset/confirm")
def confirm_password_reset(payload: PasswordResetConfirm, request: Request, db: Session = Depends(get_db)):
    reset = db.scalar(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == hash_token(payload.token),
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > utcnow(),
        )
    )
    if reset is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "재설정 링크가 만료되었거나 이미 사용되었습니다.")
    reset.user.password_hash = hash_password(payload.password)
    reset.used_at = utcnow()
    db.query(AuthSession).filter(AuthSession.user_id == reset.user_id).delete()
    _audit(db, request, reset.user_id, "auth.password_reset_completed", "user", str(reset.user_id))
    db.commit()
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request, db: Session = Depends(get_db)):
    user, auth_session = require_user(request, db)
    return _user_json(user, auth_session.csrf_token)


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    user, auth_session = require_user(request, db)
    require_csrf(request, auth_session)
    _audit(db, request, user.id, "auth.logout", "user", str(user.id))
    db.delete(auth_session)
    db.commit()
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    user, _ = require_user(request, db)
    filters = [] if user.role == "admin" else [Workout.user_id == user.id]
    total_count, total_duration, session_count = db.execute(
        select(func.coalesce(func.sum(Workout.count), 0),
               func.coalesce(func.sum(Workout.duration_seconds), 0),
               func.count(Workout.id)).where(*filters)
    ).one()
    recent_query = select(Workout, User.display_name).join(User).where(*filters).order_by(desc(Workout.started_at))
    recent = [
        _workout_json(workout, display_name)
        for workout, display_name in db.execute(recent_query).all()
    ] if (user.role == "admin" or user.can_view_history) else []
    return {
        "summary": {"count": int(total_count), "duration": int(total_duration), "sessions": int(session_count)},
        "recent": recent,
    }


def _require_admin(request: Request, db: Session) -> tuple[User, AuthSession]:
    user, auth_session = require_user(request, db)
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "관리자 권한이 필요합니다.")
    return user, auth_session


@app.get("/api/admin/users")
def users(request: Request, db: Session = Depends(get_db)):
    _require_admin(request, db)
    rows = db.scalars(select(User).order_by(User.username)).all()
    return [_user_json(user) | {"createdAt": user.created_at.isoformat()} for user in rows]


@app.post("/api/admin/users", status_code=201)
def create_user(payload: UserCreate, request: Request, db: Session = Depends(get_db)):
    actor, auth_session = _require_admin(request, db)
    require_csrf(request, auth_session)
    user = User(
        username=payload.username.lower(),
        email=payload.email.strip().lower() if payload.email else None,
        display_name=payload.display_name.strip(),
        password_hash=hash_password(payload.password),
        role=payload.role,
        can_basic=payload.can_basic,
        can_alternating=payload.can_alternating,
        can_double=payload.can_double,
        can_view_history=payload.can_view_history,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 사용 중인 아이디입니다.") from exc
    _audit(db, request, actor.id, "user.create", "user", str(user.id), f"role={user.role}")
    db.commit()
    return _user_json(user)


@app.post("/api/admin/users/bulk-delete")
def delete_users(payload: BulkDeleteInput, request: Request, db: Session = Depends(get_db)):
    actor, auth_session = _require_admin(request, db)
    require_csrf(request, auth_session)
    if actor.id in payload.ids:
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 로그인한 관리자 계정은 삭제할 수 없습니다.")
    users = db.scalars(select(User).where(User.id.in_(payload.ids))).all()
    if len(users) != len(payload.ids):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "삭제할 사용자 중 일부를 찾을 수 없습니다.")
    for user in users:
        _audit(db, request, actor.id, "user.delete", "user", str(user.id), f"username={user.username}")
        db.delete(user)
    db.commit()
    return {"deleted": len(users)}


@app.patch("/api/admin/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate, request: Request, db: Session = Depends(get_db)):
    actor, auth_session = _require_admin(request, db)
    require_csrf(request, auth_session)
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "사용자를 찾을 수 없습니다.")
    changes = payload.model_dump(exclude_unset=True)
    if user.id == actor.id and (changes.get("is_active") is False or changes.get("role") == "member"):
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 관리자 계정의 권한은 낮출 수 없습니다.")
    password = changes.pop("password", None)
    if password:
        user.password_hash = hash_password(password)
        db.query(AuthSession).filter(AuthSession.user_id == user.id, AuthSession.id != auth_session.id).delete()
    for field, value in changes.items():
        setattr(user, field, value.strip() if field == "display_name" else value)
    _audit(db, request, actor.id, "user.update", "user", str(user.id), ",".join(sorted(changes)))
    db.commit()
    return _user_json(user)


@app.post("/api/workouts/bulk-delete")
def delete_workouts(payload: BulkDeleteInput, request: Request, db: Session = Depends(get_db)):
    actor, auth_session = require_user(request, db)
    require_csrf(request, auth_session)
    query = select(Workout).where(Workout.id.in_(payload.ids))
    if actor.role != "admin":
        query = query.where(Workout.user_id == actor.id)
    workouts = db.scalars(query).all()
    if len(workouts) != len(payload.ids):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "삭제할 측정 기록 중 일부를 찾을 수 없습니다.")
    if any(workout.status == "running" for workout in workouts):
        raise HTTPException(status.HTTP_409_CONFLICT, "측정 중인 기록은 삭제할 수 없습니다.")
    for workout in workouts:
        _audit(db, request, actor.id, "workout.delete", "workout", str(workout.id), "bulk=true")
        db.delete(workout)
    db.commit()
    return {"deleted": len(workouts)}


@app.get("/api/workouts/{workout_id}")
def workout_detail(workout_id: int, request: Request, db: Session = Depends(get_db)):
    workout, _, _ = _accessible_workout(workout_id, request, db)
    return _workout_json(workout, workout.user.display_name)


@app.get("/api/workouts/{workout_id}/pdf")
def download_workout_pdf(workout_id: int, request: Request, db: Session = Depends(get_db)):
    workout, _, _ = _accessible_workout(workout_id, request, db)
    content = workout_pdf(
        workout_id=workout.id,
        user_name=workout.user.display_name,
        mode_name=MODE_NAMES.get(workout.mode, workout.mode),
        count=workout.count,
        duration=workout.duration_seconds,
        status_name=STATUS_NAMES.get(workout.status, workout.status),
        started_at=_aware(workout.started_at),
    )
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="jump-rope-result-{workout.id}.pdf"'},
    )


@app.delete("/api/workouts/{workout_id}", status_code=204)
def delete_workout(workout_id: int, request: Request, db: Session = Depends(get_db)):
    workout, actor, auth_session = _accessible_workout(workout_id, request, db)
    require_csrf(request, auth_session)
    if workout.status == "running":
        raise HTTPException(status.HTTP_409_CONFLICT, "측정 중인 기록은 삭제할 수 없습니다.")
    _audit(db, request, actor.id, "workout.delete", "workout", str(workout.id))
    db.delete(workout)
    db.commit()
    return Response(status_code=204)


@app.get("/api/admin/audit")
def audit_logs(request: Request, db: Session = Depends(get_db)):
    _require_admin(request, db)
    rows = db.execute(
        select(AuditLog, User.username).outerjoin(User, AuditLog.actor_user_id == User.id)
        .order_by(desc(AuditLog.created_at)).limit(100)
    ).all()
    return [{
        "id": log.id,
        "actor": username or "system",
        "action": log.action,
        "target": f"{log.target_type or '-'}:{log.target_id or '-'}",
        "detail": log.detail,
        "createdAt": log.created_at.isoformat(),
    } for log, username in rows]


class StreamSlots:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.lock = asyncio.Lock()

    async def reserve(self) -> bool:
        async with self.lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    async def release(self) -> None:
        async with self.lock:
            self.active = max(0, self.active - 1)


stream_slots = StreamSlots(settings.max_concurrent_streams)


@app.websocket("/ws/count/{mode}")
async def count_stream(websocket: WebSocket, mode: str):
    origin = (websocket.headers.get("origin") or "").rstrip("/")
    try:
        target_duration = int(websocket.query_params.get("duration", "60"))
    except ValueError:
        target_duration = 0
    if mode not in VALID_MODES or origin not in settings.origins or not 10 <= target_duration <= 3600:
        await websocket.close(code=1008)
        return
    db = SessionLocal()
    processor = None
    workout = None
    reserved = False
    try:
        auth_session = resolve_session(db, websocket.cookies.get(SESSION_COOKIE))
        if auth_session is None or not auth_session.user.allows(mode):
            await websocket.close(code=1008)
            return
        reserved = await stream_slots.reserve()
        if not reserved:
            await websocket.close(code=1013, reason="동시 측정 한도를 초과했습니다.")
            return
        await websocket.accept()
        from .jump_service import JumpCounterSession

        processor = await run_in_threadpool(JumpCounterSession, mode)
        workout = Workout(
            user_id=auth_session.user.id,
            mode=mode,
            status="running",
            target_duration_seconds=target_duration,
        )
        db.add(workout)
        db.flush()
        _audit(db, None, auth_session.user.id, "workout.start", "workout", str(workout.id), f"mode={mode}")
        db.commit()
        await websocket.send_json({"type": "ready", "workoutId": workout.id, "targetDuration": target_duration})

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if message.get("text"):
                command = json.loads(message["text"])
                if command.get("type") == "stop":
                    break
                continue
            payload = message.get("bytes")
            if payload is None:
                continue
            result = await run_in_threadpool(processor.process, payload, settings.max_frame_bytes)
            await websocket.send_json({
                "type": "state",
                "count": result.count,
                "phase": result.phase,
                "ready": result.ready,
                "readyProgress": round(result.ready_progress, 3),
                "countdown": round(result.countdown, 1),
                "elapsed": round(result.elapsed, 1),
                "landmarks": result.landmarks,
                "processingMs": round(result.processing_ms, 1),
            })
            if result.phase == "COUNTING" and result.elapsed >= target_duration:
                break

        workout.status = "completed"
        workout.count = processor.count
        workout.duration_seconds = round(processor.elapsed)
        workout.ended_at = utcnow()
        _audit(db, None, auth_session.user.id, "workout.complete", "workout", str(workout.id), f"count={workout.count}")
        db.commit()
        await websocket.send_json({
            "type": "complete",
            "workoutId": workout.id,
            "count": workout.count,
            "duration": workout.duration_seconds,
        })
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        if workout and workout.status == "running":
            workout.status = "interrupted"
            workout.count = processor.count if processor else 0
            workout.duration_seconds = round(processor.elapsed) if processor else 0
            workout.ended_at = utcnow()
            db.commit()
    except (ValueError, json.JSONDecodeError):
        if websocket.client_state.name == "CONNECTED":
            await websocket.close(code=1003, reason="잘못된 프레임입니다.")
    finally:
        if workout and workout.status == "running":
            workout.status = "interrupted"
            workout.count = processor.count if processor else 0
            workout.duration_seconds = round(processor.elapsed) if processor else 0
            workout.ended_at = utcnow()
            db.commit()
        if processor:
            await run_in_threadpool(processor.close)
        if reserved:
            await stream_slots.release()
        db.close()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

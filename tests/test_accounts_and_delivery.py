from starlette.websockets import WebSocketDisconnect

from app.database import SessionLocal
from app.models import User, Workout


def signup(client):
    return client.post("/api/auth/signup", json={
        "username": "jumper01",
        "email": "jumper@example.com",
        "display_name": "점퍼",
        "password": "secure-password-123",
    })


def test_signup_persists_profile_and_starts_session(client):
    response = signup(client)
    assert response.status_code == 201
    assert response.json()["email"] == "jumper@example.com"
    assert response.json()["permissions"] == {
        "basic": True, "alternating": True, "double": True, "history": True,
    }
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    with SessionLocal() as db:
        user = db.query(User).filter_by(username="jumper01").one()
        assert user.email == "jumper@example.com"


def test_signup_rejects_duplicate_email(client):
    assert signup(client).status_code == 201
    response = client.post("/api/auth/signup", json={
        "username": "jumper02",
        "email": "jumper@example.com",
        "display_name": "다른 점퍼",
        "password": "another-password-123",
    })
    assert response.status_code == 409


def test_member_can_edit_and_save_profile(client):
    signed_up = signup(client)
    client.headers["X-CSRF-Token"] = signed_up.json()["csrfToken"]

    response = client.patch("/api/auth/profile", json={
        "display_name": "리듬 점퍼",
        "email": "rhythm@example.com",
        "current_password": None,
        "new_password": None,
    })

    assert response.status_code == 200
    assert response.json()["displayName"] == "리듬 점퍼"
    assert response.json()["email"] == "rhythm@example.com"
    profile = client.get("/api/auth/me").json()
    assert profile["displayName"] == "리듬 점퍼"
    assert profile["email"] == "rhythm@example.com"


def test_profile_password_change_requires_current_password(client):
    signed_up = signup(client)
    client.headers["X-CSRF-Token"] = signed_up.json()["csrfToken"]
    payload = {
        "display_name": "점퍼",
        "email": "jumper@example.com",
        "current_password": "wrong-password",
        "new_password": "new-secure-password-456",
    }

    rejected = client.patch("/api/auth/profile", json=payload)
    assert rejected.status_code == 400

    payload["current_password"] = "secure-password-123"
    changed = client.patch("/api/auth/profile", json=payload)
    assert changed.status_code == 200
    client.post("/api/auth/logout")
    login = client.post("/api/auth/login", json={
        "username": "jumper01",
        "password": "new-secure-password-456",
    })
    assert login.status_code == 200


def test_password_reset_token_is_single_use(client):
    assert signup(client).status_code == 201
    request = client.post("/api/auth/password-reset/request", json={"email": "jumper@example.com"})
    assert request.status_code == 202
    token = request.json()["developmentToken"]
    confirmed = client.post("/api/auth/password-reset/confirm", json={
        "token": token, "password": "brand-new-password-123",
    })
    assert confirmed.status_code == 200
    assert client.post("/api/auth/password-reset/confirm", json={
        "token": token, "password": "brand-new-password-456",
    }).status_code == 400
    login = client.post("/api/auth/login", json={"username": "jumper01", "password": "brand-new-password-123"})
    assert login.status_code == 200


def test_workout_can_be_reopened_and_downloaded_as_pdf(client):
    signed_up = signup(client)
    assert signed_up.status_code == 201
    client.headers["X-CSRF-Token"] = signed_up.json()["csrfToken"]
    with SessionLocal() as db:
        user = db.query(User).filter_by(username="jumper01").one()
        workout = Workout(user_id=user.id, mode="basic", count=42, duration_seconds=60, status="completed")
        db.add(workout)
        db.commit()
        workout_id = workout.id

    detail = client.get(f"/api/workouts/{workout_id}")
    assert detail.status_code == 200
    assert detail.json()["count"] == 42
    assert detail.json()["user"] == "점퍼"

    pdf = client.get(f"/api/workouts/{workout_id}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["content-disposition"] == f'attachment; filename="jump-rope-result-{workout_id}.pdf"'
    assert pdf.content.startswith(b"%PDF-1.7")


def test_workout_can_be_deleted_and_disappears_from_history(client):
    signed_up = signup(client)
    client.headers["X-CSRF-Token"] = signed_up.json()["csrfToken"]
    with SessionLocal() as db:
        user = db.query(User).filter_by(username="jumper01").one()
        workout = Workout(user_id=user.id, mode="alternating", count=31, duration_seconds=45, status="completed")
        db.add(workout)
        db.commit()
        workout_id = workout.id

    response = client.delete(f"/api/workouts/{workout_id}")
    assert response.status_code == 204
    assert client.get(f"/api/workouts/{workout_id}").status_code == 404
    assert client.get("/api/dashboard").json()["recent"] == []


def test_selected_workouts_can_be_deleted_together(client):
    signed_up = signup(client)
    client.headers["X-CSRF-Token"] = signed_up.json()["csrfToken"]
    with SessionLocal() as db:
        user = db.query(User).filter_by(username="jumper01").one()
        workouts = [
            Workout(user_id=user.id, mode="basic", count=count, duration_seconds=30, status="completed")
            for count in (10, 20, 30)
        ]
        db.add_all(workouts)
        db.commit()
        selected_ids = [workouts[0].id, workouts[2].id]

    response = client.post("/api/workouts/bulk-delete", json={"ids": selected_ids})
    assert response.status_code == 200
    assert response.json() == {"deleted": 2}
    remaining = client.get("/api/dashboard").json()["recent"]
    assert [item["count"] for item in remaining] == [20]


def test_member_cannot_access_another_users_workout(client):
    with SessionLocal() as db:
        admin = db.query(User).filter_by(username="admin").one()
        workout = Workout(user_id=admin.id, mode="double", count=10, duration_seconds=20, status="completed")
        db.add(workout)
        db.commit()
        workout_id = workout.id
    signed_up = signup(client)
    client.headers["X-CSRF-Token"] = signed_up.json()["csrfToken"]
    assert client.get(f"/api/workouts/{workout_id}").status_code == 404
    assert client.get(f"/api/workouts/{workout_id}/pdf").status_code == 404
    assert client.delete(f"/api/workouts/{workout_id}").status_code == 404


def test_invalid_workout_duration_is_rejected_before_stream_starts(admin_client):
    try:
        with admin_client.websocket_connect("/ws/count/basic?duration=9", headers={"origin": "http://testserver"}):
            raise AssertionError("invalid duration websocket should not connect")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008


def test_invalid_workout_countdown_is_rejected_before_stream_starts(admin_client):
    try:
        with admin_client.websocket_connect(
            "/ws/count/basic?duration=60&countdown=31",
            headers={"origin": "http://testserver"},
        ):
            raise AssertionError("invalid countdown websocket should not connect")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008

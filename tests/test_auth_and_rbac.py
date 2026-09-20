def test_login_sets_http_only_cookie_and_security_headers(client):
    response = client.post("/api/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_state_change_requires_csrf(client):
    login = client.post("/api/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    assert login.status_code == 200
    response = client.post("/api/admin/users", json={
        "username": "athlete",
        "display_name": "선수",
        "password": "a-secure-password",
    })
    assert response.status_code == 403


def test_admin_can_assign_mode_permissions(admin_client):
    created = admin_client.post("/api/admin/users", json={
        "username": "athlete",
        "display_name": "선수",
        "password": "a-secure-password",
        "can_basic": True,
        "can_alternating": False,
        "can_double": False,
        "can_view_history": True,
    })
    assert created.status_code == 201
    user = created.json()
    assert user["permissions"] == {"basic": True, "alternating": False, "double": False, "history": True}

    updated = admin_client.patch(f"/api/admin/users/{user['id']}", json={"can_double": True})
    assert updated.status_code == 200
    assert updated.json()["permissions"]["double"] is True


def test_member_cannot_open_admin_api(admin_client):
    created = admin_client.post("/api/admin/users", json={
        "username": "member01",
        "display_name": "일반 사용자",
        "password": "member-password-123",
    })
    assert created.status_code == 201
    admin_client.post("/api/auth/logout")
    login = admin_client.post("/api/auth/login", json={"username": "member01", "password": "member-password-123"})
    assert login.status_code == 200
    response = admin_client.get("/api/admin/users")
    assert response.status_code == 403


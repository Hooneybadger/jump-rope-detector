def test_empty_dashboard_has_stable_shape(admin_client):
    response = admin_client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json() == {
        "summary": {"count": 0, "duration": 0, "sessions": 0},
        "recent": [],
    }


def test_dashboard_returns_all_measurement_results(admin_client):
    from app.database import SessionLocal
    from app.models import User, Workout

    with SessionLocal() as db:
        user = db.query(User).filter_by(username="admin").one()
        db.add_all([
            Workout(user_id=user.id, mode="basic", count=index, duration_seconds=30, status="completed")
            for index in range(15)
        ])
        db.commit()
    response = admin_client.get("/api/dashboard")
    assert response.status_code == 200
    assert len(response.json()["recent"]) == 15


def test_unknown_static_route_returns_404(client):
    response = client.get("/not-a-real-file.txt")
    assert response.status_code == 404


def test_workout_setup_exposes_countdown_and_encouragement_asset(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="workout-countdown"' in response.text
    assert 'data-countdown="5"' in response.text
    assert 'src="/cheer-basic.gif"' in response.text
    script = client.get("/app.js").text
    assert 'animation: "/cheer-alternating.gif"' in script
    assert 'animation: "/cheer-double.gif"' in script


def test_current_brand_profile_editor_and_full_body_guide_are_rendered(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "동작을 읽고,<br>리듬을 기록하다" in response.text
    assert "헤아리오" not in response.text
    assert 'id="profile-form"' in response.text
    assert 'class="full-body-guide"' in response.text


def test_industrial_telemetry_redesign_assets_and_accessibility_are_rendered(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/telemetry.css"' in response.text
    assert 'src="/vendor/gsap.min.js"' in response.text
    assert 'src="/vendor/ScrollTrigger.min.js"' in response.text
    assert 'class="skip-link"' in response.text
    assert 'id="mode-grid"' in response.text
    assert 'class="metric-strip"' in response.text
    assert 'role="progressbar"' in response.text

    stylesheet = client.get("/telemetry.css")
    assert stylesheet.status_code == 200
    assert "grid-auto-flow: dense" in stylesheet.text
    assert "prefers-reduced-motion: reduce" in stylesheet.text
    assert "transition: all" not in stylesheet.text

    assert client.get("/vendor/gsap.min.js").status_code == 200
    assert client.get("/vendor/ScrollTrigger.min.js").status_code == 200


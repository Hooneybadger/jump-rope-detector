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
    assert 'id="camera-guide"' in response.text
    assert 'href="#i-person-simple"' in response.text


def test_arena_redesign_assets_and_accessibility_are_rendered(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'href="/app.css"' in html
    assert "telemetry.css" not in html
    assert 'src="/vendor/gsap.min.js"' in html
    assert 'class="skip-link"' in html
    assert 'id="mode-grid"' in html
    assert 'role="progressbar"' in html
    assert "—" not in html and "–" not in html

    stylesheet = client.get("/app.css").text
    assert "prefers-reduced-motion: reduce" in stylesheet
    assert "prefers-color-scheme: dark" in stylesheet
    assert "transition: all" not in stylesheet
    for asset in ("/img/court-hero.jpg", "/img/court-ambient.jpg", "/fonts/BlackHanSans-Regular.woff2", "/fonts/BarlowCondensed-Bold.woff2"):
        assert client.get(asset).status_code == 200


def test_every_mode_shares_one_result_screen(client):
    html = client.get("/").text
    for element_id in ("result-overlay", "result-count", "result-mode", "result-status", "result-chart", "result-retry", "result-download"):
        assert f'id="{element_id}"' in html
    script = client.get("/app.js").text
    assert "function showResult(message)" in script
    assert "message.analysisFps" in script
    assert "if (!message.finished) scheduleFrame" in script


def test_page_and_code_are_revalidated_after_rebuild(client):
    for path in ("/", "/index.html", "/app.js", "/app.css"):
        assert client.get(path).headers["Cache-Control"] == "no-cache"
    assert "Cache-Control" not in client.get("/img/court-hero-1280.jpg").headers

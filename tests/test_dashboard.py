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


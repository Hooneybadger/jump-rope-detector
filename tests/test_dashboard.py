def test_empty_dashboard_has_stable_shape(admin_client):
    response = admin_client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json() == {
        "summary": {"count": 0, "duration": 0, "sessions": 0},
        "recent": [],
    }


def test_unknown_static_route_returns_404(client):
    response = client.get("/not-a-real-file.txt")
    assert response.status_code == 404


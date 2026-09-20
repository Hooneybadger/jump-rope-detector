import os

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./test_jump_rope.db"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "correct-horse-battery-staple"
os.environ["ALLOWED_HOSTS"] = "testserver,localhost,127.0.0.1"
os.environ["ALLOWED_ORIGINS"] = "http://testserver,http://localhost:8080"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app


@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app, base_url="http://testserver") as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def admin_client(client):
    response = client.post("/api/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    assert response.status_code == 200
    client.headers["X-CSRF-Token"] = response.json()["csrfToken"]
    return client

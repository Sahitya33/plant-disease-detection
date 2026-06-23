"""
Basic tests. Run with: pytest

Notice we use an in-memory SQLite database ('sqlite:///:memory:') for
tests instead of the real predictions.db — tests should never touch
your real data, and an in-memory DB is wiped automatically and is fast.
"""

import pytest
from app import create_app, db


class TestConfig:
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TESTING = True
    SECRET_KEY = "test"
    UPLOAD_FOLDER = "/tmp/test_uploads"
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MODEL_PATH = "nonexistent.h5"
    CLASS_NAMES_PATH = "nonexistent.json"
    IMAGE_SIZE = (224, 224)


@pytest.fixture
def client():
    app = create_app(config_object=TestConfig)
    with app.test_client() as client:
        yield client


def test_index_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Plant Disease Detector" in response.data


def test_predict_without_file_redirects(client):
    response = client.post("/predict", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"choose an image" in response.data


def test_prediction_model_to_dict():
    from app.models import Prediction
    from datetime import datetime

    p = Prediction(
        filename="leaf.jpg",
        predicted_class="Tomato___Late_blight",
        confidence=0.91,
        created_at=datetime(2026, 1, 1),
    )
    d = p.to_dict()
    assert d["predicted_class"] == "Tomato___Late_blight"
    assert d["confidence"] == 0.91

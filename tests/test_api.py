"""Inference smoke test against the real trained model and a held-out test image.
Skipped if the model artifact isn't present, so a fresh clone doesn't fail on a
large binary that isn't committed (see README for why the model isn't checked in).
"""
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

MODEL_PATH="models/steel_defect_unet.pt"
TEST_IMAGE=os.path.join(os.path.dirname(__file__), "..", "severstal-steel-defect-detection", "test_images", "0000f269f.jpg")

pytestmark=pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="trained model artifact not present, run training/train.py first")

client=TestClient(app)


def test_health():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()=={"status":"healthy"}


def test_predict_on_a_held_out_test_image():
    with open(TEST_IMAGE,"rb") as f:
        response=client.post("/predict", files={"file":("0000f269f.jpg", f, "image/jpeg")})

    assert response.status_code==200
    body=response.json()

    assert "defect_present" in body
    assert isinstance(body["defect_present"], bool)
    assert len(body["classes"])==4
    for c in body["classes"]:
        assert 0<=c["area_fraction"]<=1
        assert 0<=c["max_confidence"]<=1

    assert body["grade"]["letter"] in ("A","B","C","D")
    assert 0<=body["grade"]["score"]<=100


def test_predict_rejects_an_undecodable_file():
    response=client.post("/predict", files={"file":("not-an-image.jpg", b"this is not image data", "image/jpeg")})
    assert response.status_code==400

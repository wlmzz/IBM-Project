from src.api import app

def test_health():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

def test_predict():
    client = app.test_client()
    payload = {
        "customers": [
            {"country": "US", "tenure": 5, "monthly_charges": 25.0, "num_streams": 40},
            {"country": "Singapore", "tenure": 2, "monthly_charges": 18.0, "num_streams": 10},
        ]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    payload = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "prediction_label" in data
    assert "failure_probability" in data
    assert "threshold" in data

    assert data["prediction"] in [0, 1]


def test_invalid_sensor_type():
    payload = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": "hello",
        "tool_wear": 100
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 400


def test_invalid_machine_type():
    payload = {
        "type": "X",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 400
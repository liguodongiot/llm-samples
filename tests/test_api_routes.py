import pytest
from fastapi.testclient import TestClient
from matmul_ops.api.main import app

client = TestClient(app)

def test_matmul_success():
    response = client.post("/matmul", json={
        "A": [[1.0, 2.0], [3.0, 4.0]],
        "B": [[5.0, 6.0], [7.0, 8.0]]
    })
    assert response.status_code == 200
    assert response.json() == {"result": [[19.0, 22.0], [43.0, 50.0]]}

def test_matmul_invalid_dimensions():
    response = client.post("/matmul", json={
        "A": [[1.0, 2.0]],
        "B": [[1.0], [2.0], [3.0]]
    })
    assert response.status_code == 400

def test_matmul_invalid_dtype():
    response = client.post("/matmul", json={
        "A": [[1, 2]],
        "B": [[3], [4]]
    })
    assert response.status_code == 422

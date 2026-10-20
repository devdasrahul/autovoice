"""Integration test suite for the AutoVoice FastAPI service."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from autovoice.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate TestClient for FastAPI app with lifespan events triggered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Test GET /health returns 200 and expected schema."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert "version" in data
    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-Ms" in resp.headers


def test_ready_endpoint(client: TestClient) -> None:
    """Test GET /ready returns 200 when processed datasets exist."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["ready"] is True


def test_predict_ner_endpoint(client: TestClient) -> None:
    """Test POST /v1/ner extracts entities with offsets and scores."""
    payload = {
        "text": "The power steering failed at 65 mph causing loss of control and brake failure."
    }
    resp = client.post("/v1/ner", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert isinstance(data["entities"], list)
    assert len(data["entities"]) > 0
    assert "execution_time_ms" in data

    first = data["entities"][0]
    assert "start" in first
    assert "end" in first
    assert "label" in first
    assert "score" in first
    assert any(sub in first["label"] for sub in ("COMPONENT", "SYMPTOM"))


def test_predict_severity_endpoint(client: TestClient) -> None:
    """Test POST /v1/classify classifies triage severity."""
    payload = {
        "text": "Vehicle caught fire after collision with guardrail. Ambulance was called for burns."
    }
    resp = client.post("/v1/classify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_severity"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert set(data["probabilities"].keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert "hazard_indicators" in data
    assert "fire" in data["hazard_indicators"]
    assert "crash" in data["hazard_indicators"]
    assert "injury" in data["hazard_indicators"]


def test_analyze_endpoint(client: TestClient) -> None:
    """Test POST /v1/analyze returns unified NER and severity output."""
    payload = {"text": "Airbag ruptured during front-end crash with injuries reported."}
    resp = client.post("/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert "predicted_severity" in data
    assert "hazard_indicators" in data
    assert "execution_time_ms" in data


def test_ask_endpoint_factual(client: TestClient) -> None:
    """Test POST /v1/ask handles factual complaint lookup."""
    payload = {"question": "What issues have been reported for steering failures?"}
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "citations" in data
    assert "step_trace" in data
    assert data["intent"] in ("semantic_search", "stats_aggregate", "out_of_scope")


def test_ask_endpoint_out_of_scope_guard(client: TestClient) -> None:
    """Test POST /v1/ask properly triggers safety guard on non-automotive queries."""
    payload = {"question": "Can you write a poem about flowers?"}
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["refusal"] is True
    assert "designed specifically to analyze vehicle defect complaints" in data["answer"]


def test_metrics_summary_endpoint(client: TestClient) -> None:
    """Test GET /v1/metrics-summary returns verified report data."""
    resp = client.get("/v1/metrics-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "data_quality" in data
    assert "ner_metrics" in data
    assert "clf_metrics" in data
    assert "rag_metrics" in data


def test_model_info_endpoint(client: TestClient) -> None:
    """Test GET /v1/model-info returns runtime architecture metadata."""
    resp = client.get("/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "autovoice"
    assert "version" in data
    assert "llm_provider" in data


def test_validation_errors(client: TestClient) -> None:
    """Test validation errors for empty / too-short input."""
    # Too short text (< 3 chars)
    resp = client.post("/v1/ner", json={"text": "hi"})
    assert resp.status_code == 422

    resp = client.post("/v1/classify", json={"text": "no"})
    assert resp.status_code == 422

    resp = client.post("/v1/ask", json={"question": "a"})
    assert resp.status_code == 422


import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent

from agent import QueryRequest, ComplianceManager, AuditLogger, LLMService, ErrorHandler, AgentOrchestrator, sanitize_llm_output, app

from fastapi.testclient import TestClient

# =========================
# UNIT TESTS
# =========================

def test_valid_queryrequest_model_validation():
    """Ensures QueryRequest model accepts valid input and strips whitespace."""
    data = {
        "user_query": "  How do I admit a new patient?  ",
        "user_role": "  clinician ",
        "session_id": " 123e4567-e89b-12d3-a456-426614174000 "
    }
    req = QueryRequest(**data)
    assert isinstance(req, QueryRequest)
    assert req.user_query == "How do I admit a new patient?"
    assert req.user_role == "clinician"
    assert req.session_id == "123e4567-e89b-12d3-a456-426614174000"

def test_queryrequest_model_validation_fails_on_empty_fields():
    """Ensures QueryRequest model raises validation errors for empty fields."""
    data = {
        "user_query": "   ",
        "user_role": "",
        "session_id": "   "
    }
    with pytest.raises(Exception) as excinfo:
        QueryRequest(**data)
    err = str(excinfo.value)
    assert "user_query" in err or "user_role" in err or "session_id" in err

def test_compliancemanager_role_validation_success():
    """Checks ComplianceManager.validate_user_role returns True for allowed roles."""
    logger = AuditLogger()
    cm = ComplianceManager(logger)
    assert cm.validate_user_role("clinician") is True
    assert cm.validate_user_role("admin") is True
    assert cm.validate_user_role("nurse") is True
    assert cm.validate_user_role("physician") is True
    assert cm.validate_user_role("staff") is True

def test_compliancemanager_role_validation_failure():
    """Checks ComplianceManager.validate_user_role returns False and logs for disallowed roles."""
    logger = MagicMock(spec=AuditLogger)
    cm = ComplianceManager(logger)
    result = cm.validate_user_role("visitor")
    assert result is False
    logger.log_event.assert_any_call("ERR_UNAUTHORIZED_ACCESS", {"user_role": "visitor"})

def test_compliancemanager_pii_redaction():
    """Ensures redact_pii redacts emails, SSNs, and phone numbers."""
    logger = MagicMock(spec=AuditLogger)
    cm = ComplianceManager(logger)
    # Patch PIIDetector to simulate detection
    with MagicMock() as mock_detector_cls:
        mock_detector = MagicMock()
        mock_detector.detect.return_value = {
            "email": ["john.doe@example.com"],
            "ssn": ["123-45-6789"],
            "phone": ["555-123-4567"]
        }
        mock_detector_cls.return_value = mock_detector
        response = "Patient email: john.doe@example.com, SSN: 123-45-6789, Phone: 555-123-4567"
        redacted = cm.redact_pii(response)
        assert "[EMAIL_REDACTED]" in redacted
        assert "[SSN_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted
        logger.log_event.assert_any_call("ModuleType", {"pii_types": ["email", "ssn", "phone"]})

@pytest.mark.asyncio
async def test_llmservice_generate_response_handles_api_error():
    """Ensures LLMService.generate_response returns fallback error on LLM API failure."""
    svc = LLMService()
    with patch.object(svc, "get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
        with pytest.raises(RuntimeError) as excinfo:
            await svc.generate_response("prompt", "context")
        assert "LLM API error" in str(excinfo.value)

@pytest.mark.asyncio
async def test_errorhandler_handle_error_privacy_violation():
    """Checks ErrorHandler.handle_error returns correct structure for ERR_PRIVACY_VIOLATION."""
    logger = MagicMock(spec=AuditLogger)
    handler = ErrorHandler(logger)
    result = await handler.handle_error("ERR_PRIVACY_VIOLATION", {"error": "PII detected"})
    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["error_type"] == "ERR_PRIVACY_VIOLATION"
    assert "error" in result
    assert "tips" in result

def test_edge_case_oversized_user_query():
    """Ensures QueryRequest rejects user_query over 5000 characters."""
    data = {
        "user_query": "a" * 5001,
        "user_role": "clinician",
        "session_id": "123e4567-e89b-12d3-a456-426614174000"
    }
    with pytest.raises(Exception) as excinfo:
        QueryRequest(**data)
    assert "user_query exceeds 5000 characters" in str(excinfo.value)

def test_edge_case_llmservice_missing_api_key(monkeypatch):
    """Ensures LLMService.get_client raises ValueError if AZURE_OPENAI_API_KEY is missing."""
    monkeypatch.setattr(agent.Config, "AZURE_OPENAI_API_KEY", "")
    svc = LLMService()
    with pytest.raises(ValueError) as excinfo:
        svc.get_client()
    assert "AZURE_OPENAI_API_KEY" in str(excinfo.value)

def test_edge_case_compliancemanager_redact_pii_exception_handling():
    """Ensures redact_pii logs and raises ValueError on exception."""
    logger = MagicMock(spec=AuditLogger)
    cm = ComplianceManager(logger)
    with MagicMock() as mock_detector_cls:
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = Exception("PII detection failed")
        mock_detector_cls.return_value = mock_detector
        with pytest.raises(ValueError) as excinfo:
            cm.redact_pii("Sensitive info")
        assert "PII redaction failed" in str(excinfo.value)
        logger.log_event.assert_any_call("ERR_PRIVACY_VIOLATION", {"error": "PII detection failed"})

# =========================
# INTEGRATION TESTS
# =========================

@pytest.mark.asyncio
async def test_agentorchestrator_process_user_query_happy_path():
    """Tests full workflow for a valid user query (no escalation, no errors)."""
    orch = AgentOrchestrator()
    # Patch LLMService.generate_response to return a canned response
    with patch.object(orch.llm_service, "generate_response", new=AsyncMock(return_value="Admit patient steps...")), \
         patch.object(orch.compliance_manager, "redact_pii", side_effect=lambda x: x):
        result = await orch.process_user_query(
            user_query="How do I admit a new patient?",
            user_role="clinician",
            session_id="123e4567-e89b-12d3-a456-426614174000"
        )
        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert result["response"]
        assert result.get("escalation") is False
        assert "error" not in result or result["error"] is None

@pytest.mark.asyncio
async def test_agentorchestrator_process_user_query_triggers_escalation():
    """Tests escalation logic for clinical query by unauthorized role."""
    orch = AgentOrchestrator()
    # Patch LLMService.generate_response to avoid LLM call
    with patch.object(orch.llm_service, "generate_response", new=AsyncMock(return_value="Should not be called")), \
         patch.object(orch.compliance_manager, "redact_pii", side_effect=lambda x: x):
        result = await orch.process_user_query(
            user_query="What is the best treatment for hypertension?",
            user_role="staff",
            session_id="123e4567-e89b-12d3-a456-426614174000"
        )
        assert result["success"] is True
        assert result.get("escalation") is True
        assert "forwarded to a human operator" in result["response"]

@pytest.mark.asyncio
async def test_agentorchestrator_process_user_query_unauthorized_role():
    """Ensures unauthorized user_role triggers error response."""
    orch = AgentOrchestrator()
    result = await orch.process_user_query(
        user_query="How do I admit a new patient?",
        user_role="visitor",
        session_id="123e4567-e89b-12d3-a456-426614174000"
    )
    assert result["success"] is False
    assert result.get("error_type") == "ERR_UNAUTHORIZED_ACCESS"
    assert "error" in result and result["error"]

# =========================
# FUNCTIONAL TESTS
# =========================

@pytest.fixture(scope="module")
def test_client():
    """Fixture for FastAPI TestClient."""
    with TestClient(app) as client:
        yield client

def test_query_endpoint_functional_success(test_client):
    """Tests /query endpoint returns valid QueryResponse for valid input."""
    payload = {
        "user_query": "How do I admit a new patient?",
        "user_role": "clinician",
        "session_id": "123e4567-e89b-12d3-a456-426614174000"
    }
    with patch.object(agent.AgentOrchestrator, "process_user_query", new=AsyncMock(return_value={
        "success": True,
        "response": "Admit patient steps...",
        "escalation": False
    })):
        resp = test_client.post("/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["response"], str)
        assert data["escalation"] is False

def test_query_endpoint_functional_escalation(test_client):
    """Tests /query endpoint returns escalation for clinical query by unauthorized role."""
    payload = {
        "user_query": "What is the best treatment for hypertension?",
        "user_role": "staff",
        "session_id": "123e4567-e89b-12d3-a456-426614174000"
    }
    with patch.object(agent.AgentOrchestrator, "process_user_query", new=AsyncMock(return_value={
        "success": True,
        "response": "This query will be forwarded to a human operator for further assistance.\n\n[Escalation Notice: This query will be forwarded to a human operator.]",
        "escalation": True
    })):
        resp = test_client.post("/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["escalation"] is True
        assert "forwarded to a human operator" in data["response"]

def test_query_endpoint_functional_unauthorized_role(test_client):
    """Tests /query endpoint returns error for unauthorized user_role."""
    payload = {
        "user_query": "How do I admit a new patient?",
        "user_role": "visitor",
        "session_id": "123e4567-e89b-12d3-a456-426614174000"
    }
    with patch.object(agent.AgentOrchestrator, "process_user_query", new=AsyncMock(return_value={
        "success": False,
        "error_type": "ERR_UNAUTHORIZED_ACCESS",
        "error": "You are not authorized to perform this action."
    })):
        resp = test_client.post("/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error_type"] == "ERR_UNAUTHORIZED_ACCESS"

def test_query_endpoint_functional_input_validation_error(test_client):
    """Tests /query endpoint returns validation error for missing fields."""
    payload = {
        "user_query": "",
        "user_role": "",
        "session_id": ""
    }
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["success"] is False
    assert data["error_type"] == "VALIDATION_ERROR"

def test_health_check_endpoint(test_client):
    """Tests /health endpoint returns status ok."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
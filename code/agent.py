import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': True,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, Any, Dict, List
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path

from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are the Healthcare Operations Assistant, an expert virtual agent supporting healthcare professionals and administrative staff. "
    "Your responsibilities include answering user questions, guiding users through healthcare workflows, retrieving and summarizing relevant information, "
    "and escalating complex or sensitive queries to human operators when necessary.\n\n"
    "Instructions:\n\n"
    "- Communicate in a professional, clear, and concise manner.\n"
    "- Use accurate healthcare terminology and adhere to best-practice workflows.\n"
    "- Do not provide medical diagnoses or treatment recommendations.\n"
    "- Ensure all responses comply with healthcare privacy and security standards (e.g., HIPAA).\n"
    "- If a query is outside your scope or involves sensitive clinical decisions, escalate to a human operator.\n"
    "- If you do not have sufficient information to answer, politely inform the user and suggest next steps.\n\n"
    "Output Format:\n"
    "- Provide direct, actionable answers or guidance.\n"
    "- Summarize information clearly when responding to data lookup or workflow queries.\n"
    "- For escalations, clearly state that the query will be forwarded to a human operator.\n\n"
    "Fallback Response:\n"
    "- If you cannot find the requested information or the query is outside your scope, respond: \"I'm unable to provide the information you requested. Please contact a qualified healthcare professional or your system administrator for further assistance.\""
)
OUTPUT_FORMAT = (
    "- Direct answer or step-by-step guidance in text format.\n"
    "- Summaries for data lookup or workflow queries.\n"
    "- Escalation notice when forwarding to human operator."
)
FALLBACK_RESPONSE = (
    "I'm unable to provide the information you requested. Please contact a qualified healthcare professional or your system administrator for further assistance."
)
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# LLM OUTPUT SANITIZER
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# REQUEST/RESPONSE MODELS
# =========================

class QueryRequest(BaseModel):
    user_query: str = Field(..., description="Healthcare-related question or workflow need (max 5000 chars)")
    user_role: str = Field(..., description="User role (e.g., admin, clinician, nurse, staff)")
    session_id: str = Field(..., description="Session identifier (UUID or string)")

    @field_validator("user_query")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_user_query(cls, v):
        if not v or not v.strip():
            raise ValueError("user_query must not be empty")
        if len(v) > 5000:
            raise ValueError("user_query exceeds 5000 characters")
        return v.strip()

    @field_validator("user_role")
    @classmethod
    def validate_user_role(cls, v):
        if not v or not v.strip():
            raise ValueError("user_role must not be empty")
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        if not v or not v.strip():
            raise ValueError("session_id must not be empty")
        return v.strip()

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    response: Optional[str] = Field(None, description="Agent's answer or guidance")
    escalation: Optional[bool] = Field(False, description="Whether escalation to human operator is required")
    error: Optional[str] = Field(None, description="Error message if any")
    error_type: Optional[str] = Field(None, description="Error type if any")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input or retrying")

# =========================
# AUDIT LOGGER
# =========================

class AuditLogger:
    """Logs all access, workflow actions, and errors for compliance and monitoring."""

    def __init__(self):
        self.logger = logging.getLogger("audit_logger")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: Dict[str, Any]):
        try:
            self.logger.info(f"[{event_type}] {json.dumps(details, default=str)}")
        except Exception as e:
            self.logger.warning(f"Primary audit log failed: {e}. Fallback: {event_type} - {details}")

# =========================
# COMPLIANCE MANAGER
# =========================

class ComplianceManager:
    """Ensures HIPAA compliance, manages PII redaction, role-based access, and audit logging."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.allowed_roles = {"admin", "clinician", "physician", "nurse", "staff"}

    def redact_pii(self, response: str) -> str:
        """Scan and redact PII from responses."""
        try:
            # Use the guardrails PII detector (runtime module)
            from modules.guardrails.guardrails_service import PIIDetector
            detector = PIIDetector()
            detected = detector.detect(response)
            sanitized = response
            if detected:
                # Redact all detected PII types
                for pii_type, matches in detected.items():
                    for match in matches:
                        sanitized = sanitized.replace(match, f"[{pii_type.upper()}_REDACTED]")
                self.audit_logger.log_event("PII_REDACTED", {"pii_types": list(detected.keys())})
            return sanitized
        except Exception as e:
            self.audit_logger.log_event("ERR_PRIVACY_VIOLATION", {"error": str(e)})
            raise ValueError("PII redaction failed (ERR_PRIVACY_VIOLATION)")

    def validate_user_role(self, user_role: str) -> bool:
        """Check if user role is authorized for requested action."""
        if user_role.lower() not in self.allowed_roles:
            self.audit_logger.log_event("ERR_UNAUTHORIZED_ACCESS", {"user_role": user_role})
            return False
        return True

    def log_access(self, event: str, details: Dict[str, Any]):
        self.audit_logger.log_event(event, details)

# =========================
# ERROR HANDLER
# =========================

class ErrorHandler:
    """Handles errors, manages retries, fallback responses, and escalation triggers."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def handle_error(self, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Centralized error handling with logging and fallback."""
        self.audit_logger.log_event("ERROR", {"error_type": error_type, "context": context})
        # Fallback/escalation logic
        if error_type == "ERR_PRIVACY_VIOLATION":
            return {
                "success": False,
                "error": "A privacy violation was detected. The operation was stopped.",
                "error_type": error_type,
                "tips": "Ensure no sensitive PII is present in your input or output."
            }
        elif error_type == "ERR_UNAUTHORIZED_ACCESS":
            return {
                "success": False,
                "error": "You are not authorized to perform this action.",
                "error_type": error_type,
                "tips": "Contact your system administrator if you believe this is an error."
            }
        else:
            return {
                "success": False,
                "error": FALLBACK_RESPONSE,
                "error_type": error_type,
                "tips": "Try rephrasing your query or contact support if the issue persists."
            }

    async def retry_operation(self, operation, attempts: int = 3, backoff_strategy: float = 0.5, *args, **kwargs):
        """Retry logic with exponential backoff."""
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                self.audit_logger.log_event("RETRY", {"attempt": attempt, "error": str(e)})
                last_exc = e
                await asyncio.sleep(backoff_strategy * attempt)
        raise last_exc

# =========================
# KNOWLEDGE BASE API (NO RAG)
# =========================

class KnowledgeBaseAPI:
    """Retrieves relevant healthcare information and best practices from internal documentation and industry guidelines."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def fetch_information(self, user_query: str) -> Optional[str]:
        """
        Placeholder for knowledge base retrieval.
        In this agent, RAG is disabled (no source documents), so always returns None.
        """
        self.audit_logger.log_event("KNOWLEDGE_BASE_LOOKUP", {"query": user_query})
        return None

# =========================
# WORKFLOW ENGINE
# =========================

class WorkflowEngine:
    """Guides users through healthcare workflows, manages escalation logic, and tracks workflow completion."""

    def __init__(self, audit_logger: AuditLogger, compliance_manager: ComplianceManager):
        self.audit_logger = audit_logger
        self.compliance_manager = compliance_manager

    async def execute_workflow(self, user_query: str, user_role: str, session_id: str) -> Dict[str, Any]:
        """
        Dummy workflow logic: For demonstration, escalate if query contains 'escalate' or is clinical and user_role is not physician/nurse.
        """
        # Determine query type (simple heuristic)
        query_type = "clinical_decision" if any(word in user_query.lower() for word in ["diagnose", "prescribe", "treatment", "clinical"]) else "administrative"
        escalation_required = False
        if query_type == "clinical_decision" and user_role.lower() not in {"physician", "nurse"}:
            escalation_required = True
        if "escalate" in user_query.lower():
            escalation_required = True

        self.audit_logger.log_event("WORKFLOW_EXECUTION", {
            "query": user_query,
            "user_role": user_role,
            "session_id": session_id,
            "query_type": query_type,
            "escalation_required": escalation_required
        })

        if escalation_required:
            return {
                "escalation": True,
                "message": "This query will be forwarded to a human operator for further assistance."
            }
        else:
            return {
                "escalation": False,
                "message": None
            }

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def escalate_to_human(self, query: str, context: Dict[str, Any]) -> str:
        self.audit_logger.log_event("ESCALATION", {"query": query, "context": context})
        return "This query will be forwarded to a human operator for further assistance."

# =========================
# LLM SERVICE
# =========================

import openai
import asyncio

class LLMService:
    """Handles interaction with Azure OpenAI GPT-4.1."""

    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(self, prompt: str, context: Optional[str], parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Calls LLM with enhanced prompt, user query, and context.
        """
        client = self.get_client()
        model = Config.LLM_MODEL or "gpt-4.1"
        _llm_kwargs = Config.get_llm_kwargs()
        messages = [
            {"role": "system", "content": prompt + "\n\nOutput Format: " + OUTPUT_FORMAT},
        ]
        if context:
            messages.append({"role": "user", "content": context})
        try:
            _t0 = _time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=model,
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")

# =========================
# AGENT ORCHESTRATOR
# =========================

class AgentOrchestrator:
    """Coordinates user input processing, workflow guidance, LLM calls, and response formatting."""

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.compliance_manager = ComplianceManager(self.audit_logger)
        self.error_handler = ErrorHandler(self.audit_logger)
        self.knowledge_base = KnowledgeBaseAPI(self.audit_logger)
        self.workflow_engine = WorkflowEngine(self.audit_logger, self.compliance_manager)
        self.llm_service = LLMService()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_query(self, user_query: str, user_role: str, session_id: str) -> Dict[str, Any]:
        """
        Entry point for handling user queries, orchestrates workflow, LLM, and compliance checks.
        """
        async with trace_step(
            "validate_input", step_type="parse",
            decision_summary="Validate session and user role",
            output_fn=lambda r: f"valid={r}",
        ) as step:
            # Validate user role
            if not self.compliance_manager.validate_user_role(user_role):
                return await self.error_handler.handle_error("ERR_UNAUTHORIZED_ACCESS", {"user_role": user_role})
            # Validate session_id (basic check)
            if not session_id or not session_id.strip():
                return {
                    "success": False,
                    "error": "Session ID is required.",
                    "error_type": "ERR_SESSION_REQUIRED",
                    "tips": "Provide a valid session identifier."
                }
            step.capture(True)

        async with trace_step(
            "sanitize_input", step_type="parse",
            decision_summary="Sanitize input for PII",
            output_fn=lambda r: f"sanitized={r}",
        ) as step:
            try:
                sanitized_query = self.compliance_manager.redact_pii(user_query)
                step.capture(True)
            except Exception as e:
                return await self.error_handler.handle_error("ERR_PRIVACY_VIOLATION", {"error": str(e)})

        async with trace_step(
            "workflow_and_escalation", step_type="plan",
            decision_summary="Determine query type and escalation",
            output_fn=lambda r: f"escalation={r.get('escalation', False)}",
        ) as step:
            workflow_result = await self.workflow_engine.execute_workflow(sanitized_query, user_role, session_id)
            step.capture(workflow_result)
            if workflow_result.get("escalation"):
                escalation_msg = await self.workflow_engine.escalate_to_human(sanitized_query, {
                    "user_role": user_role,
                    "session_id": session_id
                })
                formatted = self.format_response(escalation_msg, escalation=True)
                self.compliance_manager.log_access("ESCALATION_TRIGGERED", {
                    "user_query": sanitized_query,
                    "user_role": user_role,
                    "session_id": session_id
                })
                return {
                    "success": True,
                    "response": formatted,
                    "escalation": True
                }

        async with trace_step(
            "knowledge_base_lookup", step_type="tool_call",
            decision_summary="Attempt knowledge base retrieval",
            output_fn=lambda r: f"kb_hit={bool(r)}",
        ) as step:
            kb_info = await self.knowledge_base.fetch_information(sanitized_query)
            step.capture(kb_info)
            context_for_llm = kb_info if kb_info else sanitized_query

        async with trace_step(
            "llm_response", step_type="llm_call",
            decision_summary="Call LLM for answer/guidance",
            output_fn=lambda r: f"llm_response={str(r)[:60]}",
        ) as step:
            try:
                llm_raw = await self.llm_service.generate_response(SYSTEM_PROMPT, context_for_llm)
                llm_clean = sanitize_llm_output(llm_raw, content_type="text")
                step.capture(llm_clean)
            except Exception as e:
                return await self.error_handler.handle_error("LLM_API_ERROR", {"error": str(e)})

        async with trace_step(
            "redact_output", step_type="process",
            decision_summary="Redact PII from output",
            output_fn=lambda r: f"redacted={str(r)[:60]}",
        ) as step:
            try:
                final_response = self.compliance_manager.redact_pii(llm_clean)
                step.capture(final_response)
            except Exception as e:
                return await self.error_handler.handle_error("ERR_PRIVACY_VIOLATION", {"error": str(e)})

        async with trace_step(
            "format_response", step_type="format",
            decision_summary="Format and return response",
            output_fn=lambda r: f"formatted={str(r)[:60]}",
        ) as step:
            formatted = self.format_response(final_response)
            step.capture(formatted)
            self.compliance_manager.log_access("QUERY_PROCESSED", {
                "user_query": sanitized_query,
                "user_role": user_role,
                "session_id": session_id
            })
            return {
                "success": True,
                "response": formatted,
                "escalation": False
            }

    def format_response(self, output: str, escalation: bool = False) -> str:
        """Format response according to output instructions."""
        if escalation:
            return f"{output}\n\n[Escalation Notice: This query will be forwarded to a human operator.]"
        return output

# =========================
# MAIN AGENT CLASS
# =========================

class HealthcareOperationsAssistant:
    """Main agent class."""

    def __init__(self):
        self.orchestrator = AgentOrchestrator()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def run(self, user_query: str, user_role: str, session_id: str) -> Dict[str, Any]:
        """Entrypoint for agent execution."""
        async with trace_step(
            "process_user_query", step_type="process",
            decision_summary="Orchestrate user query processing",
            output_fn=lambda r: f"success={r.get('success', False)}",
        ) as step:
            result = await self.orchestrator.process_user_query(user_query, user_role, session_id)
            step.capture(result)
            return result

# =========================
# FASTAPI APP & OBSERVABILITY LIFESPAN
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Healthcare Operations Assistant",
    description="A professional virtual agent for healthcare operations, workflow guidance, and compliance.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# =========================
# ERROR HANDLING FOR FASTAPI
# =========================

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed request. Please check your input JSON.",
            "error_type": "VALIDATION_ERROR",
            "tips": "Ensure your JSON is well-formed and all required fields are present.",
            "details": exc.errors()
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed request. Please check your input JSON.",
            "error_type": "VALIDATION_ERROR",
            "tips": "Ensure your JSON is well-formed and all required fields are present.",
            "details": exc.errors()
        }
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": "Malformed JSON in request body.",
            "error_type": "JSON_DECODE_ERROR",
            "tips": "Check for missing quotes, commas, or brackets in your JSON."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An unexpected error occurred.",
            "error_type": "INTERNAL_ERROR",
            "tips": "Try again later or contact support."
        }
    )

# =========================
# ENDPOINTS
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

agent_instance = HealthcareOperationsAssistant()

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """
    Main endpoint for Healthcare Operations Assistant.
    """
    try:
        result = await agent_instance.run(
            user_query=req.user_query,
            user_role=req.user_role,
            session_id=req.session_id
        )
        # Ensure LLM output is sanitized before returning
        if result.get("response"):
            result["response"] = sanitize_llm_output(result["response"], content_type="text")
        return result
    except Exception as e:
        logging.getLogger(__name__).exception("Agent error")
        return {
            "success": False,
            "error": str(e),
            "error_type": "AGENT_ERROR",
            "tips": "Try again later or contact support."
        }

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())
"""FastAPI service for AutoVoice automotive complaint intelligence.

Exposes endpoints for NER entity extraction, severity classification,
comprehensive complaint triage, and multi-step LangGraph RAG inquiry.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from autovoice.api.deps import (
    get_app_settings,
    get_ner_inferencer,
    get_rag_agent,
    get_severity_classifier,
    warmup_all_models,
)
from autovoice.api.middleware import RateLimitMiddleware, RequestIDAndTimingMiddleware
from autovoice.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AskRequest,
    AskResponse,
    EntitySpan,
    HealthResponse,
    MetricsSummaryResponse,
    ModelInfoResponse,
    PredictNERRequest,
    PredictNERResponse,
    PredictSeverityRequest,
    PredictSeverityResponse,
)
from autovoice.classifier.infer import SeverityClassifierInferencer
from autovoice.config import DATA_PROCESSED_DIR, REPORTS_DIR, Settings
from autovoice.logging_conf import configure_logging, get_logger
from autovoice.ner.infer import NERInferencer
from autovoice.rag.graph import AutoVoiceRAGAgent

# Initialize structured logging
configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan manager executing pre-warmup on startup."""
    log.info("api.starting_up")
    try:
        warmup_res = warmup_all_models()
        log.info("api.models_warmed", status=warmup_res)
    except Exception as exc:
        log.warning("api.warmup_warning", error=str(exc))
    yield
    log.info("api.shutting_down")


app = FastAPI(
    title="AutoVoice API - Automotive Complaint Intelligence",
    description=(
        "Production-grade NLP and GenAI service ingesting and analyzing NHTSA vehicle safety complaints. "
        "Provides Named Entity Recognition for components/symptoms, triage severity classification, "
        "and a LangGraph RAG agent with verified citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Apply settings and middleware
_settings = get_app_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins if _settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDAndTimingMiddleware)


# ---------------------------------------------------------------------------
# Health & Probes
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness check",
)
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Kubernetes/Cloud Run liveness probe."""
    return HealthResponse(
        status="ok",
        ready=True,
        environment=settings.environment,
        version=settings.model_version,
    )


@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["System"],
    summary="Readiness check",
)
def ready(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Readiness probe checking availability of core processed artifacts."""
    parquet_exists = (DATA_PROCESSED_DIR / "complaints.parquet").exists()
    chroma_exists = (DATA_PROCESSED_DIR / "chroma").exists()

    if not (parquet_exists or chroma_exists):
        log.warning("api.readiness_check_failed", parquet=parquet_exists, chroma=chroma_exists)
        raise HTTPException(
            status_code=status.HTTP(503),
            detail="Service initializing: processed complaint dataset or vector index pending.",
        )

    return HealthResponse(
        status="ready",
        ready=True,
        environment=settings.environment,
        version=settings.model_version,
    )


# ---------------------------------------------------------------------------
# ML Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/v1/ner",
    response_model=PredictNERResponse,
    tags=["NLP Inference"],
    summary="Extract vehicle components and failure symptoms",
)
def predict_ner(
    req: PredictNERRequest,
    ner: NERInferencer = Depends(get_ner_inferencer),
) -> PredictNERResponse:
    """Extract named entity spans (COMPONENT, SYMPTOM) from complaint narratives."""
    t0 = time.perf_counter()
    try:
        raw_entities = ner.predict(req.text)
    except Exception as exc:
        log.error("api.ner_inference_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"NER inference failed: {exc}") from exc

    entities = [
        EntitySpan(
            start=e.start,
            end=e.end,
            label=e.label,
            text=e.text,
            score=round(e.score, 4),
        )
        for e in raw_entities
    ]
    elapsed = (time.perf_counter() - t0) * 1000.0

    return PredictNERResponse(
        text=req.text,
        entities=entities,
        execution_time_ms=round(elapsed, 2),
    )


@app.post(
    "/v1/classify",
    response_model=PredictSeverityResponse,
    tags=["NLP Inference"],
    summary="Classify triage severity and domain hazards",
)
def predict_severity(
    req: PredictSeverityRequest,
    clf: SeverityClassifierInferencer = Depends(get_severity_classifier),
) -> PredictSeverityResponse:
    """Predict triage severity level (LOW, MEDIUM, HIGH) with calibrated probabilities."""
    t0 = time.perf_counter()
    try:
        res = clf.predict(req.text)
    except Exception as exc:
        log.error("api.severity_classification_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Severity classifier failed: {exc}") from exc

    elapsed = (time.perf_counter() - t0) * 1000.0

    return PredictSeverityResponse(
        predicted_severity=res["predicted_severity"],
        confidence=res["confidence"],
        probabilities=res["probabilities"],
        hazard_indicators=res["hazard_indicators"],
        execution_time_ms=round(elapsed, 2),
    )


@app.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    tags=["NLP Inference"],
    summary="Combined comprehensive narrative triage",
)
def analyze(
    req: AnalyzeRequest,
    ner: NERInferencer = Depends(get_ner_inferencer),
    clf: SeverityClassifierInferencer = Depends(get_severity_classifier),
) -> AnalyzeResponse:
    """Execute both NER extraction and Severity classification in a single request."""
    t0 = time.perf_counter()
    raw_entities = ner.predict(req.text)
    clf_res = clf.predict(req.text)

    entities = [
        EntitySpan(
            start=e.start,
            end=e.end,
            label=e.label,
            text=e.text,
            score=round(e.score, 4),
        )
        for e in raw_entities
    ]
    elapsed = (time.perf_counter() - t0) * 1000.0

    return AnalyzeResponse(
        text=req.text,
        entities=entities,
        predicted_severity=clf_res["predicted_severity"],
        confidence=clf_res["confidence"],
        probabilities=clf_res["probabilities"],
        hazard_indicators=clf_res["hazard_indicators"],
        execution_time_ms=round(elapsed, 2),
    )


@app.post(
    "/v1/ask",
    response_model=AskResponse,
    tags=["GenAI & RAG"],
    summary="Query the LangGraph RAG Agent over NHTSA complaints",
)
def ask(
    req: AskRequest,
    agent: AutoVoiceRAGAgent = Depends(get_rag_agent),
) -> AskResponse:
    """Run multi-step LangGraph workflow with intent routing, citations, and safety boundaries."""
    t0 = time.perf_counter()
    try:
        rag_res = agent.run(req.question)
    except Exception as exc:
        log.error("api.rag_agent_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"RAG agent execution failed: {exc}") from exc

    elapsed = (time.perf_counter() - t0) * 1000.0

    return AskResponse(
        question=rag_res["question"],
        answer=rag_res["answer"],
        intent=rag_res["intent"],
        citations=rag_res["citations"],
        step_trace=rag_res["step_trace"],
        refusal=rag_res["refusal"],
        execution_time_ms=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Metadata & Metrics
# ---------------------------------------------------------------------------


@app.get(
    "/v1/metrics-summary",
    response_model=MetricsSummaryResponse,
    tags=["Metrics & Governance"],
    summary="Retrieve verified evaluation metrics",
)
def get_metrics_summary() -> MetricsSummaryResponse:
    """Load and return verified metric reports generated during training and evaluation."""

    def _read_report(name: str) -> dict:
        p = REPORTS_DIR / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    return MetricsSummaryResponse(
        data_quality=_read_report("data_quality.json"),
        ner_metrics=_read_report("ner_metrics.json"),
        clf_metrics=_read_report("clf_metrics.json"),
        rag_metrics=_read_report("rag_metrics.json"),
    )


@app.get(
    "/v1/model-info",
    response_model=ModelInfoResponse,
    tags=["Metrics & Governance"],
    summary="Get loaded model configurations and environment",
)
def get_model_info(settings: Settings = Depends(get_app_settings)) -> ModelInfoResponse:
    """Return model runtime architectures, storage paths, and active provider."""
    return ModelInfoResponse(
        app_name=settings.app_name,
        version=settings.model_version,
        environment=settings.environment,
        ner_model_path=settings.ner_model_path,
        clf_model_path=settings.clf_model_path,
        llm_provider=settings.llm_provider,
        chroma_dir=settings.chroma_dir,
    )

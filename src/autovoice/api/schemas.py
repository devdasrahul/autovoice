"""Pydantic v2 schemas for AutoVoice FastAPI service request and response payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness/readiness probe response."""

    status: str = Field(..., examples=["ok"])
    ready: bool = Field(..., examples=[True])
    environment: str = Field(..., examples=["dev"])
    version: str = Field(..., examples=["0.1.0"])


class EntitySpan(BaseModel):
    """Extracted named entity span with character offsets and model confidence."""

    start: int = Field(..., description="0-indexed start character offset in original text")
    end: int = Field(..., description="0-indexed exclusive end character offset")
    label: str = Field(..., description="Entity label (COMPONENT or SYMPTOM)")
    text: str = Field(..., description="Entity surface substring")
    score: float = Field(..., description="Model confidence score [0, 1]")


class PredictNERRequest(BaseModel):
    """Payload for Named Entity Recognition endpoint."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="NHTSA complaint narrative text to extract components and symptoms from",
        examples=[
            "The power steering failed at 65 mph causing loss of control and a minor collision."
        ],
    )


class PredictNERResponse(BaseModel):
    """Response containing extracted automotive entities."""

    text: str = Field(..., description="Input narrative text")
    entities: list[EntitySpan] = Field(default_factory=list, description="Extracted entity spans")
    execution_time_ms: float = Field(..., description="Endpoint processing time in milliseconds")


class PredictSeverityRequest(BaseModel):
    """Payload for Severity Classification endpoint."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="NHTSA complaint narrative text to classify triage severity",
        examples=[
            "Vehicle caught fire while driving on highway, resulting in ambulance dispatch and burns."
        ],
    )


class PredictSeverityResponse(BaseModel):
    """Response containing predicted severity class, calibrated distribution, and hazards."""

    predicted_severity: str = Field(..., description="Predicted severity level (LOW, MEDIUM, HIGH)")
    confidence: float = Field(..., description="Confidence probability for predicted class")
    probabilities: dict[str, float] = Field(
        ..., description="Full probability distribution over [LOW, MEDIUM, HIGH]"
    )
    hazard_indicators: list[str] = Field(
        default_factory=list, description="Detected safety hazards (e.g. fire, crash, injury)"
    )
    execution_time_ms: float = Field(..., description="Endpoint processing time in milliseconds")


class AnalyzeRequest(BaseModel):
    """Payload for combined NER + Severity triage endpoint."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="NHTSA complaint narrative to comprehensively analyze",
        examples=["Front passenger airbag ruptured without deployment during front-end crash."],
    )


class AnalyzeResponse(BaseModel):
    """Unified response combining entity extraction and severity classification."""

    text: str
    entities: list[EntitySpan] = Field(default_factory=list)
    predicted_severity: str
    confidence: float
    probabilities: dict[str, float]
    hazard_indicators: list[str] = Field(default_factory=list)
    execution_time_ms: float


class AskRequest(BaseModel):
    """Payload for querying the LangGraph RAG Agent."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question regarding vehicle defects, recalls, or fleet safety stats",
        examples=["What are the common steering issues reported for Ford Explorer?"],
    )


class AskResponse(BaseModel):
    """Response from the multi-step LangGraph RAG agent."""

    question: str
    answer: str
    intent: str
    citations: list[str] = Field(default_factory=list)
    step_trace: list[str] = Field(default_factory=list)
    refusal: bool = False
    execution_time_ms: float


class MetricsSummaryResponse(BaseModel):
    """Aggregated verified evaluation metrics from reports directory."""

    data_quality: dict[str, Any] = Field(default_factory=dict)
    ner_metrics: dict[str, Any] = Field(default_factory=dict)
    clf_metrics: dict[str, Any] = Field(default_factory=dict)
    rag_metrics: dict[str, Any] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    """Metadata regarding loaded models, providers, and environment."""

    app_name: str
    version: str
    environment: str
    ner_model_path: str
    clf_model_path: str
    llm_provider: str
    chroma_dir: str

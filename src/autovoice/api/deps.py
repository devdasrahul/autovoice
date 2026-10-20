"""FastAPI dependency injection providers for ML models, agents, and configuration."""

from __future__ import annotations

from functools import lru_cache

from autovoice.classifier.infer import SeverityClassifierInferencer
from autovoice.config import Settings, get_settings
from autovoice.logging_conf import get_logger
from autovoice.ner.infer import NERInferencer
from autovoice.rag.graph import AutoVoiceRAGAgent
from autovoice.rag.index import ComplaintVectorIndex

log = get_logger(__name__)

# Singletons initialized lazily
_NER_INFERENCER: NERInferencer | None = None
_SEVERITY_CLASSIFIER: SeverityClassifierInferencer | None = None
_RAG_AGENT: AutoVoiceRAGAgent | None = None


def get_app_settings() -> Settings:
    """Return the application settings singleton."""
    return get_settings()


def get_ner_inferencer() -> NERInferencer:
    """Return cached NER inferencer singleton."""
    global _NER_INFERENCER
    if _NER_INFERENCER is None:
        settings = get_settings()
        log.info("deps.init_ner", path=settings.ner_model_path)
        _NER_INFERENCER = NERInferencer(source=settings.ner_model_path)
    return _NER_INFERENCER


def get_severity_classifier() -> SeverityClassifierInferencer:
    """Return cached Severity Classifier inferencer singleton."""
    global _SEVERITY_CLASSIFIER
    if _SEVERITY_CLASSIFIER is None:
        settings = get_settings()
        log.info("deps.init_severity_classifier", path=settings.clf_model_path)
        _SEVERITY_CLASSIFIER = SeverityClassifierInferencer(model_dir=settings.clf_model_path)
    return _SEVERITY_CLASSIFIER


def get_rag_agent() -> AutoVoiceRAGAgent:
    """Return cached AutoVoice RAG agent singleton."""
    global _RAG_AGENT
    if _RAG_AGENT is None:
        settings = get_settings()
        log.info("deps.init_rag_agent", chroma_dir=settings.chroma_dir)
        index = ComplaintVectorIndex(persist_dir=settings.chroma_dir)
        _RAG_AGENT = AutoVoiceRAGAgent(index=index)
    return _RAG_AGENT


@lru_cache(maxsize=1)
def warmup_all_models() -> dict[str, str]:
    """Pre-warm models during startup to minimize Cloud Run cold-start latency."""
    results = {}
    try:
        ner = get_ner_inferencer()
        ner.warmup()
        results["ner"] = "warmed"
    except Exception as exc:
        results["ner"] = f"error: {exc}"

    try:
        clf = get_severity_classifier()
        clf.warmup()
        results["clf"] = "warmed"
    except Exception as exc:
        results["clf"] = f"error: {exc}"

    try:
        get_rag_agent()
        results["rag"] = "ready"
    except Exception as exc:
        results["rag"] = f"error: {exc}"

    return results

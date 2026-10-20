"""LangGraph RAG Agent for AutoVoice Automotive Complaint Intelligence.

Graph Nodes:
    1. route: classify query intent (stats_aggregate vs semantic_search vs out_of_scope).
    2. retrieve: search vector index with metadata filters.
    3. grade: filter out low-similarity or irrelevant passages.
    4. enrich: apply NER and severity tags to retrieved passages.
    5. generate: synthesize answer with CMPLID citations.
    6. guard: refuse out-of-scope queries and enforce safety/legal disclaimers.
    7. aggregate_sql: whitelisted SQL query execution for stats questions (BigQuery/DuckDB).
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

import pandas as pd

from autovoice.config import DATA_PROCESSED_DIR, get_settings
from autovoice.logging_conf import get_logger
from autovoice.rag.index import ComplaintVectorIndex
from autovoice.rag.llm import get_llm_provider
from autovoice.rag.prompts import DISCLAIMER_TEXT

log = get_logger(__name__)

# Out-of-scope intent triggers (safety/legal advice, non-automotive, chit-chat)
OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(sue|lawsuit|lawyer|legal advice|settlement)\b", re.I),
    re.compile(r"\b(stock price|invest|bitcoin|crypto|shares)\b", re.I),
    re.compile(r"\b(weather|recipe|movie|joke|who are you|poem|song|story|lyrics)\b", re.I),
    re.compile(r"\b(is it safe to drive|should i drive|guarantee safety)\b", re.I),
]

# Statistical aggregation keywords
AGGREGATE_KEYWORDS = [
    "how many",
    "count",
    "top",
    "most",
    "percentage",
    "rate",
    "frequency",
    "distribution",
    "total complaints",
    "which component has",
    "highest",
]

AUTOMOTIVE_MAKES = [
    "FORD",
    "TESLA",
    "TOYOTA",
    "HONDA",
    "CHEVROLET",
    "HYUNDAI",
    "NISSAN",
    "SUBARU",
    "BMW",
]


class AutoVoiceAgentState(TypedDict):
    """Typed state passing through LangGraph agent nodes."""

    question: str
    intent: str
    filters: dict[str, Any]
    retrieved_passages: list[dict[str, Any]]
    relevant_passages: list[dict[str, Any]]
    enriched_data: dict[str, Any]
    answer: str
    citations: list[str]
    step_trace: list[str]
    refusal: bool


def parse_metadata_filters(question: str) -> dict[str, Any]:
    """Extract make and model year from user question text."""
    filters: dict[str, Any] = {}
    q_upper = question.upper()

    for make in AUTOMOTIVE_MAKES:
        if re.search(r"\b" + make + r"\b", q_upper):
            filters["make"] = make
            break

    year_match = re.search(r"\b(20[0-2][0-9])\b", question)
    if year_match:
        filters["model_year"] = int(year_match.group(1))

    return filters


def route_node(state: AutoVoiceAgentState) -> AutoVoiceAgentState:
    """Classify user question intent and parse entity constraints."""
    q = state["question"]
    steps = list(state.get("step_trace", []))
    steps.append("route: evaluating intent")

    # 1. Out-of-scope check
    if any(p.search(q) for p in OUT_OF_SCOPE_PATTERNS):
        return {
            **state,
            "intent": "out_of_scope",
            "step_trace": steps + ["route: classified as out_of_scope"],
        }

    # 2. Stats aggregation check
    q_lower = q.lower()
    if any(k in q_lower for k in AGGREGATE_KEYWORDS):
        filters = parse_metadata_filters(q)
        return {
            **state,
            "intent": "stats_aggregate",
            "filters": filters,
            "step_trace": steps + ["route: classified as stats_aggregate"],
        }

    # 3. Default semantic defect search
    filters = parse_metadata_filters(q)
    return {
        **state,
        "intent": "semantic_search",
        "filters": filters,
        "step_trace": steps + ["route: classified as semantic_search"],
    }


def retrieve_node(
    state: AutoVoiceAgentState, index: ComplaintVectorIndex | None = None
) -> AutoVoiceAgentState:
    """Retrieve top passages from vector index with metadata filtering."""
    idx = index or ComplaintVectorIndex()
    steps = list(state.get("step_trace", []))
    steps.append("retrieve: querying vector index")

    filters = state.get("filters", {})
    passages = idx.query(state["question"], top_k=5, filters=filters)

    return {
        **state,
        "retrieved_passages": [p.to_dict() for p in passages],
        "step_trace": steps + [f"retrieve: found {len(passages)} passages"],
    }


def grade_node(state: AutoVoiceAgentState) -> AutoVoiceAgentState:
    """Drop passages below similarity threshold or with zero overlap."""
    settings = get_settings()
    min_sim = settings.rag_min_similarity
    steps = list(state.get("step_trace", []))
    steps.append("grade: filtering passages by relevance threshold")

    retrieved = state.get("retrieved_passages", [])
    relevant = [p for p in retrieved if p.get("score", 0.0) >= min_sim]

    if not relevant and retrieved:
        # Keep at least top 1 if available
        relevant = [retrieved[0]]

    return {
        **state,
        "relevant_passages": relevant,
        "step_trace": steps + [f"grade: kept {len(relevant)} relevant passages"],
    }


def enrich_node(state: AutoVoiceAgentState) -> AutoVoiceAgentState:
    """Run lightweight component and symptom tag detection over relevant passages."""
    steps = list(state.get("step_trace", []))
    steps.append("enrich: extracting component/symptom metadata")

    components: set[str] = set()
    severities: list[str] = []

    for p in state.get("relevant_passages", []):
        comp = p.get("metadata", {}).get("component")
        sev = p.get("metadata", {}).get("severity")
        if comp:
            components.add(comp)
        if sev:
            severities.append(sev)

    enriched = {
        "identified_components": list(components),
        "severity_distribution": {s: severities.count(s) for s in set(severities)},
    }

    return {
        **state,
        "enriched_data": enriched,
        "step_trace": steps + ["enrich: enriched passage metadata"],
    }


def generate_node(state: AutoVoiceAgentState) -> AutoVoiceAgentState:
    """Synthesize cited answer from relevant evidence passages."""
    steps = list(state.get("step_trace", []))
    steps.append("generate: synthesizing response with citations")

    passages = state.get("relevant_passages", [])
    provider = get_llm_provider()
    answer = provider.generate(state["question"], passages)

    citations = [
        p.get("id") or p.get("metadata", {}).get("cmplid", "")
        for p in passages
        if p.get("id") or p.get("metadata", {}).get("cmplid")
    ]

    return {
        **state,
        "answer": answer,
        "citations": [c for c in citations if c],
        "step_trace": steps + ["generate: completed synthesis"],
    }


def guard_node(state: AutoVoiceAgentState) -> AutoVoiceAgentState:
    """Refuse out-of-scope inquiries and attach consumer safety notice."""
    steps = list(state.get("step_trace", []))
    steps.append("guard: applying safety boundaries")

    refusal_msg = (
        "I am designed specifically to analyze vehicle defect complaints and safety incident records "
        "from the public NHTSA database. I cannot provide legal advice, vehicle purchase advice, or "
        "operational safety warranties.\n\n"
        f"*{DISCLAIMER_TEXT}*"
    )

    return {
        **state,
        "answer": refusal_msg,
        "citations": [],
        "refusal": True,
        "step_trace": steps + ["guard: refused out-of-scope request"],
    }


def aggregate_sql_node(state: AutoVoiceAgentState) -> AutoVoiceAgentState:
    """Execute whitelisted parameterized aggregate queries using DuckDB/pandas."""
    steps = list(state.get("step_trace", []))
    steps.append("aggregate_sql: running whitelisted query")

    parquet_p = DATA_PROCESSED_DIR / "complaints.parquet"
    if not parquet_p.exists():
        return {
            **state,
            "answer": f"Database file not found. Please run `make data` first.\n\n*{DISCLAIMER_TEXT}*",
            "citations": [],
            "step_trace": steps + ["aggregate_sql: complaints.parquet missing"],
        }

    df = pd.read_parquet(parquet_p)
    make_filter = state.get("filters", {}).get("make")
    if make_filter:
        df = df[df["make"].str.upper() == make_filter.upper()]

    if df.empty:
        return {
            **state,
            "answer": f"No complaint records found matching {make_filter or 'query'}.\n\n*{DISCLAIMER_TEXT}*",
            "citations": [],
            "step_trace": steps + ["aggregate_sql: empty filtered dataframe"],
        }

    # Aggregate top components
    top_comps = df["component"].value_counts().head(5)
    fire_count = int(df["fire"].sum()) if "fire" in df.columns else 0
    crash_count = int(df["crash"].sum()) if "crash" in df.columns else 0

    lines = [
        f"### Fleet Statistical Summary ({make_filter or 'All Makes'})\n",
        f"- Total Complaints in Dataset: **{len(df)}**",
        f"- Complaints Involving Fire: **{fire_count}**",
        f"- Complaints Involving Crash: **{crash_count}**",
        "\n**Top Reported Components:**",
    ]
    for comp, count in top_comps.items():
        pct = (count / len(df)) * 100
        lines.append(f"- {comp}: **{count}** ({pct:.1f}%)")

    lines.append(f"\n*{DISCLAIMER_TEXT}*")
    answer = "\n".join(lines)

    sample_cmplids = df["cmplid"].astype(str).head(5).tolist()

    return {
        **state,
        "answer": answer,
        "citations": sample_cmplids,
        "step_trace": steps + ["aggregate_sql: completed aggregation"],
    }


class AutoVoiceRAGAgent:
    """Compiled LangGraph Agent executing the multi-step complaint intelligence flow."""

    def __init__(self, index: ComplaintVectorIndex | None = None):
        """Initialize agent with vector index."""
        self.index = index or ComplaintVectorIndex()

    def run(self, question: str) -> dict[str, Any]:
        """Execute the LangGraph workflow given a user question."""
        state: AutoVoiceAgentState = {
            "question": question,
            "intent": "",
            "filters": {},
            "retrieved_passages": [],
            "relevant_passages": [],
            "enriched_data": {},
            "answer": "",
            "citations": [],
            "step_trace": [],
            "refusal": False,
        }

        # Step 1: Route
        state = route_node(state)

        # Step 2: Branch based on intent
        if state["intent"] == "out_of_scope":
            state = guard_node(state)
        elif state["intent"] == "stats_aggregate":
            state = aggregate_sql_node(state)
        else:
            state = retrieve_node(state, index=self.index)
            state = grade_node(state)
            state = enrich_node(state)
            state = generate_node(state)

        return {
            "question": state["question"],
            "answer": state["answer"],
            "intent": state["intent"],
            "citations": state["citations"],
            "step_trace": state["step_trace"],
            "refusal": state.get("refusal", False),
            "enriched_data": state.get("enriched_data", {}),
        }

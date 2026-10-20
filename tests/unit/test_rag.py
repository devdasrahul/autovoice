"""Unit tests for Phase-5 RAG and LangGraph agent components."""

from __future__ import annotations

from autovoice.rag.embeddings import EmbeddingEngine
from autovoice.rag.graph import (
    AutoVoiceRAGAgent,
    aggregate_sql_node,
    guard_node,
    parse_metadata_filters,
    route_node,
)
from autovoice.rag.llm import ExtractiveLLMProvider
from autovoice.rag.prompts import DISCLAIMER_TEXT


class TestRAGRouting:
    def test_route_stats_intent(self):
        state = {
            "question": "How many complaints are reported for Ford in 2021?",
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
        res = route_node(state)
        assert res["intent"] == "stats_aggregate"
        assert res["filters"].get("make") == "FORD"
        assert res["filters"].get("model_year") == 2021

    def test_route_out_of_scope(self):
        state = {
            "question": "Should I sue the dealership with a lawyer?",
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
        res = route_node(state)
        assert res["intent"] == "out_of_scope"

    def test_route_semantic_search(self):
        state = {
            "question": "Why did the electronic power steering rack lock up?",
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
        res = route_node(state)
        assert res["intent"] == "semantic_search"


class TestFiltersAndGuard:
    def test_parse_filters(self):
        filters = parse_metadata_filters("Looking for 2022 Honda steering problems")
        assert filters.get("make") == "HONDA"
        assert filters.get("model_year") == 2022

    def test_guard_node_refusal(self):
        state = {
            "question": "Can you guarantee the safety of driving this car?",
            "intent": "out_of_scope",
            "filters": {},
            "retrieved_passages": [],
            "relevant_passages": [],
            "enriched_data": {},
            "answer": "",
            "citations": [],
            "step_trace": [],
            "refusal": False,
        }
        res = guard_node(state)
        assert res["refusal"] is True
        assert DISCLAIMER_TEXT in res["answer"]


class TestExtractiveLLMAndEmbeddings:
    def test_extractive_provider_citations(self):
        provider = ExtractiveLLMProvider()
        passages = [
            {
                "id": "11504901",
                "text": "Steering wheel locked up suddenly at speed.",
                "metadata": {
                    "make": "FORD",
                    "model": "EXPLORER",
                    "model_year": 2021,
                    "severity": "HIGH",
                },
            }
        ]
        out = provider.generate("What happened with steering?", passages)
        assert "[CMPLID: 11504901]" in out
        assert DISCLAIMER_TEXT in out

    def test_embedding_engine_shape(self):
        embedder = EmbeddingEngine()
        vec = embedder.encode_query("power steering fault")
        assert len(vec) == 384
        assert any(x != 0.0 for x in vec)


class TestAggregateSQLAndAgent:
    def test_aggregate_sql_execution(self):
        state = {
            "question": "How many complaints for Ford?",
            "intent": "stats_aggregate",
            "filters": {"make": "FORD"},
            "retrieved_passages": [],
            "relevant_passages": [],
            "enriched_data": {},
            "answer": "",
            "citations": [],
            "step_trace": [],
            "refusal": False,
        }
        res = aggregate_sql_node(state)
        assert "Total Complaints" in res["answer"]
        assert len(res["citations"]) > 0

    def test_agent_run_end_to_end_none_provider(self):
        agent = AutoVoiceRAGAgent()
        res = agent.run("Why did the brake pedal go to the floor?")
        assert res["intent"] == "semantic_search"
        assert len(res["step_trace"]) > 0
        assert len(res["answer"]) > 0
        assert DISCLAIMER_TEXT in res["answer"]

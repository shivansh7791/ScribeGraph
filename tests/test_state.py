"""Unit tests for state schemas and Pydantic validation models."""

import pytest
from src.state import ContentState, ContentStateModel, NodeMetric, PipelineMetrics


def test_content_state_typeddict_instantiation():
    """Verify ContentState TypedDict can be constructed with expected fields."""
    state: ContentState = {
        "topic": "Distributed Consensus",
        "research_notes": ["Raft uses leader election", "Paxos uses proposal numbers"],
        "current_draft": "# Distributed Consensus Guide",
        "editor_feedback": "Add diagrams",
        "revision_count": 1,
        "fact_check_passed": True,
        "metrics": {},
    }

    assert state["topic"] == "Distributed Consensus"
    assert len(state["research_notes"]) == 2
    assert state["revision_count"] == 1
    assert state["fact_check_passed"] is True


def test_content_state_pydantic_model_validation():
    """Verify ContentStateModel validates types and default values accurately."""
    model = ContentStateModel(
        topic="Database Sharding",
        research_notes=["Consistent Hashing"],
    )

    assert model.topic == "Database Sharding"
    assert model.research_notes == ["Consistent Hashing"]
    assert model.current_draft == ""
    assert model.editor_feedback is None
    assert model.revision_count == 0
    assert model.fact_check_passed is False
    assert model.metrics == {}

    typed_dict = model.to_dict()
    assert isinstance(typed_dict, dict)
    assert typed_dict["topic"] == "Database Sharding"


def test_node_metric_and_pipeline_metrics():
    """Verify NodeMetric calculations and aggregate PipelineMetrics."""
    node_metric = NodeMetric(
        node_name="research_agent",
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        estimated_cost_usd=0.0005,
        latency_seconds=1.23,
        model_name="gemini-2.5-flash",
    )

    assert node_metric.node_name == "research_agent"
    assert node_metric.total_tokens == 300
    assert node_metric.estimated_cost_usd == 0.0005

    pipeline_metrics = PipelineMetrics(
        total_prompt_tokens=100,
        total_completion_tokens=200,
        total_tokens=300,
        total_cost_usd=0.0005,
        total_latency_seconds=1.23,
        node_metrics={"research_agent": node_metric},
    )

    assert "research_agent" in pipeline_metrics.node_metrics
    assert pipeline_metrics.total_tokens == 300

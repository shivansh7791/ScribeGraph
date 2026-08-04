"""Unit tests for observability setup, LangSmith tracing config, and TelemetryTracker summary aggregation."""

import os
import json
import pytest
from src.config import setup_observability, settings
from src.telemetry import TelemetryTracker
from src.state import ContentState, NodeMetric


def test_setup_observability_enabled(monkeypatch):
    """Verify setup_observability sets environment variables when enabled."""
    monkeypatch.setattr(settings, "langchain_tracing_v2", True)
    monkeypatch.setattr(settings, "langchain_api_key", "test_api_key_123")
    monkeypatch.setattr(settings, "langchain_project", "test-project")

    result = setup_observability()

    assert result is True
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_API_KEY") == "test_api_key_123"
    assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"


def test_get_pipeline_summary_math_calculations():
    """Verify get_pipeline_summary correctly sums tokens, cost, latency, and steps."""
    mock_metrics = {
        "researcher_node": NodeMetric(
            node_name="researcher_node",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            estimated_cost_usd=0.0005,
            latency_seconds=1.2,
            model_name="gemini-2.5-flash",
        ).model_dump(),
        "writer_node": NodeMetric(
            node_name="writer_node",
            prompt_tokens=300,
            completion_tokens=500,
            total_tokens=800,
            estimated_cost_usd=0.0015,
            latency_seconds=2.5,
            model_name="gemini-2.5-flash",
        ).model_dump(),
    }

    state: ContentState = {
        "topic": "Distributed Consensus",
        "editor_approved": True,
        "fact_check_passed": True,
        "revision_count": 1,
        "metrics": mock_metrics,
    }

    summary = TelemetryTracker.get_pipeline_summary(state)

    assert summary["topic"] == "Distributed Consensus"
    assert summary["total_prompt_tokens"] == 400
    assert summary["total_completion_tokens"] == 700
    assert summary["total_tokens"] == 1100
    assert summary["total_cost_usd"] == round(0.0005 + 0.0015, 6)
    assert summary["total_latency_seconds"] == round(1.2 + 2.5, 4)
    assert summary["step_count"] == 2
    assert summary["revision_count"] == 1
    assert summary["editor_approved"] is True
    assert summary["fact_check_passed"] is True


def test_telemetry_summary_json_serialization(tmp_path):
    """Verify telemetry summary can be serialized to JSON file without error."""
    mock_state: ContentState = {
        "topic": "System Design",
        "revision_count": 0,
        "editor_approved": True,
        "fact_check_passed": True,
        "metrics": {
            "node_1": {
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "total_tokens": 100,
                "estimated_cost_usd": 0.0001,
                "latency_seconds": 0.5,
            }
        },
    }

    summary = TelemetryTracker.get_pipeline_summary(mock_state)
    export_path = tmp_path / "telemetry_report.json"

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    assert export_path.exists()

    with open(export_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert loaded_data["total_tokens"] == 100
    assert loaded_data["step_count"] == 1

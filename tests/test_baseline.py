"""Unit and integration tests for baseline single-prompt generator."""

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from evals.baseline import run_baseline


def test_run_baseline_with_mock_llm():
    """Verify run_baseline produces expected output and telemetry using mock LLM."""
    mock_llm = MagicMock()
    mock_response = AIMessage(
        content="# Distributed Caching Guide\n\nExecutive Summary...",
        usage_metadata={"input_tokens": 120, "output_tokens": 250, "total_tokens": 370},
    )
    mock_llm.invoke.return_value = mock_response

    topic = "Distributed Caching Strategies"
    result = run_baseline(
        topic=topic,
        provider="gemini",
        model_name="gemini-2.5-flash",
        mock_llm=mock_llm,
    )

    assert result["topic"] == topic
    assert result["provider"] == "gemini"
    assert result["model_name"] == "gemini-2.5-flash"
    assert "Distributed Caching Guide" in result["generated_draft"]

    telemetry = result["telemetry"]
    assert telemetry["node_name"] == "baseline_single_prompt"
    assert telemetry["prompt_tokens"] == 120
    assert telemetry["completion_tokens"] == 250
    assert telemetry["total_tokens"] == 370
    assert telemetry["estimated_cost_usd"] > 0
    assert telemetry["latency_seconds"] >= 0.0

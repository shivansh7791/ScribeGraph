"""Unit tests for structured logging and telemetry tracker utility."""

import time
from unittest.mock import MagicMock
from src.telemetry import TelemetryTracker
from src.config import ModelPricing


def test_cost_calculation():
    """Test cost calculation logic across standard models."""
    # Test gemini-2.5-flash: (0.000075 in, 0.0003 out per 1k tokens)
    cost = TelemetryTracker.calculate_cost("gemini-2.5-flash", 1000, 1000)
    assert cost == round(0.000075 + 0.0003, 6)

    # Test gpt-4o-mini: (0.00015 in, 0.0006 out per 1k tokens)
    cost_mini = TelemetryTracker.calculate_cost("gpt-4o-mini", 2000, 1000)
    assert cost_mini == round((2 * 0.00015) + (1 * 0.0006), 6)


def test_extract_token_usage_langchain_metadata():
    """Test token usage extraction from LangChain usage_metadata."""
    mock_response = MagicMock()
    mock_response.usage_metadata = {"input_tokens": 150, "output_tokens": 350}

    prompt_tokens, comp_tokens = TelemetryTracker.extract_token_usage(mock_response)
    assert prompt_tokens == 150
    assert comp_tokens == 350


def test_record_step_telemetry():
    """Test recording a pipeline step metric."""
    start_time = time.time() - 0.5  # Simulate 0.5s latency

    mock_response = MagicMock()
    mock_response.usage_metadata = {"input_tokens": 200, "output_tokens": 400}

    metric = TelemetryTracker.record_step(
        node_name="editor_node",
        model_name="gemini-2.5-flash",
        start_time=start_time,
        response=mock_response,
    )

    assert metric.node_name == "editor_node"
    assert metric.prompt_tokens == 200
    assert metric.completion_tokens == 400
    assert metric.total_tokens == 600
    assert metric.latency_seconds >= 0.49
    assert metric.estimated_cost_usd > 0

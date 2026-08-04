"""Structured logging and telemetry tracking utility for agent execution."""

import time
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import BaseMessage
from src.config import ModelPricing, settings
from src.state import NodeMetric

# Configure structured logging format
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("pipeline.telemetry")


class TelemetryTracker:
    """Utility class to measure, record, and log per-node execution telemetry."""

    @staticmethod
    def extract_token_usage(response: Any) -> tuple[int, int]:
        """Extract (prompt_tokens, completion_tokens) from LangChain LLM responses."""
        prompt_tokens = 0
        completion_tokens = 0

        # Case 1: LangChain response object with usage_metadata attribute
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        # Case 2: LangChain response metadata dictionary
        elif hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
            metadata = response.response_metadata
            # OpenAI / Generic format
            token_usage = metadata.get("token_usage") or metadata.get("usage", {})
            if isinstance(token_usage, dict):
                prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens", 0)
            # Gemini raw format
            elif "usage_metadata" in metadata:
                gemini_usage = metadata["usage_metadata"]
                prompt_tokens = gemini_usage.get("prompt_token_count", 0)
                completion_tokens = gemini_usage.get("candidates_token_count", 0)

        # Fallback estimation if token usage is missing (approx 4 chars per token)
        if prompt_tokens == 0 and completion_tokens == 0:
            if isinstance(response, BaseMessage) and isinstance(response.content, str):
                completion_tokens = max(1, len(response.content) // 4)

        return prompt_tokens, completion_tokens

    @classmethod
    def calculate_cost(cls, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost in USD based on input/output tokens."""
        input_price_1k, output_price_1k = ModelPricing.get_pricing(model_name)
        input_cost = (prompt_tokens / 1000.0) * input_price_1k
        output_cost = (completion_tokens / 1000.0) * output_price_1k
        return round(input_cost + output_cost, 6)

    @classmethod
    def record_step(
        cls,
        node_name: str,
        model_name: str,
        start_time: float,
        response: Optional[Any] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> NodeMetric:
        """Record and log execution metrics for a pipeline step."""
        end_time = time.time()
        latency = round(end_time - start_time, 4)

        if response is not None and (prompt_tokens is None or completion_tokens is None):
            extracted_prompt, extracted_comp = cls.extract_token_usage(response)
            prompt_tokens = prompt_tokens if prompt_tokens is not None else extracted_prompt
            completion_tokens = completion_tokens if completion_tokens is not None else extracted_comp
        else:
            prompt_tokens = prompt_tokens or 0
            completion_tokens = completion_tokens or 0

        total_tokens = prompt_tokens + completion_tokens
        cost = cls.calculate_cost(model_name, prompt_tokens, completion_tokens)

        metric = NodeMetric(
            node_name=node_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            latency_seconds=latency,
            model_name=model_name,
        )

        logger.info(
            f"Step: {node_name:15s} | Model: {model_name:20s} | "
            f"Latency: {latency:6.3f}s | Tokens: {total_tokens:5d} ({prompt_tokens} in / {completion_tokens} out) | "
            f"Cost: ${cost:.6f}"
        )

        return metric

    @classmethod
    def get_pipeline_summary(cls, state: Any) -> Dict[str, Any]:
        """Compute aggregate pipeline execution metrics and cost summary from state.

        Args:
            state: ContentState dictionary or ContentStateModel.

        Returns:
            Dict containing total token usage, total latency, total USD cost, step count,
            and per-node breakdown.
        """
        metrics = state.get("metrics", {}) if isinstance(state, dict) else getattr(state, "metrics", {})

        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        node_breakdown = {}

        for node_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                p_tokens = metric_data.get("prompt_tokens", 0)
                c_tokens = metric_data.get("completion_tokens", 0)
                t_tokens = metric_data.get("total_tokens", 0)
                cost = metric_data.get("estimated_cost_usd", 0.0)
                latency = metric_data.get("latency_seconds", 0.0)
            else:
                p_tokens = getattr(metric_data, "prompt_tokens", 0)
                c_tokens = getattr(metric_data, "completion_tokens", 0)
                t_tokens = getattr(metric_data, "total_tokens", 0)
                cost = getattr(metric_data, "estimated_cost_usd", 0.0)
                latency = getattr(metric_data, "latency_seconds", 0.0)

            total_prompt += p_tokens
            total_completion += c_tokens
            total_tokens += t_tokens
            total_cost += cost
            total_latency += latency
            node_breakdown[node_name] = metric_data

        topic = state.get("topic", "") if isinstance(state, dict) else getattr(state, "topic", "")
        revision_count = state.get("revision_count", 0) if isinstance(state, dict) else getattr(state, "revision_count", 0)
        editor_approved = state.get("editor_approved") if isinstance(state, dict) else getattr(state, "editor_approved", None)
        fact_check_passed = state.get("fact_check_passed", False) if isinstance(state, dict) else getattr(state, "fact_check_passed", False)

        return {
            "topic": topic,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_seconds": round(total_latency, 4),
            "step_count": len(metrics),
            "revision_count": revision_count,
            "editor_approved": editor_approved,
            "fact_check_passed": fact_check_passed,
            "node_breakdown": node_breakdown,
        }

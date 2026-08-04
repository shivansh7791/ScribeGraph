"""Single-prompt baseline LLM content generator for benchmark comparisons."""

import argparse
import json
import time
import sys
from pathlib import Path

# Add project root directory to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import settings
from src.telemetry import TelemetryTracker, logger


SINGLE_PROMPT_SYSTEM_TEMPLATE = """You are a Principal Tech Interview Coach and Senior System Architect.
Your task is to write a comprehensive, technical, and highly structured placement interview preparation guide on the given topic.

The guide must include:
1. Executive Summary & Core Concepts
2. System Architecture / Technical Deep-Dive
3. Trade-offs, Edge Cases, and Common Bottlenecks
4. Real-world Interview Questions & Recommended Framework Answers

Write in clean Markdown format with professional tone."""


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None):
    """Instantiate the appropriate LLM based on provider configuration."""
    provider = provider or settings.default_model_provider
    model_name = model_name or settings.default_model_name

    if provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            google_api_key=settings.gemini_api_key or "DUMMY_KEY",
        )
    elif provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key or "DUMMY_KEY",
        )
    elif provider.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.groq_api_key or "DUMMY_KEY",
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def run_baseline(
    topic: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    mock_llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute standard single-prompt LLM content generation benchmark.

    Args:
        topic: Technical subject for content generation.
        provider: LLM provider ("gemini", "openai", "groq").
        model_name: Model identifier string.
        mock_llm: Optional pre-configured LLM instance (useful for unit tests).

    Returns:
        Dict containing topic, generated content, and execution telemetry metrics.
    """
    provider = provider or settings.default_model_provider
    model_name = model_name or settings.default_model_name

    logger.info(f"Starting Baseline Single-Prompt Run for topic: '{topic}' using {provider}/{model_name}")
    start_time = time.time()

    if mock_llm is not None:
        llm = mock_llm
    else:
        llm = get_llm(provider=provider, model_name=model_name)

    messages = [
        SystemMessage(content=SINGLE_PROMPT_SYSTEM_TEMPLATE),
        HumanMessage(content=f"Topic for Tech Placement Interview Guide: {topic}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Baseline LLM invocation failed: {e}")
        # Return structured failure baseline report
        metric = TelemetryTracker.record_step(
            node_name="baseline_single_prompt",
            model_name=model_name,
            start_time=start_time,
            prompt_tokens=0,
            completion_tokens=0,
        )
        return {
            "topic": topic,
            "error": str(e),
            "generated_draft": "",
            "telemetry": metric.model_dump(),
        }

    metric = TelemetryTracker.record_step(
        node_name="baseline_single_prompt",
        model_name=model_name,
        start_time=start_time,
        response=response,
    )

    result = {
        "topic": topic,
        "provider": provider,
        "model_name": model_name,
        "generated_draft": content,
        "telemetry": metric.model_dump(),
    }

    logger.info("Baseline Single-Prompt Run completed successfully.")
    return result


def main():
    """CLI entry point for running the single-prompt baseline generator."""
    parser = argparse.ArgumentParser(description="Run Single-Prompt Baseline LLM Content Generator")
    parser.add_argument(
        "--topic",
        type=str,
        default="Distributed Caching Strategies in System Design",
        help="Technical topic for placement interview guide",
    )
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider (gemini, openai, groq)")
    parser.add_argument("--model", type=str, default=None, help="Model name identifier")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file path to save baseline result")

    args = parser.parse_args()

    result = run_baseline(topic=args.topic, provider=args.provider, model_name=args.model)

    print("\n" + "=" * 60)
    print(f"BASELINE GENERATION RESULT FOR TOPIC: {result['topic']}")
    print("=" * 60)
    print("\n--- METRICS TELEMETRY ---")
    print(json.dumps(result["telemetry"], indent=2))
    print("\n--- GENERATED DRAFT PREVIEW (First 500 chars) ---")
    print(result.get("generated_draft", "")[:500] + "...\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

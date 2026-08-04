"""Quantitative Benchmark Runner comparing Single-Prompt Baseline vs Multi-Agent Pipeline."""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage
from evals.baseline import run_baseline
from evals.evaluators import evaluate_content, check_word_count, check_formatting_heuristics, EvalRubric
from src.pipeline import run_pipeline
from src.telemetry import TelemetryTracker, logger


def create_mock_llm_for_benchmark():
    """Create a mock LLM instance for deterministic benchmark testing."""
    mock_llm = MagicMock()
    # Baseline return
    mock_llm.invoke.return_value = AIMessage(
        content="# Distributed System Guide\n\nExecutive Summary...",
        usage_metadata={"input_tokens": 150, "output_tokens": 450, "total_tokens": 600},
    )
    # Structured output mock return
    mock_eval = EvalRubric(
        factual_grounding=4,
        hook_quality=4,
        clarity_structure=5,
        actionability=4,
        overall_score=4.3,
        reasoning="Solid structure and facts verified.",
    )
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_eval
    return mock_llm


def run_benchmark(
    dataset_path: str = "evals/dataset.json",
    mock_mode: bool = False,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute evaluation benchmark comparing Single-Prompt Baseline against Multi-Agent Pipeline.

    Args:
        dataset_path: Path to dataset JSON file.
        mock_mode: If True, uses mock LLM for rapid test execution.
        output_file: Optional path to export benchmark_report.json.

    Returns:
        Dict containing aggregated comparative benchmark report metrics.
    """
    logger.info(f"=== Starting Benchmark Evaluation (Dataset: {dataset_path}, Mock Mode: {mock_mode}) ===")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    baseline_results = []
    pipeline_results = []

    mock_llm = create_mock_llm_for_benchmark() if mock_mode else None

    for entry in dataset:
        topic = entry["topic"]
        facts = entry.get("ground_truth_facts", [])
        logger.info(f"Evaluating Topic [{entry['id']}]: '{topic}'")

        # 1. Run Single-Prompt Baseline
        b_res = run_baseline(topic=topic, mock_llm=mock_llm)
        b_draft = b_res.get("generated_draft", "")
        b_telemetry = b_res.get("telemetry", {})

        b_eval = evaluate_content(topic, b_draft, facts, llm=mock_llm)
        b_word_ok = check_word_count(b_draft)
        b_fmt = check_formatting_heuristics(b_draft)

        baseline_results.append({
            "topic": topic,
            "draft": b_draft,
            "telemetry": b_telemetry,
            "eval": b_eval.model_dump() if hasattr(b_eval, "model_dump") else dict(b_eval),
            "word_count_valid": b_word_ok,
            "formatting": b_fmt,
        })

        # 2. Run Multi-Agent StateGraph Pipeline
        if mock_mode:
            # Synthetic state for mock mode pipeline test
            p_state = {
                "topic": topic,
                "current_draft": f"# Multi-Agent Guide: {topic}\n\nComprehensive architecture...",
                "editor_approved": True,
                "fact_check_passed": True,
                "revision_count": 1,
                "metrics": {
                    "researcher_node": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300, "estimated_cost_usd": 0.0005, "latency_seconds": 1.0},
                    "writer_node": {"prompt_tokens": 200, "completion_tokens": 400, "total_tokens": 600, "estimated_cost_usd": 0.0010, "latency_seconds": 2.0},
                    "editor_node": {"prompt_tokens": 150, "completion_tokens": 150, "total_tokens": 300, "estimated_cost_usd": 0.0005, "latency_seconds": 1.2},
                    "fact_checker_node": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200, "estimated_cost_usd": 0.0003, "latency_seconds": 0.8},
                }
            }
        else:
            p_state = run_pipeline(topic=topic)

        p_draft = p_state.get("current_draft", "")
        p_summary = TelemetryTracker.get_pipeline_summary(p_state)
        p_eval = evaluate_content(topic, p_draft, facts, llm=mock_llm)
        p_word_ok = check_word_count(p_draft)
        p_fmt = check_formatting_heuristics(p_draft)

        pipeline_results.append({
            "topic": topic,
            "draft": p_draft,
            "summary": p_summary,
            "eval": p_eval.model_dump() if hasattr(p_eval, "model_dump") else dict(p_eval),
            "word_count_valid": p_word_ok,
            "formatting": p_fmt,
        })

    # Compute Aggregates
    count = len(dataset)

    avg_b_quality = sum(r["eval"]["overall_score"] for r in baseline_results) / count
    avg_p_quality = sum(r["eval"]["overall_score"] for r in pipeline_results) / count

    avg_b_grounding = sum(r["eval"]["factual_grounding"] for r in baseline_results) / count
    avg_p_grounding = sum(r["eval"]["factual_grounding"] for r in pipeline_results) / count

    avg_b_latency = sum(r["telemetry"].get("latency_seconds", 0) for r in baseline_results) / count
    avg_p_latency = sum(r["summary"]["total_latency_seconds"] for r in pipeline_results) / count

    avg_b_cost = sum(r["telemetry"].get("estimated_cost_usd", 0) for r in baseline_results) / count
    avg_p_cost = sum(r["summary"]["total_cost_usd"] for r in pipeline_results) / count

    total_b_tokens = sum(r["telemetry"].get("total_tokens", 0) for r in baseline_results)
    total_p_tokens = sum(r["summary"]["total_tokens"] for r in pipeline_results)

    approval_rate = (sum(1 for r in pipeline_results if r["summary"].get("editor_approved")) / count) * 100.0
    avg_revisions = sum(r["summary"].get("revision_count", 0) for r in pipeline_results) / count

    report = {
        "dataset_size": count,
        "baseline_summary": {
            "avg_quality_score": round(avg_b_quality, 2),
            "avg_factual_grounding": round(avg_b_grounding, 2),
            "avg_latency_seconds": round(avg_b_latency, 3),
            "avg_cost_usd": round(avg_b_cost, 6),
            "total_tokens": total_b_tokens,
        },
        "pipeline_summary": {
            "avg_quality_score": round(avg_p_quality, 2),
            "avg_factual_grounding": round(avg_p_grounding, 2),
            "avg_latency_seconds": round(avg_p_latency, 3),
            "avg_cost_usd": round(avg_p_cost, 6),
            "total_tokens": total_p_tokens,
            "editor_approval_rate_pct": round(approval_rate, 1),
            "avg_revisions": round(avg_revisions, 2),
        },
        "detailed_baseline_runs": baseline_results,
        "detailed_pipeline_runs": pipeline_results,
    }

    # Print Comparative Matrix Table
    print("\n" + "=" * 80)
    print("📊 QUANTITATIVE BENCHMARK COMPARISON MATRIX")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Single-Prompt Baseline':<22} | {'Multi-Agent Pipeline':<22} | {'Delta / Trade-off'}")
    print("-" * 88)
    print(f"{'Avg Quality Score (1.0-5.0)':<30} | {avg_b_quality:<22.2f} | {avg_p_quality:<22.2f} | {avg_p_quality - avg_b_quality:+.2f} pts")
    print(f"{'Avg Factual Grounding (1-5)':<30} | {avg_b_grounding:<22.2f} | {avg_p_grounding:<22.2f} | {avg_p_grounding - avg_b_grounding:+.2f} pts")
    print(f"{'Avg Latency (seconds)':<30} | {avg_b_latency:<22.3f} | {avg_p_latency:<22.3f} | {avg_p_latency - avg_b_latency:+.3f}s")
    print(f"{'Avg Token Cost (USD)':<30} | ${avg_b_cost:<21.6f} | ${avg_p_cost:<21.6f} | ${avg_p_cost - avg_b_cost:+.6f}")
    print(f"{'Total Tokens Used':<30} | {total_b_tokens:<22d} | {total_p_tokens:<22d} | {total_p_tokens - total_b_tokens:+d} tokens")
    print(f"{'Editor Approval Rate (%)':<30} | {'N/A (No Editor)':<22} | {approval_rate:<21.1f}% | N/A")
    print(f"{'Avg Revisions':<30} | {'0 (Single Shot)':<22} | {avg_revisions:<22.2f} | N/A")
    print("=" * 80)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n📈 Full benchmark report exported to: {output_file}")

    return report


def main():
    """CLI entry point for running the evaluation benchmark suite."""
    parser = argparse.ArgumentParser(description="Run Multi-Agent Content Pipeline Benchmark Matrix")
    parser.add_argument("--dataset", type=str, default="evals/dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="evals/benchmark_report.json", help="Path to output JSON report")
    parser.add_argument("--mock", action="store_true", help="Run benchmark in fast mock LLM mode for testing")

    args = parser.parse_args()
    run_benchmark(dataset_path=args.dataset, mock_mode=args.mock, output_file=args.output)


if __name__ == "__main__":
    main()

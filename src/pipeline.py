"""Pipeline Runner CLI for executing end-to-end Multi-Agent Content StateGraph."""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
from src.config import setup_observability
from src.graph import build_graph
from src.state import ContentState
from src.telemetry import TelemetryTracker, logger


def run_pipeline(
    topic: str,
    graph_app: Any = None,
    telemetry_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end multi-agent content pipeline stream for a given topic.

    Args:
        topic: Technical interview guide topic string.
        graph_app: Optional pre-compiled StateGraph application instance.
        telemetry_file: Optional path to export JSON telemetry report.

    Returns:
        Final merged ContentState dictionary upon graph completion.
    """
    tracing_enabled = setup_observability()
    if tracing_enabled:
        logger.info("[Observability] LangSmith tracing active.")

    app = graph_app or build_graph()
    logger.info(f"=== Starting ScribeGraph Execution for Topic: '{topic}' ===")

    initial_state: ContentState = {
        "topic": topic,
        "research_notes": [],
        "current_draft": "",
        "editor_feedback": None,
        "editor_approved": None,
        "revision_count": 0,
        "fact_check_passed": False,
        "metrics": {},
    }

    final_state = dict(initial_state)

    print("\n" + "=" * 70)
    print(f"🚀 EXECUTING MULTI-AGENT GRAPH PIPELINE FOR TOPIC: '{topic}'")
    print("=" * 70)

    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n➔ [GRAPH STEP EXECUTED]: Agent Node -> '{node_name}'")
            if isinstance(state_update, dict):
                final_state.update(state_update)
                for key, val in state_update.items():
                    if key == "metrics":
                        continue
                    summary = str(val)[:120] + "..." if isinstance(val, str) and len(val) > 120 else str(val)
                    print(f"   ↳ Updated State Key [{key}]: {summary}")

    print("\n" + "=" * 70)
    print("🏁 PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

    # Compute aggregate telemetry report using TelemetryTracker
    summary_report = TelemetryTracker.get_pipeline_summary(final_state)

    print("\n--- AGGREGATE PIPELINE TELEMETRY SUMMARY ---")
    print(f"Total Nodes Executed : {summary_report['step_count']}")
    print(f"Total Latency        : {summary_report['total_latency_seconds']:.3f} seconds")
    print(f"Total Tokens Used    : {summary_report['total_tokens']} tokens ({summary_report['total_prompt_tokens']} in / {summary_report['total_completion_tokens']} out)")
    print(f"Total Estimated Cost : ${summary_report['total_cost_usd']:.6f} USD")
    print("\nPer-Node Metrics Breakdown:")
    for node, m in summary_report["node_breakdown"].items():
        if isinstance(m, dict):
            lat = m.get("latency_seconds", 0)
            tok = m.get("total_tokens", 0)
            cst = m.get("estimated_cost_usd", 0)
        else:
            lat = getattr(m, "latency_seconds", 0)
            tok = getattr(m, "total_tokens", 0)
            cst = getattr(m, "estimated_cost_usd", 0)
        print(f"  - {node:18s} | Latency: {lat:6.3f}s | Tokens: {tok:5d} | Cost: ${cst:.6f}")

    print("\n--- FINAL DRAFT STATUS ---")
    print(f"Editor Approved   : {summary_report['editor_approved']}")
    print(f"Fact Check Passed : {summary_report['fact_check_passed']}")
    print(f"Total Revisions   : {summary_report['revision_count']}")

    if telemetry_file:
        with open(telemetry_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
        print(f"\n📊 Telemetry summary report exported to: {telemetry_file}")

    return final_state


def main():
    """CLI entry point for pipeline runner."""
    parser = argparse.ArgumentParser(description="Run ScribeGraph — Enterprise Multi-Agent Content Pipeline")
    parser.add_argument(
        "--topic",
        type=str,
        default="Distributed Consensus Protocols (Raft vs Paxos)",
        help="Technical topic for placement interview guide",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file path to save output state")
    parser.add_argument(
        "--telemetry-file",
        type=str,
        default=None,
        help="Optional JSON file path to export telemetry summary report",
    )

    args = parser.parse_args()
    final_state = run_pipeline(topic=args.topic, telemetry_file=args.telemetry_file)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2)
        print(f"\nFinal State saved to: {args.output}")


if __name__ == "__main__":
    main()

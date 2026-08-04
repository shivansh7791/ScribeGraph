"""Streamlit Interactive Web Application for Multi-Agent Content Pipeline."""

import json
import time
import sys
from pathlib import Path
import streamlit as st

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import build_graph
from src.state import ContentState
from src.telemetry import TelemetryTracker
from src.config import settings, setup_observability
from evals.evaluators import evaluate_content, check_word_count, check_formatting_heuristics


def load_preset_topics():
    """Load topics from evals/dataset.json."""
    dataset_path = Path(__file__).resolve().parent / "evals" / "dataset.json"
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["topic"]: item for item in data}
    return {}


def main():
    st.set_page_config(
        page_title="ScribeGraph — Evaluated Multi-Agent Content Pipeline",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("✍️ ScribeGraph: Evaluated Multi-Agent Content Pipeline")
    st.caption("Enterprise LangGraph Orchestration Engine with Telemetry & LLM-as-a-Judge Benchmarking")

    # Initialize Observability
    setup_observability()

    preset_topics_map = load_preset_topics()

    # Sidebar Controls
    st.sidebar.header("⚙️ Pipeline Configuration")
    topic_mode = st.sidebar.radio("Select Topic Input Mode:", ["Preset Dataset Topics", "Custom Topic Input"])

    if topic_mode == "Preset Dataset Topics" and preset_topics_map:
        selected_topic_name = st.sidebar.selectbox("Choose Technical Topic:", list(preset_topics_map.keys()))
        selected_topic = selected_topic_name
        facts = preset_topics_map[selected_topic_name].get("ground_truth_facts", [])
    else:
        selected_topic = st.sidebar.text_input(
            "Enter Custom Topic:",
            value="Distributed Consensus Protocols (Raft vs Paxos)",
        )
        facts = []

    model_provider = st.sidebar.selectbox("LLM Provider:", ["groq", "gemini", "openai"], index=0)
    model_name = st.sidebar.text_input("Model Name:", value=settings.default_model_name)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Telemetry Status**")
    st.sidebar.info(f"Tracing: {'Active' if settings.langchain_tracing_v2 else 'Disabled'}")

    tab1, tab2 = st.tabs(["🚀 Multi-Agent Live Execution", "📊 Benchmark & Trade-off Matrix"])

    with tab1:
        st.subheader("Live StateGraph Execution Flow")
        st.write(f"Target Topic: **{selected_topic}**")

        if st.button("▶ Run Multi-Agent Pipeline", type="primary"):
            initial_state: ContentState = {
                "topic": selected_topic,
                "research_notes": [],
                "current_draft": "",
                "editor_feedback": None,
                "editor_approved": None,
                "revision_count": 0,
                "fact_check_passed": False,
                "metrics": {},
            }

            final_state = dict(initial_state)
            status_container = st.container()

            graph_app = build_graph()

            with st.spinner("Orchestrating Multi-Agent StateGraph..."):
                for event in graph_app.stream(initial_state):
                    for node_name, state_update in event.items():
                        final_state.update(state_update)

                        with status_container:
                            if node_name == "researcher":
                                with st.expander("🔍 1. Researcher Node Output", expanded=True):
                                    notes = state_update.get("research_notes", [])
                                    for note in notes:
                                        st.write(f"- {note}")

                            elif node_name == "writer":
                                rev_num = final_state.get("revision_count", 0)
                                title = f"📝 2. Writer Node Output (Draft Rev #{rev_num + 1})"
                                with st.expander(title, expanded=True):
                                    st.markdown(state_update.get("current_draft", ""))

                            elif node_name == "editor":
                                with st.expander("🧐 3. Editor Node Output", expanded=True):
                                    approved = state_update.get("editor_approved")
                                    critique = state_update.get("editor_feedback", "")
                                    st.write(f"**Approved Status**: `{approved}`")
                                    st.markdown(f"**Editorial Critique**:\n{critique}")

                            elif node_name == "fact_checker":
                                with st.expander("✅ 4. Fact Checker Node Output", expanded=True):
                                    passed = state_update.get("fact_check_passed")
                                    st.write(f"**Fact Check Status**: `{passed}`")

            st.success("Pipeline Execution Completed!")

            # Telemetry Metric Cards
            summary = TelemetryTracker.get_pipeline_summary(final_state)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Latency", f"{summary['total_latency_seconds']:.3f} s")
            col2.metric("Total Tokens", f"{summary['total_tokens']:,}")
            col3.metric("Estimated Cost", f"${summary['total_cost_usd']:.6f} USD")
            col4.metric("Total Revisions", f"{summary['revision_count']}")

            # Final Markdown Output
            st.markdown("---")
            st.subheader("📄 Final Published Article Guide")
            st.markdown(final_state.get("current_draft", "No draft generated."))

    with tab2:
        st.subheader("Single-Prompt Baseline vs. Multi-Agent Pipeline Benchmark")
        st.caption("Quantitative Comparison Matrix across Quality, Grounding, Latency, and Cost")

        benchmark_report_path = Path(__file__).resolve().parent / "evals" / "benchmark_report.json"
        if benchmark_report_path.exists():
            with open(benchmark_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            b_sum = report.get("baseline_summary", {})
            p_sum = report.get("pipeline_summary", {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ⚡ Single-Prompt Baseline")
                st.metric("Avg Quality Score", f"{b_sum.get('avg_quality_score', 0)} / 5.0")
                st.metric("Avg Factual Grounding", f"{b_sum.get('avg_factual_grounding', 0)} / 5")
                st.metric("Avg Latency", f"{b_sum.get('avg_latency_seconds', 0):.3f} s")
                st.metric("Avg Cost", f"${b_sum.get('avg_cost_usd', 0):.6f} USD")

            with col2:
                st.markdown("### 🤖 Multi-Agent StateGraph Pipeline")
                st.metric(
                    "Avg Quality Score",
                    f"{p_sum.get('avg_quality_score', 0)} / 5.0",
                    delta=f"{p_sum.get('avg_quality_score', 0) - b_sum.get('avg_quality_score', 0):+.2f}",
                )
                st.metric(
                    "Avg Factual Grounding",
                    f"{p_sum.get('avg_factual_grounding', 0)} / 5",
                    delta=f"{p_sum.get('avg_factual_grounding', 0) - b_sum.get('avg_factual_grounding', 0):+.2f}",
                )
                st.metric(
                    "Avg Latency",
                    f"{p_sum.get('avg_latency_seconds', 0):.3f} s",
                    delta=f"{p_sum.get('avg_latency_seconds', 0) - b_sum.get('avg_latency_seconds', 0):+.3f} s",
                )
                st.metric(
                    "Avg Cost",
                    f"${p_sum.get('avg_cost_usd', 0):.6f} USD",
                    delta=f"${p_sum.get('avg_cost_usd', 0) - b_sum.get('avg_cost_usd', 0):+.6f}",
                )

            st.markdown("---")
            st.markdown("### 📊 Trade-Off Summary Table")
            st.table({
                "Metric": ["Avg Quality", "Factual Grounding", "Avg Latency", "Avg USD Cost", "Editor Approval Rate"],
                "Baseline Control": [b_sum.get("avg_quality_score"), b_sum.get("avg_factual_grounding"), f"{b_sum.get('avg_latency_seconds')}s", f"${b_sum.get('avg_cost_usd')}", "N/A"],
                "Multi-Agent Pipeline": [p_sum.get("avg_quality_score"), p_sum.get("avg_factual_grounding"), f"{p_sum.get('avg_latency_seconds')}s", f"${p_sum.get('avg_cost_usd')}", f"{p_sum.get('editor_approval_rate_pct')}%"],
            })
        else:
            st.info("No pre-computed `evals/benchmark_report.json` found. Run `python evals/benchmark.py` to generate report.")


if __name__ == "__main__":
    main()

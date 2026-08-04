"""Researcher Agent node implementation."""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ContentState
from src.schemas import ResearchOutput
from src.prompts import RESEARCHER_SYSTEM_PROMPT
from src.telemetry import TelemetryTracker, logger
from evals.baseline import get_llm


def researcher_node(state: ContentState, llm: Optional[Any] = None) -> Dict[str, Any]:
    """Execute Research Agent node to gather technical domain notes.

    Args:
        state: Current ContentState dict.
        llm: Optional pre-configured LLM instance (used for testing/mocking).

    Returns:
        Dict with updated 'research_notes' and 'metrics' keys.
    """
    topic = state.get("topic", "System Architecture")
    logger.info(f"[Researcher Node] Executing research for topic: '{topic}'")
    start_time = time.time()

    if llm is None:
        base_llm = get_llm()
        structured_llm = base_llm.with_structured_output(ResearchOutput)
    else:
        structured_llm = llm

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Conduct technical placement research on: {topic}"),
    ]

    response = structured_llm.invoke(messages)

    if isinstance(response, ResearchOutput):
        research_data = response
    elif isinstance(response, dict):
        research_data = ResearchOutput(**response)
    else:
        research_data = response

    # Format research items into list of strings for ContentState
    notes = []
    if hasattr(research_data, "key_points") and research_data.key_points:
        notes.extend([f"Key Point: {kp}" for kp in research_data.key_points])
    if hasattr(research_data, "background_facts") and research_data.background_facts:
        notes.extend([f"Fact: {bf}" for bf in research_data.background_facts])
    if hasattr(research_data, "key_takeaways") and research_data.key_takeaways:
        notes.extend([f"Takeaway: {kt}" for kt in research_data.key_takeaways])

    model_name = getattr(llm, "model_name", getattr(llm, "model", "default_model"))
    metric = TelemetryTracker.record_step(
        node_name="researcher_node",
        model_name=str(model_name),
        start_time=start_time,
        response=response,
    )

    existing_metrics = dict(state.get("metrics", {}))
    existing_metrics["researcher_node"] = metric.model_dump()

    return {
        "research_notes": notes,
        "metrics": existing_metrics,
    }

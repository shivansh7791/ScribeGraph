"""Writer Agent node implementation."""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ContentState
from src.schemas import WriterOutput
from src.prompts import WRITER_SYSTEM_PROMPT
from src.telemetry import TelemetryTracker, logger
from evals.baseline import get_llm


def writer_node(state: ContentState, llm: Optional[Any] = None) -> Dict[str, Any]:
    """Execute Writer Agent node to generate initial draft or write revisions.

    Args:
        state: Current ContentState dict.
        llm: Optional pre-configured LLM instance (used for testing/mocking).

    Returns:
        Dict with updated 'current_draft' and 'metrics' keys.
    """
    topic = state.get("topic", "System Architecture")
    research_notes = state.get("research_notes", [])
    editor_feedback = state.get("editor_feedback")
    current_draft = state.get("current_draft", "")

    start_time = time.time()

    if llm is None:
        base_llm = get_llm()
        structured_llm = base_llm.with_structured_output(WriterOutput)
    else:
        structured_llm = llm

    if editor_feedback and current_draft:
        logger.info(f"[Writer Node] Performing REVISION for topic: '{topic}' based on Editor feedback.")
        user_content = (
            f"Topic: {topic}\n\n"
            f"Current Article Draft:\n{current_draft}\n\n"
            f"Editor Critique & Action Items:\n{editor_feedback}\n\n"
            "Please write an updated, revised draft addressing all points raised by the editor."
        )
    else:
        logger.info(f"[Writer Node] Generating INITIAL DRAFT for topic: '{topic}'.")
        notes_str = "\n".join(f"- {note}" for note in research_notes) if research_notes else "No research notes provided."
        user_content = (
            f"Topic: {topic}\n\n"
            f"Research Notes:\n{notes_str}\n\n"
            "Please write a comprehensive, technical placement interview article draft."
        )

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = structured_llm.invoke(messages)

    if isinstance(response, WriterOutput):
        writer_data = response
    elif isinstance(response, dict):
        writer_data = WriterOutput(**response)
    else:
        writer_data = response

    draft_content = getattr(writer_data, "draft_content", str(response))

    model_name = getattr(llm, "model_name", getattr(llm, "model", "default_model"))
    metric = TelemetryTracker.record_step(
        node_name="writer_node",
        model_name=str(model_name),
        start_time=start_time,
        response=response,
    )

    existing_metrics = dict(state.get("metrics", {}))
    existing_metrics["writer_node"] = metric.model_dump()

    return {
        "current_draft": draft_content,
        "metrics": existing_metrics,
    }

"""Editor Agent node implementation."""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ContentState
from src.schemas import EditorOutput
from src.prompts import EDITOR_SYSTEM_PROMPT
from src.telemetry import TelemetryTracker, logger
from evals.baseline import get_llm


def editor_node(state: ContentState, llm: Optional[Any] = None) -> Dict[str, Any]:
    """Execute Editor Agent node to critique draft, assign scores, and increment revision count.

    Args:
        state: Current ContentState dict.
        llm: Optional pre-configured LLM instance (used for testing/mocking).

    Returns:
        Dict with updated 'editor_feedback', 'revision_count', and 'metrics' keys.
    """
    topic = state.get("topic", "System Architecture")
    current_draft = state.get("current_draft", "")
    current_revision_count = state.get("revision_count", 0)

    logger.info(f"[Editor Node] Evaluating draft for topic: '{topic}' (Current Revisions: {current_revision_count})")
    start_time = time.time()

    if llm is None:
        base_llm = get_llm()
        structured_llm = base_llm.with_structured_output(EditorOutput)
    else:
        structured_llm = llm

    messages = [
        SystemMessage(content=EDITOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Article Draft for Evaluation:\n{current_draft}\n\n"
                "Evaluate the draft thoroughly against technical placement standards."
            )
        ),
    ]

    response = structured_llm.invoke(messages)

    if isinstance(response, EditorOutput):
        editor_data = response
    elif isinstance(response, dict):
        editor_data = EditorOutput(**response)
    else:
        editor_data = response

    is_approved = getattr(editor_data, "is_approved", False)
    critique_str = getattr(editor_data, "detailed_critique", "No detailed critique provided.")
    action_items = getattr(editor_data, "action_items", [])
    if action_items:
        critique_str += "\n\nAction Items:\n" + "\n".join(f"- {item}" for item in action_items)

    new_revision_count = current_revision_count + 1

    model_name = getattr(llm, "model_name", getattr(llm, "model", "default_model"))
    metric = TelemetryTracker.record_step(
        node_name="editor_node",
        model_name=str(model_name),
        start_time=start_time,
        response=response,
    )

    existing_metrics = dict(state.get("metrics", {}))
    existing_metrics["editor_node"] = metric.model_dump()

    return {
        "editor_feedback": critique_str,
        "editor_approved": is_approved,
        "revision_count": new_revision_count,
        "metrics": existing_metrics,
    }

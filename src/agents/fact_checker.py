"""Fact Checker Agent node implementation."""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import ContentState
from src.schemas import FactCheckOutput
from src.prompts import FACT_CHECKER_SYSTEM_PROMPT
from src.telemetry import TelemetryTracker, logger
from evals.baseline import get_llm


def fact_checker_node(state: ContentState, llm: Optional[Any] = None) -> Dict[str, Any]:
    """Execute Fact Checker Agent node to audit technical correctness and update fact_check_passed.

    Args:
        state: Current ContentState dict.
        llm: Optional pre-configured LLM instance (used for testing/mocking).

    Returns:
        Dict with updated 'fact_check_passed' and 'metrics' keys.
    """
    topic = state.get("topic", "System Architecture")
    current_draft = state.get("current_draft", "")

    logger.info(f"[Fact Checker Node] Verifying technical claims for topic: '{topic}'")
    start_time = time.time()

    if llm is None:
        base_llm = get_llm()
        structured_llm = base_llm.with_structured_output(FactCheckOutput)
    else:
        structured_llm = llm

    messages = [
        SystemMessage(content=FACT_CHECKER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Technical Article Draft to Audit:\n{current_draft}\n\n"
                "Audit technical assertions, complexity claims, and system trade-offs."
            )
        ),
    ]

    response = structured_llm.invoke(messages)

    if isinstance(response, FactCheckOutput):
        fact_data = response
    elif isinstance(response, dict):
        fact_data = FactCheckOutput(**response)
    else:
        fact_data = response

    fact_passed = getattr(fact_data, "fact_check_passed", False)

    model_name = getattr(llm, "model_name", getattr(llm, "model", "default_model"))
    metric = TelemetryTracker.record_step(
        node_name="fact_checker_node",
        model_name=str(model_name),
        start_time=start_time,
        response=response,
    )

    existing_metrics = dict(state.get("metrics", {}))
    existing_metrics["fact_checker_node"] = metric.model_dump()

    return {
        "fact_check_passed": fact_passed,
        "metrics": existing_metrics,
    }

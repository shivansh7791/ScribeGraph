"""Router module defining conditional edge logic and loop guards."""

from src.state import ContentState
from src.config import settings
from src.telemetry import logger


def route_after_editor(state: ContentState) -> str:
    """Conditional router determining graph branch after Editor node execution.

    Args:
        state: Current ContentState dictionary.

    Returns:
        Target node string name: 'fact_checker' or 'writer'.
    """
    is_approved = state.get("editor_approved", False)
    revision_count = state.get("revision_count", 0)
    max_revisions = settings.max_revisions

    if is_approved:
        logger.info("[Router] Draft APPROVED by Editor. Routing to 'fact_checker'.")
        return "fact_checker"

    if revision_count < max_revisions:
        logger.info(
            f"[Router] Draft REJECTED by Editor. Revision count: {revision_count}/{max_revisions}. "
            "Routing back to 'writer' for revision loop."
        )
        return "writer"

    logger.warning(
        f"[Router] Loop Guard Triggered! Revision limit reached ({revision_count} >= {max_revisions}). "
        "Forcing exit to 'fact_checker'."
    )
    return "fact_checker"

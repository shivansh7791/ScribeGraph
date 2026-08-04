"""StateGraph builder and compiled graph application module."""

from typing import Callable, Any, Optional
from langgraph.graph import StateGraph, START, END
from src.state import ContentState
from src.agents import researcher_node, writer_node, editor_node, fact_checker_node
from src.router import route_after_editor


def build_graph(
    researcher_fn: Optional[Callable[[ContentState], dict]] = None,
    writer_fn: Optional[Callable[[ContentState], dict]] = None,
    editor_fn: Optional[Callable[[ContentState], dict]] = None,
    fact_checker_fn: Optional[Callable[[ContentState], dict]] = None,
) -> Any:
    """Construct, wire, and compile the Multi-Agent Content Pipeline StateGraph.

    Args:
        researcher_fn: Node function override for researcher (useful for mock testing).
        writer_fn: Node function override for writer.
        editor_fn: Node function override for editor.
        fact_checker_fn: Node function override for fact checker.

    Returns:
        Compiled LangGraph StateGraph runnable app instance.
    """
    builder = StateGraph(ContentState)

    # 1. Add agent nodes
    builder.add_node("researcher", researcher_fn or researcher_node)
    builder.add_node("writer", writer_fn or writer_node)
    builder.add_node("editor", editor_fn or editor_node)
    builder.add_node("fact_checker", fact_checker_fn or fact_checker_node)

    # 2. Wire static edges
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "editor")

    # 3. Wire conditional edge from Editor node with loop guard
    builder.add_conditional_edges(
        "editor",
        route_after_editor,
        {
            "writer": "writer",
            "fact_checker": "fact_checker",
        },
    )

    # 4. Wire final edge to END
    builder.add_edge("fact_checker", END)

    # 5. Compile state graph
    return builder.compile()


# Default compiled graph instance
app = build_graph()

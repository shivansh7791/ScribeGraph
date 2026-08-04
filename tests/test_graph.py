"""Integration unit tests for LangGraph StateGraph orchestration and conditional routing."""

import pytest
from src.state import ContentState
from src.config import settings
from src.router import route_after_editor
from src.graph import build_graph


def test_router_logic():
    """Verify route_after_editor conditional logic directly."""
    # 1. Approved draft -> fact_checker
    state_approved: ContentState = {"editor_approved": True, "revision_count": 1}
    assert route_after_editor(state_approved) == "fact_checker"

    # 2. Rejected draft under max revisions -> writer
    state_rejected: ContentState = {"editor_approved": False, "revision_count": 1}
    assert route_after_editor(state_rejected) == "writer"

    # 3. Max revisions reached -> fact_checker (forced exit)
    state_max_rev: ContentState = {"editor_approved": False, "revision_count": settings.max_revisions}
    assert route_after_editor(state_max_rev) == "fact_checker"


def test_graph_happy_path_execution():
    """Verify single-pass draft approval graph execution path."""
    execution_trace = []

    def mock_researcher(state: ContentState):
        execution_trace.append("researcher")
        return {"research_notes": ["Note 1"]}

    def mock_writer(state: ContentState):
        execution_trace.append("writer")
        return {"current_draft": "Draft v1"}

    def mock_editor(state: ContentState):
        execution_trace.append("editor")
        return {"editor_approved": True, "editor_feedback": "Approved!", "revision_count": 1}

    def mock_fact_checker(state: ContentState):
        execution_trace.append("fact_checker")
        return {"fact_check_passed": True}

    compiled_app = build_graph(
        researcher_fn=mock_researcher,
        writer_fn=mock_writer,
        editor_fn=mock_editor,
        fact_checker_fn=mock_fact_checker,
    )

    initial_state: ContentState = {"topic": "Distributed Consensus", "revision_count": 0}
    final_state = compiled_app.invoke(initial_state)

    assert execution_trace == ["researcher", "writer", "editor", "fact_checker"]
    assert final_state["editor_approved"] is True
    assert final_state["fact_check_passed"] is True
    assert final_state["revision_count"] == 1


def test_graph_revision_loop_execution():
    """Verify draft rejection triggers a revision loop back to writer node."""
    execution_trace = []
    editor_calls = 0

    def mock_researcher(state: ContentState):
        execution_trace.append("researcher")
        return {"research_notes": ["Note 1"]}

    def mock_writer(state: ContentState):
        execution_trace.append("writer")
        return {"current_draft": f"Draft v{state.get('revision_count', 0) + 1}"}

    def mock_editor(state: ContentState):
        nonlocal editor_calls
        editor_calls += 1
        execution_trace.append("editor")
        current_rev = state.get("revision_count", 0) + 1
        # Approve on 2nd editor pass
        approved = (editor_calls == 2)
        return {
            "editor_approved": approved,
            "editor_feedback": "Approved" if approved else "Needs revisions",
            "revision_count": current_rev,
        }

    def mock_fact_checker(state: ContentState):
        execution_trace.append("fact_checker")
        return {"fact_check_passed": True}

    compiled_app = build_graph(
        researcher_fn=mock_researcher,
        writer_fn=mock_writer,
        editor_fn=mock_editor,
        fact_checker_fn=mock_fact_checker,
    )

    initial_state: ContentState = {"topic": "Distributed Consensus", "revision_count": 0}
    final_state = compiled_app.invoke(initial_state)

    # Expected sequence: researcher -> writer -> editor (reject) -> writer -> editor (approve) -> fact_checker
    assert execution_trace == ["researcher", "writer", "editor", "writer", "editor", "fact_checker"]
    assert final_state["editor_approved"] is True
    assert final_state["revision_count"] == 2


def test_graph_loop_guard_max_revisions():
    """Verify graph terminates revision loop when revision_count reaches MAX_REVISIONS limit."""
    execution_trace = []

    def mock_researcher(state: ContentState):
        execution_trace.append("researcher")
        return {"research_notes": ["Note 1"]}

    def mock_writer(state: ContentState):
        execution_trace.append("writer")
        return {"current_draft": f"Draft v{state.get('revision_count', 0) + 1}"}

    def mock_editor(state: ContentState):
        execution_trace.append("editor")
        current_rev = state.get("revision_count", 0) + 1
        # Never approve
        return {
            "editor_approved": False,
            "editor_feedback": "Reject draft",
            "revision_count": current_rev,
        }

    def mock_fact_checker(state: ContentState):
        execution_trace.append("fact_checker")
        return {"fact_check_passed": True}

    compiled_app = build_graph(
        researcher_fn=mock_researcher,
        writer_fn=mock_writer,
        editor_fn=mock_editor,
        fact_checker_fn=mock_fact_checker,
    )

    initial_state: ContentState = {"topic": "Distributed Consensus", "revision_count": 0}
    final_state = compiled_app.invoke(initial_state)

    # With MAX_REVISIONS = 3:
    # 1st editor pass: rev=1 (<3) -> writer
    # 2nd editor pass: rev=2 (<3) -> writer
    # 3rd editor pass: rev=3 (>=3) -> fact_checker
    assert final_state["revision_count"] == 3
    assert final_state["editor_approved"] is False
    assert execution_trace == [
        "researcher", "writer", "editor",
        "writer", "editor",
        "writer", "editor",
        "fact_checker"
    ]

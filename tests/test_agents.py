"""Isolated unit tests for all 4 specialized agent nodes."""

from unittest.mock import MagicMock
import pytest
from src.state import ContentState
from src.schemas import ResearchOutput, WriterOutput, EditorOutput, FactCheckOutput
from src.agents import researcher_node, writer_node, editor_node, fact_checker_node


def test_researcher_node_isolated():
    """Verify researcher_node parses ResearchOutput schema and updates state."""
    mock_llm = MagicMock()
    mock_output = ResearchOutput(
        topic="Distributed Caching",
        key_points=["Cache eviction policies (LRU, LFU)", "Cache invalidation strategies"],
        background_facts=["Redis is single-threaded event loop"],
        key_takeaways=["Explain Cache-Aside vs Write-Through in interview"],
    )
    mock_llm.invoke.return_value = mock_output

    initial_state: ContentState = {
        "topic": "Distributed Caching",
        "research_notes": [],
        "metrics": {},
    }

    result = researcher_node(initial_state, llm=mock_llm)

    assert "research_notes" in result
    assert len(result["research_notes"]) == 4
    assert any("LRU" in note for note in result["research_notes"])
    assert "metrics" in result
    assert "researcher_node" in result["metrics"]


def test_writer_node_initial_draft():
    """Verify writer_node generates initial draft when no editor_feedback is present."""
    mock_llm = MagicMock()
    mock_output = WriterOutput(
        draft_content="# Distributed Caching Interview Guide\n\nExecutive Summary...",
        word_count=500,
        target_audience="Senior System Architect",
    )
    mock_llm.invoke.return_value = mock_output

    initial_state: ContentState = {
        "topic": "Distributed Caching",
        "research_notes": ["Key Point: Cache eviction"],
        "metrics": {},
    }

    result = writer_node(initial_state, llm=mock_llm)

    assert "current_draft" in result
    assert "Distributed Caching Interview Guide" in result["current_draft"]
    assert "metrics" in result
    assert "writer_node" in result["metrics"]


def test_writer_node_revision_loop():
    """Verify writer_node handles revision request when editor_feedback is provided."""
    mock_llm = MagicMock()
    mock_output = WriterOutput(
        draft_content="# Distributed Caching Guide (Revised)\n\nAdded code snippet...",
        word_count=650,
        target_audience="Senior System Architect",
    )
    mock_llm.invoke.return_value = mock_output

    state_with_critique: ContentState = {
        "topic": "Distributed Caching",
        "current_draft": "# Draft 1",
        "editor_feedback": "Critique: Needs code example for LRU cache",
        "revision_count": 1,
        "metrics": {},
    }

    result = writer_node(state_with_critique, llm=mock_llm)

    assert "current_draft" in result
    assert "(Revised)" in result["current_draft"]


def test_editor_node_and_revision_increment():
    """Verify editor_node critiques draft and increments revision_count."""
    mock_llm = MagicMock()
    mock_output = EditorOutput(
        is_approved=False,
        quality_score=6,
        hook_rating=3,
        detailed_critique="Detailed critique: Technical depth in section 3 is lacking.",
        action_items=["Add latency trade-off comparison table"],
    )
    mock_llm.invoke.return_value = mock_output

    state: ContentState = {
        "topic": "Distributed Caching",
        "current_draft": "# Draft Content",
        "revision_count": 0,
        "metrics": {},
    }

    result = editor_node(state, llm=mock_llm)

    assert "editor_feedback" in result
    assert "action_items" in result["editor_feedback"] or "Action Items:" in result["editor_feedback"]
    assert result["revision_count"] == 1
    assert "metrics" in result
    assert "editor_node" in result["metrics"]


def test_fact_checker_node():
    """Verify fact_checker_node evaluates claims and updates fact_check_passed."""
    mock_llm = MagicMock()
    mock_output = FactCheckOutput(
        fact_check_passed=True,
        verification_score=92,
        verified_claims=["Redis LRU policy verified", "Consistent hashing math verified"],
        flagged_discrepancies=[],
    )
    mock_llm.invoke.return_value = mock_output

    state: ContentState = {
        "topic": "Distributed Caching",
        "current_draft": "# Draft Content",
        "fact_check_passed": False,
        "metrics": {},
    }

    result = fact_checker_node(state, llm=mock_llm)

    assert result["fact_check_passed"] is True
    assert "metrics" in result
    assert "fact_checker_node" in result["metrics"]

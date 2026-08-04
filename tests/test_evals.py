"""Unit and integration tests for Phase 5 Evaluation Suite and Benchmark Runner."""

import json
from pathlib import Path
import pytest
from evals.evaluators import EvalRubric, evaluate_content, check_word_count, check_formatting_heuristics
from evals.benchmark import run_benchmark, create_mock_llm_for_benchmark


def test_dataset_json_structure():
    """Verify dataset.json exists, is valid JSON, and contains expected fields."""
    dataset_path = Path("evals/dataset.json")
    assert dataset_path.exists()

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 5

    for item in data:
        assert "id" in item
        assert "topic" in item
        assert "ground_truth_facts" in item
        assert "target_audience" in item
        assert isinstance(item["ground_truth_facts"], list)


def test_eval_rubric_validation():
    """Verify EvalRubric schema validates score boundaries accurately."""
    rubric = EvalRubric(
        factual_grounding=5,
        hook_quality=4,
        clarity_structure=5,
        actionability=4,
        overall_score=4.5,
        reasoning="Excellent technical depth and factual accuracy.",
    )

    assert rubric.factual_grounding == 5
    assert rubric.overall_score == 4.5

    # Test out of bounds validation
    with pytest.raises(Exception):
        EvalRubric(
            factual_grounding=6,  # max is 5
            hook_quality=4,
            clarity_structure=5,
            actionability=4,
            overall_score=4.5,
            reasoning="Invalid score test",
        )


def test_rule_based_heuristics():
    """Verify check_word_count and check_formatting_heuristics rule logic."""
    short_text = "Word " * 50
    valid_text = "# Header\n\n- Point 1\n- Point 2\n\n**Bold callout**\n" + ("Word " * 200)

    assert check_word_count(short_text, min_words=100) is False
    assert check_word_count(valid_text, min_words=150, max_words=800) is True

    fmt = check_formatting_heuristics(valid_text)
    assert fmt["has_headers"] is True
    assert fmt["has_bullet_points"] is True
    assert fmt["has_callouts_or_bold"] is True


def test_benchmark_runner_mock_execution():
    """Verify run_benchmark in mock_mode aggregates comparative metrics properly."""
    report = run_benchmark(dataset_path="evals/dataset.json", mock_mode=True)

    assert "dataset_size" in report
    assert report["dataset_size"] >= 5

    assert "baseline_summary" in report
    assert "pipeline_summary" in report

    b_sum = report["baseline_summary"]
    p_sum = report["pipeline_summary"]

    assert b_sum["avg_quality_score"] > 0
    assert p_sum["avg_quality_score"] > 0
    assert p_sum["editor_approval_rate_pct"] >= 0.0

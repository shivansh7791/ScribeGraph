"""LLM-as-a-Judge and rule-based heuristic evaluators."""

import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from evals.baseline import get_llm
from src.telemetry import logger


class EvalRubric(BaseModel):
    """Pydantic v2 schema for LLM-as-a-Judge evaluation rubric."""
    factual_grounding: int = Field(
        ...,
        ge=1,
        le=5,
        description="Factual accuracy score (1-5) verifying alignment with ground truth technical facts"
    )
    hook_quality: int = Field(
        ...,
        ge=1,
        le=5,
        description="Introduction and executive summary hook score (1-5)"
    )
    clarity_structure: int = Field(
        ...,
        ge=1,
        le=5,
        description="Markdown formatting, section organization, and clarity score (1-5)"
    )
    actionability: int = Field(
        ...,
        ge=1,
        le=5,
        description="Actionability for placement interview preparation (1-5)"
    )
    overall_score: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Weighted overall quality score from 1.0 to 5.0"
    )
    reasoning: str = Field(
        ...,
        description="Detailed qualitative reasoning explaining score assignments"
    )


JUDGE_SYSTEM_PROMPT = """You are an Expert LLM Evaluation Judge and Senior Staff System Architect.
Your task is to perform a rigorous evaluation of a technical placement interview guide against specified ground truth facts.

Evaluation Rubric Criteria (1 to 5 scale):
1. Factual Grounding (1-5): Does the article correctly reflect the provided ground truth technical facts without hallucination or factual inaccuracies?
2. Hook Quality (1-5): Is the executive summary engaging, authoritative, and relevant for senior engineering candidates?
3. Clarity & Structure (1-5): Is the draft clearly organized with Markdown headers, bullet points, and logical progression?
4. Actionability (1-5): Does the guide provide clear trade-offs, architecture insights, and placement interview answers?
5. Overall Score (1.0-5.0): Compute a holistic grade reflecting overall quality.

Respond with structured JSON strictly matching the output schema provided."""


def evaluate_content(
    topic: str,
    draft: str,
    facts: List[str],
    llm: Optional[Any] = None,
) -> EvalRubric:
    """Evaluate generated content draft using an LLM-as-a-Judge with structured output.

    Args:
        topic: Technical subject topic.
        draft: Generated article markdown draft.
        facts: List of ground truth technical facts.
        llm: Optional judge LLM instance.

    Returns:
        EvalRubric instance containing numerical scores and qualitative reasoning.
    """
    logger.info(f"[LLM Judge] Evaluating draft for topic: '{topic}'")
    if llm is None:
        base_llm = get_llm()
        judge_llm = base_llm.with_structured_output(EvalRubric)
    elif hasattr(llm, "with_structured_output") and callable(getattr(llm, "with_structured_output")):
        try:
            judge_llm = llm.with_structured_output(EvalRubric)
        except Exception:
            judge_llm = llm
    else:
        judge_llm = llm

    facts_str = "\n".join(f"- {fact}" for fact in facts) if facts else "No ground truth facts provided."
    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Ground Truth Technical Facts:\n{facts_str}\n\n"
        f"Article Draft to Evaluate:\n{draft}\n\n"
        "Evaluate the draft according to the rubric criteria."
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = judge_llm.invoke(messages)

    if isinstance(response, EvalRubric):
        return response
    elif isinstance(response, dict):
        return EvalRubric(**response)
    else:
        return response


def check_word_count(draft: str, min_words: int = 150, max_words: int = 800) -> bool:
    """Rule-based heuristic: Check if article draft word count falls within bounds."""
    if not draft:
        return False
    words = draft.split()
    count = len(words)
    return min_words <= count <= max_words


def check_formatting_heuristics(draft: str) -> Dict[str, bool]:
    """Rule-based heuristic: Check structural Markdown formatting elements."""
    if not draft:
        return {"has_headers": False, "has_bullet_points": False, "has_callouts_or_bold": False}

    has_headers = any(line.strip().startswith("#") for line in draft.splitlines())
    has_bullet_points = any(
        line.strip().startswith("- ") or line.strip().startswith("* ") or line.strip().startswith("1. ")
        for line in draft.splitlines()
    )
    has_callouts_or_bold = "**" in draft or "```" in draft or "> " in draft

    return {
        "has_headers": has_headers,
        "has_bullet_points": has_bullet_points,
        "has_callouts_or_bold": has_callouts_or_bold,
    }

"""Pydantic v2 schemas for agent structured outputs."""

from typing import List
from pydantic import BaseModel, Field


class ResearchOutput(BaseModel):
    """Structured output for the Research Agent node."""
    topic: str = Field(..., description="The technical topic being researched")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key system design concepts, core mechanisms, and architectural components"
    )
    background_facts: List[str] = Field(
        default_factory=list,
        description="Background technical facts, historical context, and foundational principles"
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Actionable tech placement interview takeaways and interview answer strategies"
    )


class WriterOutput(BaseModel):
    """Structured output for the Writer Agent node."""
    draft_content: str = Field(..., description="Full technical article draft written in Markdown")
    word_count: int = Field(default=0, description="Approximate word count of generated draft")
    target_audience: str = Field(
        default="Senior & Staff Engineers",
        description="Target audience skill level for placement preparation"
    )


class EditorOutput(BaseModel):
    """Structured output for the Editor Agent node."""
    is_approved: bool = Field(
        ...,
        description="True if draft meets technical quality standard; False if revisions are required"
    )
    quality_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Overall article quality score out of 10"
    )
    hook_rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Introduction and executive summary hook rating out of 5"
    )
    detailed_critique: str = Field(
        ...,
        description="Detailed constructive criticism outlining strengths and weaknesses"
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Specific actionable improvements required for the next revision"
    )


class FactCheckOutput(BaseModel):
    """Structured output for the Fact Checker Agent node."""
    fact_check_passed: bool = Field(
        ...,
        description="True if technical claims are verified and accurate; False if discrepancies found"
    )
    verification_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Technical factual correctness score out of 100"
    )
    verified_claims: List[str] = Field(
        default_factory=list,
        description="List of verified accurate technical claims and architecture specs"
    )
    flagged_discrepancies: List[str] = Field(
        default_factory=list,
        description="List of inaccurate, misleading, or unverified claims needing correction"
    )

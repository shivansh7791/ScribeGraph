"""State schema and Pydantic validation models for LangGraph orchestration."""

from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field


class NodeMetric(BaseModel):
    """Execution metrics for an individual agent/node in the pipeline."""
    node_name: str = Field(description="Name of the node or step executed")
    prompt_tokens: int = Field(default=0, description="Input token count")
    completion_tokens: int = Field(default=0, description="Output token count")
    total_tokens: int = Field(default=0, description="Total tokens used")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    latency_seconds: float = Field(default=0.0, description="Execution latency in seconds")
    model_name: str = Field(default="", description="Model used for execution")


class PipelineMetrics(BaseModel):
    """Aggregate metrics across all node executions."""
    total_prompt_tokens: int = Field(default=0)
    total_completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    total_latency_seconds: float = Field(default=0.0)
    node_metrics: Dict[str, NodeMetric] = Field(default_factory=dict)


class ContentState(TypedDict, total=False):
    """LangGraph State Definition for Multi-Agent Content Pipeline."""
    topic: str
    research_notes: List[str]
    current_draft: str
    editor_feedback: Optional[str]
    editor_approved: Optional[bool]
    revision_count: int
    fact_check_passed: bool
    metrics: Dict[str, Any]


class ContentStateModel(BaseModel):
    """Pydantic v2 validation schema mirroring ContentState for strict runtime validation."""
    topic: str = Field(..., description="Subject or prompt for content generation")
    research_notes: List[str] = Field(default_factory=list, description="Gathered research notes")
    current_draft: str = Field(default="", description="Current markdown article draft")
    editor_feedback: Optional[str] = Field(default=None, description="Editor review and recommendations")
    editor_approved: Optional[bool] = Field(default=None, description="Approval flag from Editor node")
    revision_count: int = Field(default=0, description="Number of revisions completed")
    fact_check_passed: bool = Field(default=False, description="Verification status of technical claims")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics per node")

    def to_dict(self) -> ContentState:
        """Convert Pydantic model to TypedDict ContentState."""
        return ContentState(
            topic=self.topic,
            research_notes=self.research_notes,
            current_draft=self.current_draft,
            editor_feedback=self.editor_feedback,
            editor_approved=self.editor_approved,
            revision_count=self.revision_count,
            fact_check_passed=self.fact_check_passed,
            metrics=self.metrics,
        )

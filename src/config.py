"""Configuration settings for the Multi-Agent Content Pipeline."""

import os
from typing import Dict, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class ModelPricing:
    """Estimated pricing per 1,000 tokens (input_cost, output_cost) in USD."""
    PRICING_MAP: Dict[str, Tuple[float, float]] = {
        # provider / model -> (input_per_1k, output_per_1k)
        "gemini-2.5-flash": (0.000075, 0.0003),
        "gemini-1.5-pro": (0.00125, 0.005),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4o": (0.0025, 0.010),
        "llama-3.3-70b-versatile": (0.00059, 0.00079),
        "mixtral-8x7b-32768": (0.00027, 0.00027),
    }

    @classmethod
    def get_pricing(cls, model_name: str) -> Tuple[float, float]:
        """Return (input_cost_per_1k, output_cost_per_1k) for a model."""
        return cls.PRICING_MAP.get(model_name.lower(), (0.0001, 0.0004))


class Settings(BaseSettings):
    """Application Settings using Pydantic Settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Default LLM configurations
    default_model_provider: str = Field(default="gemini")
    default_model_name: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    max_revisions: int = Field(default=3, description="Maximum editor revision loops allowed")

    # Logging & Telemetry
    log_level: str = Field(default="INFO")
    enable_telemetry: bool = Field(default=True)

    # LangSmith Observability
    langchain_tracing_v2: bool = Field(default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true")
    langchain_endpoint: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_ENDPOINT", "https://aws.api.smith.langchain.com"))
    langchain_api_key: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", ""))
    langchain_project: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "ScribeGraph"))


settings = Settings()


def setup_observability() -> bool:
    """Validate and set up LangSmith / LangChain V2 tracing environment variables.

    Returns:
        bool: True if LangSmith tracing was enabled and configured, False otherwise.
    """
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        return True
    elif settings.langchain_tracing_v2:
        # Tracing enabled without API key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        return False
    return False

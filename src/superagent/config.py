"""Configuration objects and utilities for the SuperAgent project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpenAIConfig:
    """Holds configuration for accessing the OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.7
    timeout: Optional[float] = 60.0


@dataclass(frozen=True)
class TavilyConfig:
    """Configuration for Tavily search integration."""

    api_key: str
    max_results: int = 5


@dataclass(frozen=True)
class SuperAgentConfig:
    """Top level configuration container."""

    openai: OpenAIConfig
    tavily: TavilyConfig
    enable_search_memory: bool = True
    enable_task_memory: bool = True


DEFAULT_CONFIG = SuperAgentConfig(
    openai=OpenAIConfig(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="ce3021da-bb4e-4038-8238-08d0987d590e",
        model_name="ep-20250331104046-lr9zp",
    ),
    tavily=TavilyConfig(
        api_key="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",
    ),
)

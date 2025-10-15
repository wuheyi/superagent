"""Tool factory helpers."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

from .config import TavilyConfig
from .memory.mem0_adapter import MemoryStore


def build_tavily_tool(config: TavilyConfig) -> TavilySearchResults:
    """Instantiate the Tavily search tool with project defaults."""

    return TavilySearchResults(tavily_api_key=config.api_key, max_results=config.max_results)


def build_memory_tools(memory_store: MemoryStore) -> List[Tool]:
    """Expose read/write memory utilities as LangChain tools."""

    def write_memory(content: str, **metadata: Any) -> str:
        memory_store.write(content, **metadata)
        return "Memory stored."

    def read_memory(query: str, limit: int = 5) -> str:
        matches = list(memory_store.search(query, limit=limit))
        if not matches:
            return "No related memories."
        return "\n".join(
            f"- {record.content} | metadata: {record.metadata}" for record in matches
        )

    return [
        Tool(
            name="write_memory",
            func=write_memory,
            description="Persist an insight or result into long-term memory.",
        ),
        Tool(
            name="read_memory",
            func=read_memory,
            description="Retrieve relevant historical memories for the given query.",
        ),
    ]

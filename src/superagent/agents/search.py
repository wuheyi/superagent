"""Agent responsible for research-style interactions."""
from __future__ import annotations

from typing import Iterable, Sequence

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..prompts import SEARCH_AGENT_SYSTEM_PROMPT


def build_search_agent(llm: ChatOpenAI, tools: Sequence[Tool]):
    """Create a research agent that can call Tavily."""

    return create_react_agent(
        llm=llm,
        tools=list(tools),
        state_modifier=SEARCH_AGENT_SYSTEM_PROMPT,
    )

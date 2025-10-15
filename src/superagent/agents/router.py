"""Routing agent that decides which strategy to use."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..prompts import ROUTER_SYSTEM_PROMPT


@dataclass
class RouterDecision:
    route: str
    needs_search: bool
    justification: str


class RouterAgent:
    """LLM-based classifier that selects the processing route."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

    async def aroute(self, question: str) -> RouterDecision:
        response = await self._llm.ainvoke(self._prompt.format(question=question))
        payload = json.loads(response.content)
        return RouterDecision(
            route=payload["route"],
            needs_search=bool(payload.get("needs_search", False)),
            justification=payload.get("justification", ""),
        )

    def route(self, question: str) -> RouterDecision:
        response = self._llm.invoke(self._prompt.format(question=question))
        payload = json.loads(response.content)
        return RouterDecision(
            route=payload["route"],
            needs_search=bool(payload.get("needs_search", False)),
            justification=payload.get("justification", ""),
        )

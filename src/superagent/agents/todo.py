"""Todo writing agent for complex tasks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..prompts import TODO_WRITER_SYSTEM_PROMPT


@dataclass
class Todo:
    description: str
    prompt: str
    tools: List[str]
    reason: str
    sub_query: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Todo":
        return cls(
            description=payload["description"],
            prompt=payload["prompt"],
            tools=list(payload.get("tools", [])),
            reason=payload["reason"],
            sub_query=payload["sub_query"],
        )


class TodoWriterAgent:
    """Agent dedicated to decomposing tasks into todos."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TODO_WRITER_SYSTEM_PROMPT),
                ("human", "{context}"),
            ]
        )

    def create_todos(self, user_request: str) -> List[Todo]:
        response = self._llm.invoke(self._prompt.format(context=user_request))
        payload = json.loads(response.content)
        todos = [Todo.from_dict(item) for item in payload.get("todos", [])]
        if not todos:
            raise ValueError("Todo writer returned no tasks; ensure the request is complex enough.")
        return todos

    async def acreate_todos(self, user_request: str) -> List[Todo]:
        response = await self._llm.ainvoke(self._prompt.format(context=user_request))
        payload = json.loads(response.content)
        todos = [Todo.from_dict(item) for item in payload.get("todos", [])]
        if not todos:
            raise ValueError("Todo writer returned no tasks; ensure the request is complex enough.")
        return todos

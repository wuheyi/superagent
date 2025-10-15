"""Final response synthesis agent."""
from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..prompts import FINAL_ANSWER_SYSTEM_PROMPT


class FinalAnswerAgent:
    """Synthesises the final response from all intermediate results."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FINAL_ANSWER_SYSTEM_PROMPT),
                (
                    "human",
                    "<user_request>{user_request}</user_request>\n"
                    "<router_note>{router_note}</router_note>\n"
                    "<todo_results>{todo_results}</todo_results>",
                ),
            ]
        )

    def run(self, user_request: str, router_note: str, todo_results: List[str]) -> str:
        response = self._llm.invoke(
            self._prompt.format(
                user_request=user_request,
                router_note=router_note,
                todo_results="\n\n".join(todo_results) if todo_results else "No todos executed.",
            )
        )
        return response.content

    async def arun(self, user_request: str, router_note: str, todo_results: List[str]) -> str:
        response = await self._llm.ainvoke(
            self._prompt.format(
                user_request=user_request,
                router_note=router_note,
                todo_results="\n\n".join(todo_results) if todo_results else "No todos executed.",
            )
        )
        return response.content

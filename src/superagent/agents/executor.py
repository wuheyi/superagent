"""Todo execution utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..prompts import EXECUTOR_SYSTEM_PROMPT
from ..memory.mem0_adapter import MemoryStore
from .todo import Todo


class TodoExecutor:
    """Runs each todo using a LangGraph ReAct agent with memory support."""

    def __init__(self, llm: ChatOpenAI, tool_catalog: Dict[str, Tool], memory_store: MemoryStore) -> None:
        self._llm = llm
        self._tool_catalog = tool_catalog
        self._memory_store = memory_store

    def _resolve_tools(self, todo: Todo) -> List[Tool]:
        tools: List[Tool] = []
        for name in todo.tools:
            tool = self._tool_catalog.get(name)
            if tool is None:
                raise KeyError(f"Tool '{name}' requested by todo but not registered.")
            tools.append(tool)
        # Always expose memory helpers
        for memory_tool in ("write_memory", "read_memory"):
            tool = self._tool_catalog.get(memory_tool)
            if tool and tool not in tools:
                tools.append(tool)
        return tools

    def run(self, todo: Todo) -> str:
        tools = self._resolve_tools(todo)
        agent = create_react_agent(
            llm=self._llm,
            tools=tools,
            state_modifier=self._build_state_modifier(todo),
        )
        result = agent.invoke(
            {
                "input": self._build_user_message(todo),
            }
        )
        output = result.get("output", "")
        self._memory_store.write(
            content=f"Todo: {todo.description}\nResult: {output}",
            sub_query=todo.sub_query,
        )
        return output

    async def arun(self, todo: Todo) -> str:
        tools = self._resolve_tools(todo)
        agent = create_react_agent(
            llm=self._llm,
            tools=tools,
            state_modifier=self._build_state_modifier(todo),
        )
        result = await agent.ainvoke(
            {
                "input": self._build_user_message(todo),
            }
        )
        output = result.get("output", "")
        self._memory_store.write(
            content=f"Todo: {todo.description}\nResult: {output}",
            sub_query=todo.sub_query,
        )
        return output

    def _build_state_modifier(self, todo: Todo) -> str:
        return (
            f"{EXECUTOR_SYSTEM_PROMPT}\n\n"
            "<todo_spec>\n"
            f"Description: {todo.description}\n"
            f"Reason: {todo.reason}\n"
            f"Focused query: {todo.sub_query}\n"
            "</todo_spec>\n\n"
            f"{todo.prompt}"
        )

    def _build_user_message(self, todo: Todo) -> str:
        return (
            "Follow the todo specification to produce a concrete result. "
            "Respond with the outcomes, references, and any next steps."
        )

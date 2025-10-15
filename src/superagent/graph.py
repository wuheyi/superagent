"""Factory for assembling the SuperAgent component graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from .agents.executor import TodoExecutor
from .agents.finalizer import FinalAnswerAgent
from .agents.router import RouterAgent
from .agents.search import build_search_agent
from .agents.todo import TodoWriterAgent
from .config import DEFAULT_CONFIG, SuperAgentConfig
from .memory.mem0_adapter import MemoryStore
from .tools import build_memory_tools, build_tavily_tool


@dataclass
class SuperAgentGraph:
    router: RouterAgent
    simple_llm: ChatOpenAI
    search_agent: any
    todo_writer: TodoWriterAgent
    todo_executor: TodoExecutor
    final_agent: FinalAnswerAgent
    tool_catalog: Dict[str, Tool]
    memory_store: MemoryStore


def build_superagent_graph(config: SuperAgentConfig = DEFAULT_CONFIG) -> SuperAgentGraph:
    base_llm = ChatOpenAI(
        model=config.openai.model_name,
        api_key=config.openai.api_key,
        base_url=config.openai.base_url,
        temperature=config.openai.temperature,
        timeout=config.openai.timeout,
    )

    router_llm = ChatOpenAI(
        model=config.openai.model_name,
        api_key=config.openai.api_key,
        base_url=config.openai.base_url,
        temperature=0.0,
        timeout=config.openai.timeout,
    )

    simple_llm = ChatOpenAI(
        model=config.openai.model_name,
        api_key=config.openai.api_key,
        base_url=config.openai.base_url,
        temperature=0.2,
        timeout=config.openai.timeout,
    )

    tavily_tool = build_tavily_tool(config.tavily)

    memory_store = MemoryStore(namespace="superagent")
    memory_tools = build_memory_tools(memory_store)

    tool_catalog: Dict[str, Tool] = {
        "TavilySearchResults": tavily_tool,
    }
    for tool in memory_tools:
        tool_catalog[tool.name] = tool

    search_agent = build_search_agent(base_llm, [tavily_tool])
    todo_writer = TodoWriterAgent(base_llm)
    todo_executor = TodoExecutor(base_llm, tool_catalog=tool_catalog, memory_store=memory_store)
    final_agent = FinalAnswerAgent(base_llm)
    router = RouterAgent(router_llm)

    return SuperAgentGraph(
        router=router,
        simple_llm=simple_llm,
        search_agent=search_agent,
        todo_writer=todo_writer,
        todo_executor=todo_executor,
        final_agent=final_agent,
        tool_catalog=tool_catalog,
        memory_store=memory_store,
    )

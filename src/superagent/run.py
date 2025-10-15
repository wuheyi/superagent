"""Runtime orchestration for the SuperAgent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from .agents.todo import Todo
from .config import DEFAULT_CONFIG, SuperAgentConfig
from .graph import SuperAgentGraph, build_superagent_graph
from .prompts import SIMPLE_RESPONSE_SYSTEM_PROMPT


@dataclass
class SuperAgentResult:
    route: str
    justification: str
    answer: str
    todos: List[Todo]


class SuperAgent:
    """High-level interface that coordinates the specialised agents."""

    def __init__(self, config: SuperAgentConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._graph: SuperAgentGraph = build_superagent_graph(config)
        self._simple_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SIMPLE_RESPONSE_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

    def respond(self, question: str) -> SuperAgentResult:
        router_decision = self._graph.router.route(question)

        if router_decision.route == "simple":
            response = self._graph.simple_llm.invoke(
                self._simple_prompt.format(question=question)
            )
            return SuperAgentResult(
                route="simple",
                justification=router_decision.justification,
                answer=response.content,
                todos=[],
            )

        if router_decision.route == "search":
            result = self._graph.search_agent.invoke({"input": question})
            output = result.get("output", "")
            self._graph.memory_store.write(
                content=f"Search answer for '{question}': {output}",
                route="search",
            )
            return SuperAgentResult(
                route="search",
                justification=router_decision.justification,
                answer=output,
                todos=[],
            )

        if router_decision.route == "complex":
            search_summary = ""
            context_for_todos = question
            pre_notes: List[str] = []
            if router_decision.needs_search:
                search_result = self._graph.search_agent.invoke({"input": question})
                search_summary = search_result.get("output", "")
                if search_summary:
                    pre_notes.append(f"Initial research:\n{search_summary}")
                    context_for_todos = (
                        f"User request: {question}\n\n"
                        f"Initial research findings:\n{search_summary}"
                    )

            todos = self._graph.todo_writer.create_todos(context_for_todos)

            todo_outputs: List[str] = []
            for todo in todos:
                result_text = self._graph.todo_executor.run(todo)
                todo_outputs.append(f"Todo: {todo.description}\nOutput: {result_text}")

            combined_notes = pre_notes + todo_outputs
            final_answer = self._graph.final_agent.run(
                user_request=question,
                router_note=router_decision.justification,
                todo_results=combined_notes,
            )

            self._graph.memory_store.write(
                content=f"Final answer for '{question}': {final_answer}",
                route="complex",
            )

            return SuperAgentResult(
                route="complex",
                justification=router_decision.justification,
                answer=final_answer,
                todos=todos,
            )

        raise ValueError(f"Unknown route: {router_decision.route}")

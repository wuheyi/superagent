# SuperAgent

SuperAgent demonstrates a layered, LangGraph-powered orchestration pipeline that routes user
requests between direct answers, research-powered replies, and a multi-todo execution flow
with long-term memory via mem0.

## Features

- **Adaptive routing** — a router agent classifies questions as simple, search-heavy, or
  complex multi-step tasks.
- **Tavily research** — up-to-date information retrieval with Tavily search integration
  for both standalone queries and complex task bootstrapping.
- **Todo architecture** — complex tasks are decomposed into world-class todo prompts with
  description, tool selections, reasoning, and focused sub-queries.
- **LangGraph ReAct executors** — each todo is executed by a dedicated ReAct agent that can
  access the specified tools and record insights.
- **mem0 long-term memory** — executor outcomes and final answers are persisted for future
  retrieval.
- **Final synthesis** — a final reviewer agent summarises all intermediate work into a
  polished Markdown response.

## Getting started

```bash
pip install -r requirements.txt  # ensure langgraph, langchain, langchain-openai, mem0, tavily, etc.
```

Then, in Python:

```python
from superagent import SuperAgent

agent = SuperAgent()
result = agent.respond("帮我总结一下人工智能代理的最新发展，并规划一个学习路线")
print(result.route)
print(result.answer)
```

> **Note**
> The project reads API credentials from `superagent.config.DEFAULT_CONFIG`. Adjust those
> values or supply a custom `SuperAgentConfig` when instantiating `SuperAgent`.

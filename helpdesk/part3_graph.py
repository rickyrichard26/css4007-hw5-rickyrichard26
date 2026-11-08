import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .part2_graph import MAX_STEPS, SYSTEM_PROMPT, AgentState, make_llm

MEMORY_RULES = """
Memory rules:
- Earlier messages in this conversation are available to you. When the user says
  "that", "it" or "I already tried", use them to understand what is meant.
- If the user asks you to reply in a particular language (for example "reply in
  Kazakh from now on"), call save_language_preference once with that language,
  then reply in it.
- Use only details the user stated in this conversation. Never guess missing ones.
- Never request or store passwords, API keys or recovery codes."""


def build_memory_graph(tools, user_memory, user_id, checkpointer, llm=None,
                       max_steps: int = MAX_STEPS):
    llm = llm or make_llm()
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def system_prompt() -> str:
        text = SYSTEM_PROMPT + "\n" + MEMORY_RULES
        language = user_memory.get(user_id).get("reply_language")  # user memory read
        if language:
            text += f"\n\nSaved user preference: always reply in {language}."
        return text

    def model_node(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {
                "messages": [AIMessage(content=(
                    "I stopped because I reached the step limit. "
                    "Please try again or contact a human helpdesk agent."))],
                "limit_reached": True,
            }
        reply = llm_with_tools.invoke(
            [SystemMessage(content=system_prompt())] + state["messages"]
        )
        return {"messages": [reply], "steps": steps + 1}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            try:
                output = tools_by_name[call["name"]].invoke(call["args"])
            except Exception as exc:
                output = {"error": "tool_failed", "detail": str(exc)}
            results.append(ToolMessage(
                content=json.dumps(output, ensure_ascii=False),
                tool_call_id=call["id"],
                name=call["name"],
            ))
        return {"messages": results}

    def route_after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tool_node" if getattr(last, "tool_calls", None) else END

    g = StateGraph(AgentState)
    g.add_node("model_node", model_node)
    g.add_node("tool_node", tool_node)
    g.add_edge(START, "model_node")
    g.add_conditional_edges(
        "model_node", route_after_model, {"tool_node": "tool_node", END: END}
    )
    g.add_edge("tool_node", "model_node")
    return g.compile(checkpointer=checkpointer)  # conversation memory

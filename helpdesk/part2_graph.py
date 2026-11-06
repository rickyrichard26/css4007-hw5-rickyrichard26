import json
import os

from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MAX_STEPS = 8  # maksimum pemanggilan model dalam satu permintaan

SYSTEM_PROMPT = """You are a university IT helpdesk assistant (practice exercise, fictional data).
Scope: Wi-Fi troubleshooting, password recovery guidance, service status lookup,
and preparing LOCAL ticket drafts. Politely decline anything else.

Rules:
- Use find_instructions for steps. Never invent instructions.
- Use check_service_status when the user asks whether a service is down, and
  before preparing any ticket. Never guess a service status.
- If the status is 'operational': you may prepare a ticket if the user wants one.
- If the status is 'outage': do NOT prepare a ticket. Explain that there is a
  known outage. Do not invent a repair time.
- If the status is 'unknown': say the status cannot be confirmed (it is neither
  operational nor a confirmed outage) and offer to prepare a draft ticket.
- prepare_ticket needs a problem description and the steps already tried. If the
  user has not given them, ask the user. Never invent details.
- Never ask for or store passwords, API keys or recovery codes.
- After using tools, answer clearly and briefly."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    steps: int
    limit_reached: bool


def make_llm():
    load_dotenv()
    return ChatOpenAI(
        model=os.environ["MODEL_NAME"],
        temperature=0,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


def build_agent_graph(tools, llm=None, max_steps: int = MAX_STEPS):
    llm = llm or make_llm()
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def model_node(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:  # batas loop ditegakkan di sini
            return {
                "messages": [AIMessage(content=(
                    "I stopped because I reached the step limit. "
                    "Please try again or contact a human helpdesk agent."))],
                "limit_reached": True,
            }
        reply = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {"messages": [reply], "steps": steps + 1}

    def tool_node(state: AgentState) -> dict:
        """Program kita yang MENJALANKAN tool yang diminta model."""
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            try:
                output = tools_by_name[call["name"]].invoke(call["args"])
            except Exception as exc:  # loop tidak boleh crash karena satu tool gagal
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
    return g.compile()

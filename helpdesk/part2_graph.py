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
- TICKETS: call prepare_ticket only when the user's message explicitly asks for
  a ticket (for example "prepare a ticket" or "if it is operating, prepare a
  ticket") or accepts your offer of one. A user who only describes a problem,
  or only asks you to remember a preference, has NOT asked for a ticket: do not
  call prepare_ticket and do not mention tickets.
- If the user explicitly asks for a ticket and has already said what is wrong
  and what they tried, do NOT ask for those details again. Use only the user's
  own words for summary and steps_tried, and call prepare_ticket. If the user
  asks for a ticket but has not said what is wrong or what they tried, ask for
  both before calling any tool.
- Conditional ticket ("check whether X is down; if it is operating, prepare a
  ticket"): first call check_service_status, then act on the status:
  * 'operational': call prepare_ticket in your next step.
  * 'outage': do NOT prepare it. Say there is a known outage and that is why no
    ticket was prepared. Do not invent a repair time.
  * 'unknown': do NOT prepare it yet. Say the status could not be determined and
    cannot be confirmed as operating, and ask whether the user wants a draft
    ticket explaining that the status could not be checked.
- Never ask for or store passwords, API keys or recovery codes.
- After using tools, answer clearly and briefly.

Examples:
1. User: "I cannot connect to Wi-Fi. I enabled it and forgot the saved network,
   but reconnecting failed." -> Help or ask a question. No prepare_ticket, and
   do not mention tickets.
2. User: "Check whether Wi-Fi is down. If it is operating, prepare a ticket.
   I enabled Wi-Fi and reconnected." -> Call check_service_status; if
   operational, call prepare_ticket with category wifi and steps_tried taken
   from the user's words ("Enabled Wi-Fi", "Reconnected").
3. User: "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi."
   -> Call save_language_preference, reply in Kazakh. No ticket."""


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

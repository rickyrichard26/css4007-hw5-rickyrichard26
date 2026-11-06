from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .data_loader import load_helpdesk


class HelpdeskState(TypedDict, total=False):
    message: str
    category: str   # "wifi" | "password" | "both" | "other"
    reply: str


HELPDESK = load_helpdesk()
ALIASES = HELPDESK["topic_aliases"]   # {"wifi": ["wifi", "wi-fi"], "password": ["password"]}


def _mentions(text: str, topic: str) -> bool:
    t = text.lower()
    return any(alias in t for alias in ALIASES[topic])


def _format_instructions(topic: str) -> str:
    item = HELPDESK["instructions"][topic]
    lines = [f"[{item['instruction_id']}] {item['title']}"]
    lines += [f"{i}. {step}" for i, step in enumerate(item["steps"], 1)]
    return "\n".join(lines)


# ---------- Node-node ----------
def read_request(state: HelpdeskState) -> dict:
    """Graph asli: aturan pertama menang (wifi), lalu password, lalu other."""
    msg = state["message"]
    if _mentions(msg, "wifi"):
        category = "wifi"
    elif _mentions(msg, "password"):
        category = "password"
    else:
        category = "other"
    return {"category": category}


def read_request_v2(state: HelpdeskState) -> dict:
    """Graph yang diubah: jika wifi DAN password disebut, kategori = both."""
    msg = state["message"]
    wifi = _mentions(msg, "wifi")
    pw = _mentions(msg, "password")
    if wifi and pw:
        category = "both"
    elif wifi:
        category = "wifi"
    elif pw:
        category = "password"
    else:
        category = "other"
    return {"category": category}


def wifi_help(state: HelpdeskState) -> dict:
    return {"reply": _format_instructions("wifi")}


def password_help(state: HelpdeskState) -> dict:
    return {"reply": _format_instructions("password")}


def other_help(state: HelpdeskState) -> dict:
    return {"reply": "Is your problem about Wi-Fi or a password?"}


def clarify_problem(state: HelpdeskState) -> dict:
    return {
        "reply": "Your message mentions both Wi-Fi and a password. "
                 "Which problem do you want to solve first: connecting to "
                 "Wi-Fi, or recovering your account password?"
    }


# ---------- Conditional edge ----------
def route(state: HelpdeskState) -> str:
    return {
        "wifi": "wifi_help",
        "password": "password_help",
        "both": "clarify_problem",
    }.get(state["category"], "other_help")


# ---------- Build ----------
def build_original_graph():
    g = StateGraph(HelpdeskState)
    g.add_node("read_request", read_request)
    g.add_node("wifi_help", wifi_help)
    g.add_node("password_help", password_help)
    g.add_node("other_help", other_help)
    g.add_edge(START, "read_request")
    g.add_conditional_edges(
        "read_request",
        route,
        {
            "wifi_help": "wifi_help",
            "password_help": "password_help",
            "other_help": "other_help",
        },
    )
    for n in ("wifi_help", "password_help", "other_help"):
        g.add_edge(n, END)
    return g.compile()


def build_clarify_graph():
    g = StateGraph(HelpdeskState)
    g.add_node("read_request", read_request_v2)
    g.add_node("wifi_help", wifi_help)
    g.add_node("password_help", password_help)
    g.add_node("other_help", other_help)
    g.add_node("clarify_problem", clarify_problem)
    g.add_edge(START, "read_request")
    g.add_conditional_edges(
        "read_request",
        route,
        {
            "wifi_help": "wifi_help",
            "password_help": "password_help",
            "other_help": "other_help",
            "clarify_problem": "clarify_problem",
        },
    )
    for n in ("wifi_help", "password_help", "other_help", "clarify_problem"):
        g.add_edge(n, END)
    return g.compile()
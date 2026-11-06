import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from helpdesk.part2_graph import MAX_STEPS, build_agent_graph, make_llm
from helpdesk.tools import build_tools

load_dotenv()


def serialize(msg) -> dict:
    d = {"type": msg.type, "content": msg.content}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = msg.tool_calls
    if msg.type == "tool":
        d["tool_name"] = msg.name
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    choices=["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
    ap.add_argument("--message", required=True)
    ap.add_argument("--user", default="demo-user-a")
    ap.add_argument("--thread", default="part2-demo")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    storage = os.getenv("HELPDESK_STORAGE_DIR", "local-storage")
    tools = build_tools(profile=args.profile, user_id=args.user,
                        thread_id=args.thread, storage_dir=storage)
    graph = build_agent_graph(tools, llm=make_llm())

    events, nodes, limit_reached = [], [], False
    init = {"messages": [HumanMessage(content=args.message)], "steps": 0}
    for update in graph.stream(init, stream_mode="updates"):
        for node, delta in update.items():
            nodes.append(node)
            for m in delta.get("messages", []):
                events.append({"node": node, **serialize(m)})
            limit_reached = limit_reached or bool(delta.get("limit_reached"))

    base_url = os.getenv("OPENAI_BASE_URL")
    log = {
        "status_profile": args.profile,
        "user_id": args.user,
        "thread_id": args.thread,
        "model": os.getenv("MODEL_NAME"),
        "provider": urlparse(base_url).netloc if base_url else "openai",
        "max_steps": MAX_STEPS,
        "limit_reached": limit_reached,
        "nodes_visited": nodes,
        "events": events,
    }
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Nodes:", " -> ".join(nodes))
    for e in events:
        if e.get("tool_calls"):
            print("TOOL REQUEST:", [(c["name"], c["args"]) for c in e["tool_calls"]])
        elif e["type"] == "tool":
            print("TOOL RESULT :", e["tool_name"], e["content"][:200])
    final = [e for e in events if e["type"] == "ai" and not e.get("tool_calls")]
    print("\nFINAL ANSWER:\n", final[-1]["content"] if final else "(none)")


if __name__ == "__main__":
    main()

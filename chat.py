import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from helpdesk.memory import UserMemory, build_memory_tool
from helpdesk.part2_graph import make_llm
from helpdesk.part3_graph import build_memory_graph
from helpdesk.tools import build_tools

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # safe printing of non-Latin text on Windows

SOURCES = {
    "find_instructions": "helpdesk reference file (data/helpdesk.json)",
    "check_service_status": "service status snapshot (data/service_status.json)",
    "prepare_ticket": "local ticket draft file (write)",
    "save_language_preference": "user memory (write)",
}


def serialize(msg) -> dict:
    d = {"type": msg.type, "content": msg.content}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = msg.tool_calls
    if msg.type == "tool":
        d["tool_name"] = msg.name
    return d


def run_turn(graph, config, args, memory, message):
    snapshot = graph.get_state(config)
    prior = (snapshot.values or {}).get("messages", [])   # conversation memory read
    prefs_before = memory.get(args.user)                   # user memory read

    init = {"messages": [HumanMessage(content=message)], "steps": 0, "limit_reached": False}
    events, nodes, limit_reached = [], [], False
    for update in graph.stream(init, config, stream_mode="updates"):
        for node, delta in update.items():
            nodes.append(node)
            for m in delta.get("messages", []):
                events.append({"node": node, **serialize(m)})
            limit_reached = limit_reached or bool(delta.get("limit_reached"))

    tool_requests, memory_writes, tickets = [], [], []
    for e in events:
        for c in e.get("tool_calls") or []:
            tool_requests.append({"tool": c["name"], "args": c["args"],
                                  "source": SOURCES.get(c["name"], "unknown")})
        if e["type"] == "tool":
            try:
                result = json.loads(e["content"])
            except Exception:
                continue
            if e["tool_name"] == "save_language_preference" and result.get("saved"):
                memory_writes.append({"reply_language": result["reply_language"]})
            if e["tool_name"] == "prepare_ticket" and "ticket" in result:
                tickets.append(result["saved_to"])

    ai = [e for e in events if e["type"] == "ai" and not e.get("tool_calls")]
    final = ai[-1]["content"].strip() if ai else "(none)"
    base_url = os.getenv("OPENAI_BASE_URL")

    entry = {
        "scenario": args.scenario,
        "process_id": os.getpid(),
        "user_id": args.user,
        "thread_id": args.thread,
        "checkpoint_key": config["configurable"]["thread_id"],
        "status_profile": args.profile,
        "model": os.getenv("MODEL_NAME"),
        "provider": urlparse(base_url).netloc if base_url else "openai",
        "user_message": message,
        "read_conversation_memory": {"earlier_messages_loaded": len(prior)},
        "read_user_memory": prefs_before,
        "nodes_visited": nodes,
        "tool_requests": tool_requests,
        "memory_writes": memory_writes,
        "tickets_saved": tickets,
        "user_memory_after": memory.get(args.user),
        "limit_reached": limit_reached,
        "final_answer": final,
    }
    if args.log:
        path = Path(args.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        entries.append(entry)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[pid {entry['process_id']}] user={args.user} thread={args.thread} "
          f"profile={args.profile}")
    print(f"  conversation memory: {len(prior)} earlier messages loaded")
    print(f"  user memory read   : {prefs_before or '(empty)'}")
    print(f"  nodes              : {' -> '.join(nodes)}")
    for t in tool_requests:
        print(f"  tool               : {t['tool']} {t['args']}  <- {t['source']}")
    if memory_writes:
        print(f"  user memory write  : {memory_writes}")
    print("\nASSISTANT:\n", final, "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--thread", required=True)
    ap.add_argument("--profile", default="normal",
                    choices=["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
    ap.add_argument("--message", help="one turn, then exit (new process each time)")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--log")
    ap.add_argument("--reset", action="store_true",
                    help="clear conversation memory and user memory before starting")
    args = ap.parse_args()

    storage = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))
    storage.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for name in ("conversations.sqlite", "conversations.sqlite-wal",
                     "conversations.sqlite-shm", "user_memory.json"):
            (storage / name).unlink(missing_ok=True)
        print("Memory reset: conversation memory and user memory cleared.")

    memory = UserMemory(str(storage))
    tools = build_tools(profile=args.profile, user_id=args.user,
                        thread_id=args.thread, storage_dir=str(storage))
    tools.append(build_memory_tool(memory, args.user))

    conn = sqlite3.connect(str(storage / "conversations.sqlite"), check_same_thread=False)
    graph = build_memory_graph(tools, memory, args.user, SqliteSaver(conn), llm=make_llm())
    # Checkpoint key combines user and thread, so a reused thread name cannot leak across users.
    config = {"configurable": {"thread_id": f"{args.user}:{args.thread}"}}

    try:
        if args.message:
            run_turn(graph, config, args, memory, args.message)
        else:
            print("Interactive mode. Empty line or /quit to exit.")
            while True:
                text = input("you> ").strip()
                if not text or text == "/quit":
                    break
                run_turn(graph, config, args, memory, text)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

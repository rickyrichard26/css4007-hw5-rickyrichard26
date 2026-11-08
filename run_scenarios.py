import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
STORAGE = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))


def reset_storage():
    """Clear conversation memory and user memory (tickets are kept)."""
    STORAGE.mkdir(parents=True, exist_ok=True)
    for name in ("conversations.sqlite", "conversations.sqlite-wal",
                 "conversations.sqlite-shm", "user_memory.json"):
        (STORAGE / name).unlink(missing_ok=True)


# ---------------- Part 1 scenarios (no model) ----------------
def run_part1(sc):
    from helpdesk.part1_graph import build_clarify_graph, build_original_graph

    variant = sc.get("graph_variant", "original")
    graph = build_clarify_graph() if variant == "with_clarification" else build_original_graph()
    entries = []
    for ev in sc["events"]:
        if ev["action"] != "message":
            continue
        state = {"message": ev["message"]}
        nodes, states = [], []
        for update in graph.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                nodes.append(node)
                state = {**state, **delta}
                states.append({"after_node": node, "state": dict(state)})
        helped = nodes[-1] in ("wifi_help", "password_help")
        entries.append({
            "scenario": sc["id"],
            "scenario_title": sc["title"],
            "scenario_part": sc["part"],
            "launch_command": f"python run_scenarios.py --only {sc['id']}",
            "graph_variant": variant,
            "status_profile": sc["status_profile"],
            "user_id": ev["user_id"],
            "thread_id": ev["thread_id"],
            "user_message": ev["message"],
            "nodes_visited": nodes,
            "states_after_node": states,
            "information_source": ("reference file (data/helpdesk.json)" if helped
                                   else "none (fixed rules in the graph)"),
            "final_response": state.get("reply"),
            "ticket_file": None,
            "loop_limit_reached": "not applicable (no model, no loop)",
        })
    return entries


# ---------------- Part 2/3 scenarios (agent via chat.py) ----------------
def split_segments(events):
    """Messages that share a process: split at restart_process or when user/thread changes."""
    segments, current, key = [], [], None
    for ev in events:
        if ev["action"] == "restart_process":
            if current:
                segments.append(current)
            current, key = [], None
            continue
        if ev["action"] != "message":
            continue
        k = (ev["user_id"], ev["thread_id"])
        if current and k != key:
            segments.append(current)
            current = []
        current.append(ev)
        key = k
    if current:
        segments.append(current)
    return segments


def read_log(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def run_agent(sc, log_path):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for seg in split_segments(sc["events"]):
        first = seg[0]
        cmd = [sys.executable, "chat.py", "--user", first["user_id"],
               "--thread", first["thread_id"], "--profile", sc["status_profile"],
               "--scenario", sc["id"], "--log", str(log_path)]
        shown = (f"python chat.py --user {first['user_id']} --thread {first['thread_id']} "
                 f"--profile {sc['status_profile']} --scenario {sc['id']} --log outputs/{sc['id']}.json")
        before = len(read_log(log_path))
        stdin_text = "\n".join(ev["message"] for ev in seg) + "\n\n"
        res = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                             encoding="utf-8", env=env, cwd=ROOT)
        if res.returncode != 0:
            print("  ERROR in chat.py:\n", res.stderr[-1500:])
        entries = read_log(log_path)
        for e in entries[before:]:
            e["launch_command"] = shown + "   (messages sent through stdin, one per turn)"
        log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    entries = read_log(log_path)
    (OUT / "tickets").mkdir(parents=True, exist_ok=True)
    for e in entries:
        e["scenario_title"] = sc["title"]
        e["scenario_part"] = sc["part"]
        copies = []
        for p in e.get("tickets_saved", []):
            src = Path(p)
            if src.exists():
                dst = OUT / "tickets" / src.name
                shutil.copyfile(src, dst)
                copies.append(f"outputs/tickets/{src.name}")
        e["ticket_copies"] = copies
    log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries


def short(text, n=220):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n].rstrip() + "..."


def summarize(entries):
    for e in entries:
        if e["scenario_part"] == 1:
            print(f"  [{e['graph_variant']}] {' -> '.join(e['nodes_visited'])}")
            print(f"  reply: {short(e['final_response'])}")
        else:
            tools = [t["tool"] for t in e["tool_requests"]]
            print(f"  pid {e['process_id']} user={e['user_id']} thread={e['thread_id']} "
                  f"| earlier msgs={e['read_conversation_memory']['earlier_messages_loaded']} "
                  f"| prefs read={e['read_user_memory']}")
            print(f"  tools={tools} | tickets={e.get('ticket_copies')} | limit={e['limit_reached']}")
            print(f"  answer: {short(e['final_answer'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario ids, for example S06 S07")
    args = ap.parse_args()

    data = json.loads(Path("data/scenarios.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    for sc in data["scenarios"]:
        if args.only and sc["id"] not in args.only:
            continue
        print(f"\n=== {sc['id']} {sc['title']} (profile {sc['status_profile']}) ===")
        log_path = OUT / f"{sc['id']}.json"
        log_path.unlink(missing_ok=True)
        if sc.get("reset_before"):
            reset_storage()
        if sc["part"] == 1:
            entries = run_part1(sc)
            log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        else:
            entries = run_agent(sc, log_path)
        summarize(entries)
        print(f"  log: outputs/{sc['id']}.json")


if __name__ == "__main__":
    main()

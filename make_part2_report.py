import json
from pathlib import Path

PROFILES = ["normal", "wifi_outage", "status_unavailable"]


def load(p):
    return json.loads(Path(f"outputs/part2_{p}.json").read_text(encoding="utf-8"))


def tool_calls(d):
    out = []
    for e in d["events"]:
        for c in e.get("tool_calls") or []:
            out.append((c["name"], c["args"]))
    return out


def final_answer(d):
    ai = [e for e in d["events"] if e["type"] == "ai" and not e.get("tool_calls")]
    return ai[-1]["content"].strip() if ai else "(none)"


def ticket_id(d):
    for e in d["events"]:
        if e["type"] == "tool" and e.get("tool_name") == "prepare_ticket":
            try:
                r = json.loads(e["content"])
            except Exception:
                continue
            if "ticket" in r:
                return r["ticket"]["ticket_id"]
    return None


def one_line(text, n=260):
    t = " ".join(text.split()).replace("|", "/")
    return t if len(t) <= n else t[:n].rstrip() + "..."


print("### Tabel hasil nyata (kolom Actual)\n")
print("| Status snapshot | Actual tool calls | Actual final behaviour |")
print("| --- | --- | --- |")
for p in PROFILES:
    d = load(p)
    calls = " -> ".join(f"{n}({json.dumps(a, ensure_ascii=False)})" for n, a in tool_calls(d)) or "(no tool calls)"
    tid = ticket_id(d)
    ticket = f"Ticket draft {tid} created. " if tid else "No ticket created. "
    limit = "Step limit reached. " if d["limit_reached"] else ""
    print(f"| `{p}` | {one_line(calls, 300)} | {ticket}{limit}{one_line(final_answer(d))} |")

d = load("normal")
print("\n### Blok run nyata (normal)\n")
print(f"Model: {d['model']} via {d['provider']}; max_steps={d['max_steps']}; limit_reached={d['limit_reached']}\n")
print("```")
print("Nodes:", " -> ".join(d["nodes_visited"]))
for e in d["events"]:
    if e.get("tool_calls"):
        print("TOOL REQUEST:", [(c["name"], c["args"]) for c in e["tool_calls"]])
    elif e["type"] == "tool":
        print("TOOL RESULT :", e["tool_name"], e["content"][:300])
print("FINAL ANSWER:", final_answer(d))
print("```")

print("\n### Perbedaan antar profil\n")
for p in PROFILES:
    d = load(p)
    names = [n for n, _ in tool_calls(d)]
    print(f"- {p}: tools = {names}, ticket = {ticket_id(d) or 'none'}")

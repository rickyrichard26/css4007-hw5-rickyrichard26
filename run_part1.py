import json
from pathlib import Path

from helpdesk.part1_graph import build_original_graph, build_clarify_graph

MESSAGES = [
    "I cannot connect to Wi-Fi.",
    "I forgot my PASSWORD.",
    "My laptop screen is broken.",
    "My Wi-Fi password does not work.",
]


def run(graph, message, label):
    """Jalankan graph dan catat node yang dikunjungi + state setelah tiap node."""
    state = {"message": message}
    nodes = []
    states_after_node = []
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_update in update.items():
            nodes.append(node_name)
            state = {**state, **node_update}
            states_after_node.append({"after_node": node_name, "state": dict(state)})
    return {
        "graph": label,
        "message": message,
        "nodes_visited": nodes,
        "states_after_node": states_after_node,
    }


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    results = []
    for label, graph in [
        ("original", build_original_graph()),
        ("with_clarify", build_clarify_graph()),
    ]:
        for msg in MESSAGES:
            results.append(run(graph, msg, label))

    Path("outputs/part1_traces.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for r in results:
        print(f"[{r['graph']}] {r['message']}")
        print("   ", " -> ".join(r["nodes_visited"]))

# HW5 — My LangGraph helpdesk

## Part 1 — How the graph works

### My graph and state

The state has three fields:

| Field | What it holds |
| --- | --- |
| `message` | The user's original message |
| `category` | `wifi`, `password`, `other` (or `both` in the changed graph), set by `read_request` |
| `reply` | The text produced by a help node |

Graph generated from my code with `get_graph().draw_mermaid()`. Solid arrows are fixed edges. Dotted arrows are the conditional edge chosen by my `route` function.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        read_request(read_request)
        wifi_help(wifi_help)
        password_help(password_help)
        other_help(other_help)
        __end__([<p>__end__</p>]):::last
        __start__ --> read_request;
        read_request -.-> other_help;
        read_request -.-> password_help;
        read_request -.-> wifi_help;
        other_help --> __end__;
        password_help --> __end__;
        wifi_help --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
What each node reads and changes:

| Node | Reads | Changes |
| --- | --- | --- |
| `read_request` | `message` | `category` |
| `wifi_help` | nothing from the state | `reply` (Wi-Fi steps from `helpdesk.json`) |
| `password_help` | nothing from the state | `reply` (password steps from `helpdesk.json`) |
| `other_help` | nothing from the state | `reply` (asks whether the problem is Wi-Fi or a password) |
| `route` (conditional edge) | `category` | nothing; it only returns the name of the next node |

**Answers to the four questions**

1. **What information does each node read, and what does it change?** See the table above. Only `read_request` reads the user's message, and it writes `category`. Each help node writes `reply`.
2. **Node update vs. conditional edge.** A node returns a partial update (a dictionary) that LangGraph merges into the state. A conditional edge does not change the state. It reads the state and returns the name of the next node.
3. **What does the next node receive?** The current graph state, with all earlier updates already merged in. It does not receive only the previous node's update.
4. **What do `START`, `END` and compiling mean?** `START` and `END` are virtual entry and exit points of the graph. `compile()` checks that the graph structure is valid and returns an object that can be run.
   
### Predictions before running
I wrote these predictions from the routing rules in the brief (Wi-Fi is checked first, then password, then everything else; case is ignored). I then compared them with the real runs saved in `outputs/part1_traces.json`.

| Message | Predicted route | Actual route | What I learned |
| --- | --- | --- | --- |
| I cannot connect to Wi-Fi. | read_request → wifi_help | read_request → wifi_help | "Wi-Fi" matches the first rule. |
| I forgot my PASSWORD. | read_request → password_help | read_request → password_help | Upper case does not matter because the message is lowercased before matching. |
| My laptop screen is broken. | read_request → other_help | read_request → other_help | No keyword matches, so the fallback node asks which topic the user means. |
| My Wi-Fi password does not work. | read_request → wifi_help | read_request → wifi_help | The message mentions both topics, but the Wi-Fi rule is checked first, so the password part is ignored. |

State after each node for the last message in the original graph (from `outputs/part1_traces.json`):

```
after read_request:
  {"message": "My Wi-Fi password does not work.", "category": "wifi"}
after wifi_help:
  {"message": "My Wi-Fi password does not work.", "category": "wifi",
   "reply": "[GUIDE-WIFI] Connect to the practice campus network\n1. Check that Wi-Fi is enabled and airplane mode is off.\n..."}
```

**What happens when a message mentions both Wi-Fi and a password.** In the original graph `read_request` uses `if / elif`. For "My Wi-Fi password does not work." the Wi-Fi condition is true, so the password condition is never checked. The user gets Wi-Fi instructions only, even though the message also mentions a password. The first matching rule causes this.

### My change to the graph

I added a node `clarify_problem` and a second classifier `read_request_v2`. When a message mentions both Wi-Fi and a password, `category` becomes `both`, and the router sends the request to `clarify_problem`. That node asks which problem the user wants to solve first. The other three messages keep the same routes.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        read_request(read_request)
        wifi_help(wifi_help)
        password_help(password_help)
        other_help(other_help)
        clarify_problem(clarify_problem)
        __end__([<p>__end__</p>]):::last
        __start__ --> read_request;
        read_request -.-> clarify_problem;
        read_request -.-> other_help;
        read_request -.-> password_help;
        read_request -.-> wifi_help;
        clarify_problem --> __end__;
        other_help --> __end__;
        password_help --> __end__;
        wifi_help --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

New execution trace for "My Wi-Fi password does not work.":

```
read_request -> clarify_problem

after read_request:
  {"message": "My Wi-Fi password does not work.", "category": "both"}
after clarify_problem:
  {"message": "My Wi-Fi password does not work.", "category": "both",
   "reply": "Your message mentions both Wi-Fi and a password. Which problem do you want to solve first: connecting to Wi-Fi, or recovering your account password?"}
```

Before the change the route was `read_request -> wifi_help`. After the change it is `read_request -> clarify_problem`, so the user is asked instead of receiving only half of the answer.

**What LangGraph controls, and what my functions decide.** LangGraph runs the nodes in order, merges each node's update into the shared state, and follows the edges, including the edge picked by the router. My Python functions decide how a message is classified (`read_request`, `read_request_v2`), which node name the router returns (`route`), and what each reply says. There is no model in Part 1, so no model makes any decision.

## Part 2 — Model and tools

TODO: draw the model/tool loop. Answer the four questions in the brief. Record the model and provider, direct calls to each tool, and one actual run that uses multiple tools. Explain the loop limit.

## Part 3 — Memory

TODO: explain conversation state, thread_id, user identifier, user preference storage, and retrieval from helpdesk.json.

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | | | |
| Close, restart and resume | | | |
| Same user, new conversation | | | |
| Different user | | | |

## Part 4 — Running and checking my work

TODO: point to the README setup steps, package versions, memory-reset instructions, tests and GitHub Actions workflow. Explain which mistake each test catches and which runs use a fake model.

## Choices and limitations

TODO: explain the storage and libraries I chose, why they fit this task, and what still does not work reliably.

## Supplied scenarios — my actual results

| Scenario | Command or manual steps | Log file | What happened / what I changed |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | | | |
| S02 — Case-insensitive password route | | | |
| S03 — Unsupported request | | | |
| S04 — Two topics before the graph change | | | |
| S05 — Two topics after adding clarification | | | |
| S06 — Instructions then local ticket | | | |
| S07 — The same request during a Wi-Fi outage | | | |
| S08 — Status source unavailable | | | |
| S09 — Missing ticket details | | | |
| S10 — Use the current conversation | | | |
| S11 — Persist and recall after restart | | | |
| S12 — Preference across threads without replay | | | |
| S13 — Do not share memory between users | | | |
| S14 — Password recovery outage | | | |

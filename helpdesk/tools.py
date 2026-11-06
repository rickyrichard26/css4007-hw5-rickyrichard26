import json
import re
import uuid
from pathlib import Path

from langchain_core.tools import tool

from .data_loader import DATA_DIR, load_helpdesk

HELPDESK = load_helpdesk()
TOPIC_SERVICES = HELPDESK["topic_services"]   # {"wifi": "wifi", "password": "account_recovery"}
TOPIC_ALIASES = HELPDESK["topic_aliases"]

SECRET_RE = re.compile(
    r"(password|passcode|api[ _-]?key|recovery code)\s*(is|=|:)\s*\S+", re.IGNORECASE
)


def _load_json(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def normalize_topic(text: str):
    """Ubah teks bebas ('Wi-Fi', 'password', 'account_recovery') menjadi 'wifi'/'password'."""
    t = text.strip().lower().replace("_", " ")
    for topic, aliases in TOPIC_ALIASES.items():
        if t == topic or any(a in t for a in aliases):
            return topic
    if "recovery" in t:
        return "password"
    return None


# ---------- Fungsi inti ----------
def find_instructions_impl(topic: str) -> dict:
    key = normalize_topic(topic)
    if key is None:
        return {"error": "unknown_topic", "supported_topics": list(TOPIC_ALIASES)}
    item = HELPDESK["instructions"][key]
    return {
        "topic": key,
        "instruction_id": item["instruction_id"],
        "title": item["title"],
        "steps": item["steps"],
    }


def check_service_status_impl(service: str, profile: str) -> dict:
    key = normalize_topic(service)
    if key is None:
        return {"error": "unknown_service", "supported_topics": list(TOPIC_ALIASES)}
    service_name = TOPIC_SERVICES[key]
    entry = _load_json("service_status.json")["profiles"][profile][service_name]
    # status: "operational" | "outage" | "unknown"
    return {
        "service": service_name,
        "status": entry["status"],
        "message": entry["message"],
        "status_profile": profile,
    }


def prepare_ticket_impl(
    category: str,
    summary: str,
    steps_tried: list,
    *,
    profile: str,
    user_id: str,
    thread_id: str,
    storage_dir: str,
) -> dict:
    key = normalize_topic(category)
    steps = [s.strip() for s in (steps_tried or []) if isinstance(s, str) and s.strip()]
    missing = []
    if key is None:
        missing.append("category (wifi or password)")
    if not summary or not summary.strip():
        missing.append("problem description")
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {"error": "missing_information", "missing": missing,
                "note": "Ask the user for these details; do not invent them."}

    if SECRET_RE.search(summary) or any(SECRET_RE.search(s) for s in steps):
        return {"error": "secret_information",
                "note": "Do not include passwords, API keys or recovery codes in a ticket."}

    status = check_service_status_impl(category, profile)
    ticket = _load_json("ticket_template.json")["ticket"]
    ticket_id = "HD-" + uuid.uuid4().hex[:8].upper()
    ticket.update({
        "ticket_id": ticket_id,
        "status": "draft",
        "user_id": user_id,
        "thread_id": thread_id,
        "category": key,
        "summary": summary.strip(),
        "steps_tried": steps,
        "service_status": status["status"],
        "status_profile": profile,
        "reference_ids": [HELPDESK["instructions"][key]["instruction_id"]],
    })

    out_dir = Path(storage_dir) / "tickets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticket_id}.json"
    path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ticket": ticket, "saved_to": str(path)}


# ---------- Pembungkus tool untuk model ----------
def build_tools(*, profile: str, user_id: str, thread_id: str, storage_dir: str):
    """Profil status, user_id dan thread_id ditentukan aplikasi, bukan model."""

    @tool
    def find_instructions(topic: str) -> dict:
        """Look up the official helpdesk instructions for a topic.
        topic must be 'wifi' or 'password'. Returns instruction_id, title and steps."""
        return find_instructions_impl(topic)

    @tool
    def check_service_status(service: str) -> dict:
        """Check whether a service is operational, in an outage, or unknown.
        service is 'wifi' or 'password' (password uses the account recovery service).
        An 'unknown' status means the status source is unavailable."""
        return check_service_status_impl(service, profile)

    @tool
    def prepare_ticket(category: str, summary: str, steps_tried: list[str]) -> dict:
        """Prepare a LOCAL draft support ticket (nothing is sent anywhere).
        category is 'wifi' or 'password'. summary describes the problem.
        steps_tried lists what the user already tried. All three are required;
        if the user has not given them, ask the user instead of guessing."""
        return prepare_ticket_impl(
            category, summary, steps_tried,
            profile=profile, user_id=user_id, thread_id=thread_id,
            storage_dir=storage_dir,
        )

    return [find_instructions, check_service_status, prepare_ticket]

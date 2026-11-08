import json
import re
from pathlib import Path

from langchain_core.tools import tool

LANGUAGE_RE = re.compile(r"^[A-Za-z][A-Za-z \-]{0,38}$")


class UserMemory:
    """Persistent per-user preferences, stored in one JSON file kept out of git."""

    def __init__(self, storage_dir: str):
        self.path = Path(storage_dir) / "user_memory.json"

    def _read(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def get(self, user_id: str) -> dict:
        return dict(self._read().get(user_id, {}))

    def set(self, user_id: str, key: str, value: str) -> None:
        data = self._read()
        data.setdefault(user_id, {})[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def build_memory_tool(memory: UserMemory, user_id: str):
    """user_id is fixed by the application; the model cannot choose whose data to write."""

    @tool
    def save_language_preference(language: str) -> dict:
        """Save the user's preferred language for replies, for future conversations.
        Call this only when the user explicitly asks to be answered in a language
        (for example 'reply in Kazakh from now on'). language is a plain language name."""
        lang = (language or "").strip()
        if not LANGUAGE_RE.match(lang):
            return {"error": "invalid_language",
                    "note": "Give a plain language name such as 'Kazakh'."}
        memory.set(user_id, "reply_language", lang)
        return {"saved": True, "reply_language": lang}

    return save_language_preference

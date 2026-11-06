import json
import os
from pathlib import Path

from dotenv import load_dotenv

from helpdesk.tools import build_tools

load_dotenv()
STORAGE = os.getenv("HELPDESK_STORAGE_DIR", "local-storage")

tools = {t.name: t for t in build_tools(
    profile="normal", user_id="demo-user-a", thread_id="direct-demo", storage_dir=STORAGE)}

results = {
    "find_instructions(wifi)": tools["find_instructions"].invoke({"topic": "wifi"}),
    "find_instructions(unknown)": tools["find_instructions"].invoke({"topic": "printer"}),
    "check_service_status(wifi, normal)": tools["check_service_status"].invoke({"service": "wifi"}),
    "prepare_ticket(complete)": tools["prepare_ticket"].invoke({
        "category": "wifi",
        "summary": "Cannot connect to Campus-Demo Wi-Fi.",
        "steps_tried": ["Turned Wi-Fi off and on", "Forgot the network and reconnected"],
    }),
    "prepare_ticket(missing steps)": tools["prepare_ticket"].invoke({
        "category": "wifi", "summary": "Cannot connect to Wi-Fi.", "steps_tried": [],
    }),
}

Path("outputs").mkdir(exist_ok=True)
Path("outputs/part2_tools_direct.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))

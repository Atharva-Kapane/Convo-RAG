import json
import os
from collections import defaultdict
from tqdm import tqdm

from models.llm import llm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MESSAGES_PATH = os.path.join(BASE_DIR, "processed_data", "messages_with_conv.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "personas.json")

MAX_CONVS = 20   # test first (increase later)
MAX_CHARS = 2500


# -----------------------------
# LOAD
# -----------------------------
def load_messages():
    with open(MESSAGES_PATH, "r") as f:
        return json.load(f)


# -----------------------------
# GROUP BY CONVERSATION
# -----------------------------
def group_conversations(messages):
    convs = defaultdict(list)

    for m in messages:
        convs[m["conv_id"]].append(m)

    return convs


# -----------------------------
# BUILD CLEAN CONVERSATION TEXT
# -----------------------------
def build_conversation_text(conv_msgs):
    selected = []

    low_info = {"ok", "okay", "nice", "cool", "lol", "haha", "thanks"}

    for m in conv_msgs:
        text = m["text"].strip()

        if len(text) < 5:
            continue

        if text.lower() in low_info:
            continue

        selected.append(f"{m['speaker']}: {text}")

    full_text = "\n".join(selected)

    return full_text[:MAX_CHARS]


# -----------------------------
# PROMPT
# -----------------------------
def build_prompt(conversation_text):
    return f"""
You are a strict information extraction system.

TASK:
Extract factual personas for User 1 and User 2 from the conversation.

RULES:
- ONLY use explicitly stated facts
- NO guessing or assumptions
- If no data → return empty list or empty string
- DO NOT explain anything
- DO NOT add text outside JSON
- OUTPUT MUST BE VALID JSON

CONVERSATION:
{conversation_text}

RETURN EXACTLY THIS JSON:

{{
  "user1": {{
    "habits": [],
    "personal_facts": [],
    "personality_traits": [],
    "communication_style": {{
      "tone": "",
      "length": "",
      "style": ""
    }}
  }},
  "user2": {{
    "habits": [],
    "personal_facts": [],
    "personality_traits": [],
    "communication_style": {{
      "tone": "",
      "length": "",
      "style": ""
    }}
  }}
}}
"""


# -----------------------------
# LLM CALL
# -----------------------------
def extract_persona(conv_text):
    try:
        prompt = build_prompt(conv_text)

        response = llm.generate_json(prompt)

        return json.loads(response)

    except Exception as e:
        print("❌ JSON ERROR:", e)
        return None


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading messages...")
    messages = load_messages()

    print("Grouping conversations...")
    convs = group_conversations(messages)

    personas = []

    print(f"Extracting personas (first {MAX_CONVS} convs)...")

    for i, (conv_id, conv_msgs) in enumerate(tqdm(convs.items())):
        if i >= MAX_CONVS:
            break

        conv_text = build_conversation_text(conv_msgs)

        if not conv_text.strip():
            personas.append({
                "conv_id": conv_id,
                "persona": None
            })
            continue

        persona = extract_persona(conv_text)

        if persona is None:
            print(f"⚠️ Failed conv_id: {conv_id}")
            print(conv_text[:300])

        personas.append({
            "conv_id": conv_id,
            "persona": persona
        })

    print("Saving personas.json...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(personas, f, indent=2)

    print("DONE")


if __name__ == "__main__":
    main()
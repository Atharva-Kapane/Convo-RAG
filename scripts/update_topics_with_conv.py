import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MESSAGES_PATH = os.path.join(BASE_DIR, "processed_data", "messages_with_conv.json")
TOPICS_PATH = os.path.join(BASE_DIR, "processed_data", "topics.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "topics_with_conv.json")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    messages = load_json(MESSAGES_PATH)
    topics = load_json(TOPICS_PATH)

    msg_to_conv = {m["msg_id"]: m["conv_id"] for m in messages}

    for t in topics:
        start = t["start_msg"]
        t["conv_id"] = msg_to_conv.get(start, None)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(topics, f, indent=2)

    print("topics_with_conv.json created.")


if __name__ == "__main__":
    main()
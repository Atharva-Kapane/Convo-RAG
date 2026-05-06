import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "conversations.csv")
MESSAGES_PATH = os.path.join(BASE_DIR, "processed_data", "messages.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data", "messages_with_conv.json")


def load_messages():
    with open(MESSAGES_PATH, "r") as f:
        return json.load(f)


def parse_csv_lengths():
    df = pd.read_csv(CSV_PATH)

    lengths = []

    for i, row in df.iterrows():
        text = str(row.iloc[0])

        msgs = [line.strip() for line in text.split("\n") if line.strip()]
        lengths.append(len(msgs))

    return lengths


def assign_conv_ids(messages, conv_lengths):
    msg_idx = 0
    conv_id = 1

    for length in conv_lengths:
        for _ in range(length):
            if msg_idx >= len(messages):
                break

            messages[msg_idx]["conv_id"] = conv_id
            msg_idx += 1

        conv_id += 1

    return messages


def main():
    print("Loading...")
    messages = load_messages()

    print("Reading CSV structure...")
    conv_lengths = parse_csv_lengths()

    print("Assigning conv_ids...")
    messages = assign_conv_ids(messages, conv_lengths)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(messages, f, indent=2)

    print("messages_with_conv.json created.")


if __name__ == "__main__":
    main()
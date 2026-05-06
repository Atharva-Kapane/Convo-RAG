import json
from tqdm import tqdm

from preprocessing.parser import load_dataset
from preprocessing.embedder import Embedder
from preprocessing.topic_detector import detect_topics
from preprocessing.summarizer import summarize_text
from preprocessing.hundred_summarizer import generate_100_summaries

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "conversations.csv")
OUTPUT_MESSAGES = os.path.join(BASE_DIR, "processed_data", "messages.json")
OUTPUT_TOPICS = os.path.join(BASE_DIR, "processed_data", "topics.json")
OUTPUT_100 = os.path.join(BASE_DIR, "processed_data", "summaries_100.json")

MIN_SUMMARY_SIZE = 2


def main():
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    print("Loading dataset...")
    messages = load_dataset(CSV_PATH)
    messages = messages[:50000]
    print(f"Total messages: {len(messages)}")

    # -----------------------------
    # EMBEDDINGS
    # -----------------------------
    print("Generating embeddings...")
    embedder = Embedder()

    texts = [m["text"] for m in messages]
    embeddings = embedder.embed_texts(texts)

    for i, emb in enumerate(embeddings):
        messages[i]["embedding"] = emb.tolist()

    print("Saving messages.json...")
    with open(OUTPUT_MESSAGES, "w") as f:
        json.dump(messages, f, indent=2)

    # -----------------------------
    # TOPIC DETECTION
    # -----------------------------
    print("Detecting topics...")
    topics = detect_topics(messages)
    topics = topics[:500]
    print(f"Total topics detected: {len(topics)}")

    # -----------------------------
    # TOPIC SUMMARIES
    # -----------------------------
    print("Generating summaries...")

    topics_output = []

    for t in tqdm(topics):
        topic_messages = t["messages"]

        topic_text = " ".join([m["text"] for m in topic_messages])

        if len(topic_messages) < MIN_SUMMARY_SIZE:
            summary = topic_text[:300].rsplit(" ", 1)[0]
        else:
            summary = summarize_text(topic_text)

        topics_output.append({
            "topic_id": t["topic_id"],
            "start_msg": topic_messages[0]["msg_id"],
            "end_msg": topic_messages[-1]["msg_id"],
            "summary": summary,
            "text": topic_text[:500]
        })

    print("Saving topics.json...")
    with open(OUTPUT_TOPICS, "w") as f:
        json.dump(topics_output, f, indent=2)

    # -----------------------------
    # 100 MESSAGES SUMMARIES
    # -----------------------------
    print("Generating 100-message summaries...")
    hundred_summaries = generate_100_summaries(messages[:2000])
    with open(OUTPUT_100, "w") as f:
        json.dump(hundred_summaries, f, indent=2)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
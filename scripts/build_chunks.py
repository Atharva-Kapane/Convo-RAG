import json
import os
import faiss

from preprocessing.chunker import create_chunks
from preprocessing.embedder import Embedder
from preprocessing.indexer import build_index

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MESSAGES_PATH = os.path.join(BASE_DIR, "processed_data", "messages_with_conv.json")
TOPICS_PATH = os.path.join(BASE_DIR, "processed_data", "topics_with_conv.json")

CHUNKS_PATH = os.path.join(BASE_DIR, "processed_data", "chunks.json")
CHUNK_MAP_PATH = os.path.join(BASE_DIR, "processed_data", "chunk_map.json")

CHUNK_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "chunks.index")
TOPIC_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "topics.index")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    print("Loading data...")
    messages = load_json(MESSAGES_PATH)
    topics = load_json(TOPICS_PATH)

    embedder = Embedder()

    # -----------------------------
    # CHUNKING
    # -----------------------------
    print("Creating chunks...")
    chunks = create_chunks(messages, topics)

    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Total chunks: {len(chunks)}")

    # -----------------------------
    # CHUNK INDEX
    # -----------------------------
    print("Building chunk index...")

    chunk_texts = [c["text"] for c in chunks]
    chunk_index, _ = build_index(chunk_texts, embedder)

    faiss.write_index(chunk_index, CHUNK_INDEX_PATH)

    # IMPORTANT → mapping
    chunk_map = {i: chunks[i] for i in range(len(chunks))}

    with open(CHUNK_MAP_PATH, "w") as f:
        json.dump(chunk_map, f, indent=2)

    # -----------------------------
    # TOPIC INDEX
    # -----------------------------
    print("Building topic index...")

    topic_texts = [t["summary"] for t in topics]
    topic_index, _ = build_index(topic_texts, embedder)

    faiss.write_index(topic_index, TOPIC_INDEX_PATH)

    print("Chunking + Indexing complete.")


if __name__ == "__main__":
    main()
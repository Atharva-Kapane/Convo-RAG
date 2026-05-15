import json
import os
import faiss
import numpy as np

from preprocessing.embedder import Embedder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH = os.path.join(BASE_DIR, "processed_data", "chunks.json")
TOPICS_PATH = os.path.join(BASE_DIR, "processed_data", "topics_with_conv.json")

CHUNK_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "chunks.index")
TOPIC_INDEX_PATH = os.path.join(BASE_DIR, "processed_data", "topics.index")


class Retriever:
    def __init__(self):
        self.embedder = Embedder()

        # load data
        with open(CHUNKS_PATH) as f:
            self.chunks = json.load(f)

        with open(TOPICS_PATH) as f:
            self.topics = json.load(f)

        # load indexes
        self.chunk_index = faiss.read_index(CHUNK_INDEX_PATH)
        self.topic_index = faiss.read_index(TOPIC_INDEX_PATH)

    # -----------------------------
    # cosine similarity
    # -----------------------------
    def cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # -----------------------------
    # retrieve + rerank chunks
    # -----------------------------
    def retrieve_chunks(self, query, top_k=10):
        q_emb = self.embedder.embed_texts([query])[0].astype("float32")

        D, I = self.chunk_index.search(np.array([q_emb]), top_k * 3)

        candidates = [self.chunks[i] for i in I[0]]

        # rerank
        scored = []
        for c in candidates:
            c_emb = self.embedder.embed_texts([c["text"]])[0]
            sim = self.cosine(q_emb, c_emb)
            scored.append((sim, c))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [c for _, c in scored[:top_k]]

    # -----------------------------
    # retrieve + rerank topics
    # -----------------------------
    def retrieve_topics(self, query, top_k=5):
        q_emb = self.embedder.embed_texts([query])[0].astype("float32")

        D, I = self.topic_index.search(np.array([q_emb]), top_k * 3)

        candidates = [self.topics[i] for i in I[0]]

        # rerank
        scored = []
        for t in candidates:
            t_emb = self.embedder.embed_texts([t["summary"]])[0]
            sim = self.cosine(q_emb, t_emb)
            scored.append((sim, t))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [t for _, t in scored[:top_k]]
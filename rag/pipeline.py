import json
import os

from rag.retriever import Retriever
from rag.router import is_persona_query
from rag.prompt_builder import build_prompt
from rag.conflict_resolver import ConflictResolver

from models.gemini import gemini  # use singleton


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSONA_PATH = os.path.join(BASE_DIR, "processed_data", "personas.json")


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.conflict_resolver = ConflictResolver()

        # load personas once
        with open(PERSONA_PATH, "r") as f:
            self.personas = json.load(f)

    # -----------------------------
    # PERSONA FETCH
    # -----------------------------
    def get_persona(self, conv_ids):
        result = []

        conv_set = set(conv_ids)

        for p in self.personas:
            if p["conv_id"] in conv_set and p["persona"] is not None:
                result.append(p["persona"])

        return result if result else None

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run(self, query):
        # -------------------------
        # RETRIEVE
        # -------------------------
        chunks = self.retriever.retrieve_chunks(query, top_k=8)
        topics = self.retriever.retrieve_topics(query, top_k=5)

        # safety filter
        chunks = [c for c in chunks if c["text"].strip()]
        topics = [t for t in topics if t["summary"].strip()]

        # --------------------------------
        # conflict resolution
        # --------------------------------
        ranked_chunks = self.conflict_resolver.rerank_chunks(chunks)

        contradictions = self.conflict_resolver.detect_contradictions(
        ranked_chunks
        )

        # -------------------------
        # PERSONA ROUTING
        # -------------------------
        personas = None

        if is_persona_query(query):
            conv_ids = list(set([c["conv_id"] for c in chunks]))
            personas = self.get_persona(conv_ids)

        # -------------------------
        # PROMPT BUILD
        # -------------------------
        prompt = build_prompt(
            query=query,
            chunks=ranked_chunks,
            topics=topics,
            personas=personas,
            contradictions=contradictions
        )

        # -------------------------
        # LLM CALL
        # -------------------------
        response = gemini.generate(prompt)

        if not response:
            return "Error: No response from model."

        return response
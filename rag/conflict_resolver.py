import re
import numpy as np


class ConflictResolver:

    def __init__(self):
        self.positive_words = {
            "love", "helped", "support", "happy",
            "great", "close", "care", "enjoy"
        }

        self.negative_words = {
            "fight", "angry", "upset", "hate",
            "frustrated", "argument", "cry", "sad"
        }

    # ---------------------------------------------------
    # emotional score
    # ---------------------------------------------------
    def emotional_weight(self, text):
        text = text.lower()

        pos = sum(word in text for word in self.positive_words)
        neg = sum(word in text for word in self.negative_words)

        return pos + neg

    # ---------------------------------------------------
    # recency score
    # higher conv_id = more recent
    # ---------------------------------------------------
    def recency_score(self, conv_id, max_conv):
        if max_conv == 0:
            return 0

        return conv_id / max_conv

    # ---------------------------------------------------
    # contradiction detection
    # ---------------------------------------------------
    def detect_contradictions(self, chunks):

        contradictions = []

        positive_chunks = []
        negative_chunks = []

        for c in chunks:
            text = c["text"].lower()

            pos = any(w in text for w in self.positive_words)
            neg = any(w in text for w in self.negative_words)

            if pos:
                positive_chunks.append(c)

            if neg:
                negative_chunks.append(c)

        if positive_chunks and negative_chunks:

            contradictions.append({
                "type": "emotional_conflict",
                "reason": "Conversation sentiment changes across memories",
                "positive_examples": [
                    c["text"][:120]
                    for c in positive_chunks[:2]
                ],
                "negative_examples": [
                    c["text"][:120]
                    for c in negative_chunks[:2]
                ]
            })

        return contradictions

    # ---------------------------------------------------
    # rerank
    # ---------------------------------------------------
    def rerank_chunks(self, chunks):

        if not chunks:
            return []

        max_conv = max(c["conv_id"] for c in chunks)

        scored = []

        for c in chunks:

            emotion = self.emotional_weight(c["text"])
            recency = self.recency_score(c["conv_id"], max_conv)

            final_score = (
                emotion * 0.4 +
                recency * 0.6
            )

            c["emotion_score"] = round(emotion, 3)
            c["recency_score"] = round(recency, 3)
            c["final_score"] = round(final_score, 3)

            scored.append(c)

        scored.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return scored
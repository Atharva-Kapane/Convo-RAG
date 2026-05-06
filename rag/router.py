from preprocessing.embedder import Embedder
import numpy as np

embedder = Embedder()

persona_anchors = [
    "who is this person",
    "tell me about their personality",
    "what kind of person are they",
    "describe the speaker",
    "what are their traits"
]

persona_vecs = embedder.embed_texts(persona_anchors)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_persona_query(query):
    q_vec = embedder.embed_texts([query])[0]

    sims = [cosine(q_vec, p) for p in persona_vecs]

    return max(sims) > 0.6
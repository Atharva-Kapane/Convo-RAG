import numpy as np

PERSONA_ANCHORS = [
    "what kind of person is this user",
    "describe their personality",
    "what are their habits",
    "how do they communicate"
]

CONVO_ANCHORS = [
    "what did they talk about",
    "summarize the conversation",
    "what happened",
    "what was discussed"
]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_anchor_vectors(embedder):
    persona_vec = np.mean(embedder.embed_texts(PERSONA_ANCHORS), axis=0)
    convo_vec = np.mean(embedder.embed_texts(CONVO_ANCHORS), axis=0)
    return persona_vec, convo_vec


def route_query(query, embedder, persona_vec, convo_vec):
    q_vec = embedder.embed_single(query)

    sim_persona = cosine(q_vec, persona_vec)
    sim_convo = cosine(q_vec, convo_vec)

    return sim_persona > 0.6
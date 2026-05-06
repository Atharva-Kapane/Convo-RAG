import numpy as np
from tqdm import tqdm

SIM_THRESHOLD = 0.55
MIN_TOPIC_SIZE = 5
WINDOW = 5


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_window_mean(embeddings, start, end):
    window = embeddings[start:end]
    return np.mean(window, axis=0)


def detect_topics(messages):
    embeddings = [np.array(m["embedding"]) for m in messages]

    topics = []
    current_topic = []
    topic_id = 1

    for i in tqdm(range(len(messages))):
        current_topic.append(messages[i])

        # skip early indices
        if i < WINDOW or i + WINDOW >= len(messages):
            continue

        prev_mean = get_window_mean(embeddings, i - WINDOW, i)
        next_mean = get_window_mean(embeddings, i, i + WINDOW)

        sim = cosine(prev_mean, next_mean)

        # topic break condition
        if sim < SIM_THRESHOLD and len(current_topic) >= MIN_TOPIC_SIZE:
            topics.append({
                "topic_id": topic_id,
                "messages": current_topic.copy()
            })

            topic_id += 1
            current_topic = []

    # last topic
    if current_topic:
        topics.append({
            "topic_id": topic_id,
            "messages": current_topic
        })

    return topics
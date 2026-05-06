CHUNK_SIZE = 6
OVERLAP = 2
MIN_MESSAGES = 3   # 🔥 new


def create_chunks(messages, topics):
    chunks = []
    chunk_id = 0

    msg_map = {m["msg_id"]: m for m in messages}

    for topic in topics:
        start = topic["start_msg"]
        end = topic["end_msg"]

        topic_msgs = [msg_map[i] for i in range(start, end + 1) if i in msg_map]

        i = 0
        while i < len(topic_msgs):
            window = topic_msgs[i:i + CHUNK_SIZE]

            # 🔥 SKIP SMALL CHUNKS
            if len(window) < MIN_MESSAGES:
                break

            text = " ".join([m["text"] for m in window])

            chunks.append({
                "chunk_id": chunk_id,
                "topic_id": topic["topic_id"],
                "conv_id": topic["conv_id"],
                "start_msg": window[0]["msg_id"],
                "end_msg": window[-1]["msg_id"],
                "text": text
            })

            chunk_id += 1
            i += (CHUNK_SIZE - OVERLAP)

    return chunks
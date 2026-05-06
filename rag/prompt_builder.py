def build_prompt(query, chunks, topics, personas=None):
    chunk_block = "\n\n".join([
        f"[Chunk {i+1}] {c['text']}"
        for i, c in enumerate(chunks)
    ])

    topic_block = "\n\n".join([
        f"[Topic {i+1}] {t['summary']}"
        for i, t in enumerate(topics)
    ])

    persona_block = ""
    if personas:
        persona_block = f"\nPERSONA INFO:\n{personas}\n"

    return f"""
You are a precise conversational analysis assistant chatbot.

USER QUERY:
{query}

-----------------------------------
TOPIC SUMMARIES (may contain noise):
{topic_block}

-----------------------------------
CONVERSATION CHUNKS (may contain noise):
{chunk_block}

{persona_block}

-----------------------------------
INSTRUCTIONS:

1. Identify which topics and chunks are actually relevant
2. Ignore irrelevant ones completely
3. Answer ONLY from relevant information
4. If answer not found → say "Not found in conversation"
5. Do NOT hallucinate
6. If asked about people in conversation use persona INFO
7. Do not give information beyond the given context
8. You may describe the conversation in short and concise manner in your words,
as if you are chatting to the user who is giving user query.
-----------------------------------
FINAL ANSWER:
"""
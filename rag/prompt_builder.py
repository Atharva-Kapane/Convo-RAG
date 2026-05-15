def build_prompt(
    query,
    chunks,
    topics,
    personas=None,
    contradictions=None
):

    # -----------------------------------
    # CHUNKS
    # -----------------------------------
    chunk_block = "\n\n".join([
        (
            f"[Chunk {i+1}]\n"
            f"Conv ID: {c['conv_id']}\n"
            f"Recency Score: {c.get('recency_score', 0)}\n"
            f"Emotion Score: {c.get('emotion_score', 0)}\n"
            f"Text: {c['text']}"
        )
        for i, c in enumerate(chunks)
    ])

    # -----------------------------------
    # TOPICS
    # -----------------------------------
    topic_block = "\n\n".join([
        (
            f"[Topic {i+1}]\n"
            f"Conv ID: {t['conv_id']}\n"
            f"Summary: {t['summary']}"
        )
        for i, t in enumerate(topics)
    ])

    # -----------------------------------
    # PERSONAS
    # -----------------------------------
    persona_block = ""

    if personas:
        persona_block = f"""
-----------------------------------
PERSONA INFORMATION:
Use ONLY if relevant to the query.

{personas}
"""

    # -----------------------------------
    # CONTRADICTIONS
    # -----------------------------------
    contradiction_block = ""

    if contradictions:
        contradiction_block = f"""
-----------------------------------
POSSIBLE CONTRADICTIONS DETECTED:

{contradictions}

If contradictions exist:
- explain how conversation changed over time
- prioritize the MOST RECENT conversation first
- mention emotional shifts if visible
"""

    # -----------------------------------
    # FINAL PROMPT
    # -----------------------------------
    return f"""
You are an AI conversation analyst.

You are analyzing retrieved conversations between User 1 and User 2.

IMPORTANT:
- You are NOT one of the participants
- These are NOT conversations with you

Your task is to analyze and summarize the retrieved conversations clearly and naturally.

-----------------------------------
USER QUERY:
{query}

-----------------------------------
TOPIC CONTEXT:
{topic_block}

-----------------------------------
RETRIEVED CONVERSATION MEMORY:
{chunk_block}

{persona_block}

{contradiction_block}

-----------------------------------
IMPORTANT BEHAVIOR RULES:

1. Use conversation chunks as PRIMARY evidence
2. Use topic summaries only as supporting context
3. Ignore irrelevant chunks/topics completely
4. Prioritize:
   - newer conversations
   - emotionally stronger memories
5. If contradictions exist:
   - acknowledge them naturally
   - explain how context changed over time
6. If persona information is relevant:
   - use it naturally
   - do NOT dump raw persona JSON
7. NEVER hallucinate
8. NEVER invent missing details
9. If answer is unavailable:
   say exactly:
   "Not found in conversation"
10. Speak as an external analyst observing conversations
11. Never pretend you participated in the conversations

-----------------------------------
RESPONSE STYLE:

- Sound natural and conversational
- Keep response concise but insightful
- Avoid giant paragraphs
- Use short structured paragraphs
- Mention the MOST RECENT memory first
- Mention older conflicting memories afterward if relevant
- If emotional drift exists, explain it briefly
- Do NOT repeat raw chunks
- Do NOT sound robotic
- Do NOT list every retrieved memory unnecessarily

-----------------------------------
GOOD RESPONSE EXAMPLE:

"You mentioned your sister in multiple conversations.

Most recently, you talked about having an argument with her, which carried a frustrated tone.

In older conversations, you described her more positively, including mentioning that she helped you during exams.

This suggests the relationship tone changed over time depending on the situation."

-----------------------------------
FINAL ANSWER:
"""
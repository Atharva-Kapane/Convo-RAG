from models.llm import llm
from tqdm import tqdm


def summarize_block(text):
    prompt = f"""
Summarize this 100-message conversation chunk.

Rules:
- Be specific (mention concrete topics, events, or facts)
- Avoid generic phrases like "overall discussion"
- Include important entities (places, hobbies, jobs, relationships)
- Make it useful for retrieval
- Do not write anything outside given conversations. Be concise and on point.

Output:
- 7 to 10 bullet points
- Each bullet = one key idea

Conversation:
{text}
"""
    return llm.generate(prompt)


def generate_100_summaries(messages):
    summaries = []

    for i in tqdm(range(0, len(messages), 100)):
        block = messages[i:i+100]

        text = " ".join([m["text"] for m in block])

        summary = summarize_block(text)

        summaries.append({
            "start_msg": block[0]["msg_id"],
            "end_msg": block[-1]["msg_id"],
            "summary": summary
        })

    return summaries
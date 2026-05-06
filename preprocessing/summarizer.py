from models.llm import llm


def summarize_text(text):
    prompt = f"""
Summarize this conversation segment.
Do not add any knowledge from outside of given conversation, 
only consider conversation, but summarize in your words but CONCISELY.
Give concise short but on point summary. 
Basically you have to describe the topics in the conversation in short.
Do not write anything else besides summary

Focus on:
what things happened in the conversation and describe them as a third person

Keep it concise (3-5 lines).

Conversation:
{text}
"""
    return llm.generate(prompt)
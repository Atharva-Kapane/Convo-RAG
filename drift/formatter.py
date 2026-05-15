def build_day_context(conv_id, chunks, persona, selected_user):

    chunk_text = "\n".join([
        f"- {c['text']}"
        for c in chunks
    ])

    selected_persona = persona.get(selected_user, {})

    return f"""
DAY: {conv_id}

USER:
{selected_user}

PERSONA:
{selected_persona}

CONVERSATION:
{chunk_text}
"""
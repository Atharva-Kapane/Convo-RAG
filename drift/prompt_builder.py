def build_drift_prompt(days_context, selected_user):
    return f"""
You are an expert behavioral drift analyst.

Your task:
Analyze emotional and behavioral drift for ONLY {selected_user}
across chronological days.

IMPORTANT RULES:
- Treat each conv_id as one chronological day.
- Analyze ONLY {selected_user}
- Detect:
  1. tone
  2. emotional drift
  3. communication style changes
  4. trigger causing the drift

STRICT REQUIREMENTS:
- Trigger MUST be a factual ONE-LINE summary
  explaining the core reason behind the drift.
- Use ONLY provided context.
- NO hallucinations.
- NO assumptions.
- Keep outputs concise.
- Compare days sequentially.

OUTPUT MUST BE VALID JSON ONLY.

DAYS DATA:
{days_context}

OUTPUT FORMAT:

{{
  "timeline": [
    {{
      "day": 1,
      "tone": "curious and formal",
      "drift_from_previous_day": "initial state",
      "trigger": "discussion about relocating to Portland"
    }},
    {{
      "day": 2,
      "tone": "more casual and expressive",
      "drift_from_previous_day": "became more socially open",
      "trigger": "discussion about hobbies and music interests"
    }}
  ]
}}
"""
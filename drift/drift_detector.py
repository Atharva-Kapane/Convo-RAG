import json
import os

from drift.prompt_builder import build_drift_prompt
from drift.formatter import build_day_context

from models.gemini import gemini


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH = os.path.join(BASE_DIR, "processed_data", "chunks.json")
PERSONA_PATH = os.path.join(BASE_DIR, "processed_data", "personas.json")

MAX_ALLOWED_RANGE = 3


class DriftDetector:

    def __init__(self):

        self.chunks = self.load_json(CHUNKS_PATH)
        self.personas = self.load_json(PERSONA_PATH)

    # -----------------------------------
    # LOAD JSON
    # -----------------------------------
    def load_json(self, path):

        with open(path, "r") as f:
            return json.load(f)

    # -----------------------------------
    # VALIDATE
    # -----------------------------------
    def validate_inputs(self, start_day, end_day, selected_user):

        if start_day > end_day:
            raise ValueError(
                "start_day cannot be greater than end_day"
            )

        total_days = end_day - start_day + 1

        if total_days > MAX_ALLOWED_RANGE:
            raise ValueError(
                f"Maximum allowed range is {MAX_ALLOWED_RANGE} days"
            )

        if selected_user not in ["user1", "user2"]:
            raise ValueError(
                "selected_user must be user1 or user2"
            )

    # -----------------------------------
    # GET DAY CHUNKS
    # -----------------------------------
    def get_day_chunks(self, conv_id):

        return [
            c for c in self.chunks
            if c["conv_id"] == conv_id
        ]

    # -----------------------------------
    # GET DAY PERSONA
    # -----------------------------------
    def get_day_persona(self, conv_id):

        for p in self.personas:

            if p["conv_id"] == conv_id:
                return p["persona"]

        return {}

    # -----------------------------------
    # BUILD CONTEXT
    # -----------------------------------
    def build_context(
        self,
        start_day,
        end_day,
        selected_user
    ):

        contexts = []

        for conv_id in range(start_day, end_day + 1):

            chunks = self.get_day_chunks(conv_id)

            if not chunks:
                continue

            persona = self.get_day_persona(conv_id)

            context = build_day_context(
                conv_id=conv_id,
                chunks=chunks,
                persona=persona,
                selected_user=selected_user
            )

            contexts.append(context)

        return "\n\n".join(contexts)

    # -----------------------------------
    # MAIN ANALYSIS
    # -----------------------------------
    def analyze(
        self,
        start_day,
        end_day,
        selected_user
    ):

        self.validate_inputs(
            start_day,
            end_day,
            selected_user
        )

        context = self.build_context(
            start_day,
            end_day,
            selected_user
        )

        prompt = build_drift_prompt(
            context,
            selected_user
        )

        result = gemini.generate_json(
            prompt,
            temperature=0.2
        )

        return result
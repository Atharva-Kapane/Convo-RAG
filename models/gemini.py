import os
import json
import time

from google import genai
from dotenv import load_dotenv

load_dotenv()


class Gemini:

    def __init__(self, model_name="gemini-3.1-flash-lite-preview"):

        self.client = genai.Client(
            api_key=os.getenv("GENAI_API_KEY")
        )

        self.model = model_name

    # -----------------------------------
    # LIST AVAILABLE MODELS
    # -----------------------------------
    def list_available_models(self):

        try:
            for m in self.client.models.list():
                print(f"Model: {m.name}")

        except Exception as e:
            print(f"Error listing models: {e}")

    # -----------------------------------
    # NORMAL TEXT GENERATION
    # -----------------------------------
    def generate(self, prompt, temperature=0.3):

        retries = 3

        for attempt in range(retries):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": temperature
                    }
                )

                return response.text.strip()

            except Exception as e:

                print(f"Gemini Error Attempt {attempt+1}: {e}")

                time.sleep(2)

        return "Model temporarily unavailable."

    # -----------------------------------
    # STRICT JSON GENERATION
    # -----------------------------------
    def generate_json(self, prompt, temperature=0.2):

        try:

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json"
                }
            )

            return json.loads(response.text)

        except Exception as e:

            print("Gemini JSON Error:", e)
            return {
                "error": str(e)
            }


# Singleton
gemini = Gemini()
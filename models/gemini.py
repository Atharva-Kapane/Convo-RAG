import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


class Gemini:
    def __init__(self, model_name="gemini-3.1-flash-lite-preview"):
        self.client = genai.Client(
            api_key=os.getenv("GENAI_API_KEY")
        )
        self.model = model_name

    def list_available_models(self):
        """Prints all models supported by the current API key."""
        try:
            for m in self.client.models.list():
                print(f"Model: {m.name}")
        except Exception as e:
            print(f"Error listing models: {e}")

    def generate(self, prompt, temperature=0.3):
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
            print("Gemini Error:", e)
            return None


# Singleton
gemini = Gemini()
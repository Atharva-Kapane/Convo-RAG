import ollama

MODEL_NAME = "qwen3:8b"


class LLM:
    def __init__(self, model_name=MODEL_NAME):
        self.model = model_name

    def generate_json(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",  # forces valid JSON
            options={
                "temperature": 0.1
            }
        )
        return response["message"]["content"].strip()


# Singleton
llm = LLM()
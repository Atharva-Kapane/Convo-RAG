from models.gemini import gemini


def main():
    prompt = "Hi"

    response = gemini.generate(prompt)

    print("\nResponse:\n")
    print(response)

    gemini.list_available_models()


if __name__ == "__main__":
    main()
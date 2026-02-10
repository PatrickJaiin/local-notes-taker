import ollama


def summarize(transcript: str, model: str = "qwen3:8b", use_case: str = "general") -> str:
    prompt = (
        f"You are a note-taking assistant. The following is a transcript from a {use_case}.\n"
        "Create well-structured, clear notes from this transcript.\n"
        "Choose the most appropriate format and sections for this type of content.\n"
        "Be concise but thorough."
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    return response["message"]["content"]

import ollama


SYSTEM_PROMPTS = {
    "Meeting": (
        "You are a note-taking assistant. Summarize this meeting transcript into structured notes.\n"
        "Include: attendees (if mentioned), key decisions, action items, and discussion highlights.\n"
        "Use bullet points. Be concise but capture all important details."
    ),
    "Lecture": (
        "You are a note-taking assistant. Summarize this lecture transcript into study notes.\n"
        "Include: main topics, key concepts, definitions, and examples.\n"
        "Organize by topic. Use bullet points and highlight important terms."
    ),
    "Brainstorm": (
        "You are a note-taking assistant. Organize this brainstorming session.\n"
        "Group ideas by theme. Highlight the most promising ones.\n"
        "Include any decisions made and next steps."
    ),
    "Interview": (
        "You are a note-taking assistant. Summarize this interview.\n"
        "Include: key questions asked, notable answers, strengths observed, and overall impressions.\n"
        "Be structured and objective."
    ),
    "Stand-up": (
        "You are a note-taking assistant. Summarize this stand-up meeting.\n"
        "For each person mentioned, capture: what they did, what they're doing next, and any blockers.\n"
        "Keep it brief — this should be scannable in 30 seconds."
    ),
}

_DEFAULT_PROMPT = (
    "You are a note-taking assistant. The following is a transcript from a {use_case}.\n"
    "Create well-structured, clear notes from this transcript.\n"
    "Choose the most appropriate format and sections for this type of content.\n"
    "Be concise but thorough."
)


def summarize(
    transcript: str,
    model: str = "qwen3:8b",
    use_case: str = "general",
    host: str | None = None,
) -> str:
    prompt = SYSTEM_PROMPTS.get(use_case, _DEFAULT_PROMPT.format(use_case=use_case))

    client = ollama.Client(host=host) if host else ollama

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
        )
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure the Ollama app is running."
            ) from e
        if "not found" in msg or "404" in msg:
            raise RuntimeError(
                f"Model '{model}' not found. Run: ollama pull {model}"
            ) from e
        raise RuntimeError(f"Summarization failed: {e}") from e

    return response["message"]["content"]

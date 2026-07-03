from __future__ import annotations

import re
from typing import Callable

from app.transcribers.base import ProgressCb

# Per-use-case system prompts. The user's chosen use case decides the output
# format and processing; custom use cases fall through to _DEFAULT_PROMPT.
SYSTEM_PROMPTS: dict[str, str] = {
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

OnToken = Callable[[str], None]
CancelCheck = Callable[[], bool]

_MAX_TOKENS = 4096
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# model_id -> (model, tokenizer)
_cache: dict[str, tuple] = {}


def _get_model(model_id: str, progress_cb: ProgressCb | None = None):
    if model_id in _cache:
        return _cache[model_id]
    if progress_cb:
        progress_cb(None, "Loading summary model")
    from mlx_lm import load

    pair = load(model_id)
    _cache[model_id] = pair
    if progress_cb:
        progress_cb(None, "")  # clear "Loading…" so the title shows "Summarizing"
    return pair


def summarize(
    transcript: str,
    *,
    model_id: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    use_case: str = "Meeting",
    on_token: OnToken | None = None,
    cancel_check: CancelCheck | None = None,
    progress_cb: ProgressCb | None = None,
) -> str:
    """Generate a use-case-formatted summary locally via mlx-lm, streaming tokens
    to ``on_token`` for a live preview. Returns the final summary text."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_text = SYSTEM_PROMPTS.get(use_case, _DEFAULT_PROMPT.format(use_case=use_case))

    try:
        model, tokenizer = _get_model(model_id, progress_cb)
    except Exception as e:
        raise RuntimeError(f"Could not load summary model '{model_id}': {e}") from e

    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": transcript},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=0.3, top_p=0.95)

    pieces: list[str] = []
    try:
        for response in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=_MAX_TOKENS, sampler=sampler
        ):
            pieces.append(response.text)
            if on_token is not None:
                on_token("".join(pieces))
            if cancel_check is not None and cancel_check():
                break
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {e}") from e

    return _strip_thinking("".join(pieces)).strip()


def _strip_thinking(text: str) -> str:
    """Remove a leading <think>...</think> block if a thinking model leaks one."""
    return _THINK_RE.sub("", text, count=1)


def unload() -> None:
    """Drop cached summary models to free memory."""
    from app.models.manager import free_memory

    _cache.clear()
    free_memory()

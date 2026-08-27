from __future__ import annotations

import re
import threading
from typing import Callable

from app.mlx_runtime import on_mlx_thread
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

# Above this, summarize in sections and then summarize the section summaries. An
# hour-long lecture is ~9,000 words; handing that to a 4B model in one shot fits
# the context window but flattens the detail, and can run past _MAX_TOKENS.
_MAP_REDUCE_WORD_THRESHOLD = 4500
_SECTION_WORDS = 2500

_SECTION_PROMPT = (
    "You are summarizing ONE SECTION of a longer {use_case} transcript.\n"
    "Capture every substantive point in this section as detailed bullet points.\n"
    "Do not write an introduction or conclusion — this will be merged with other sections."
)
_REDUCE_PREFIX = (
    "The following are section-by-section notes from a single {use_case}, in order. "
    "Merge them into one coherent set of notes, removing duplication and keeping "
    "the detail.\n\n"
)

# model_id -> (model, tokenizer)
_cache: dict[str, tuple] = {}
_load_lock = threading.Lock()


@on_mlx_thread
def _get_model(model_id: str, progress_cb: ProgressCb | None = None):
    # Loaded on the shared MLX worker thread — mlx-lm weights are pinned to the
    # thread that created them, and any other thread loses them (see
    # app.mlx_runtime). The generation in _generate runs on that same thread.
    cached = _cache.get(model_id)
    if cached is not None:
        return cached
    # Serialize loads: two threads racing here would each pull a multi-GB model
    # into memory before either populated the cache.
    with _load_lock:
        cached = _cache.get(model_id)
        if cached is not None:
            return cached
        if progress_cb:
            progress_cb(None, "Loading summary model")
        from mlx_lm import load

        pair = load(model_id)
        _cache[model_id] = pair
    if progress_cb:
        progress_cb(None, "")  # clear "Loading…" so the title shows "Summarizing"
    return pair


def _split_sections(transcript: str, section_words: int = _SECTION_WORDS) -> list[str]:
    """Split a transcript into roughly equal word-count sections."""
    words = transcript.split()
    if len(words) <= section_words:
        return [transcript]
    count = max(2, -(-len(words) // section_words))  # ceil
    per = -(-len(words) // count)
    return [" ".join(words[i:i + per]) for i in range(0, len(words), per)]


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
    to ``on_token`` for a live preview. Returns the final summary text.

    Long transcripts are summarized section by section and then merged, so an
    hour-long recording keeps its detail instead of being flattened into a
    handful of bullets."""
    prompt_text = SYSTEM_PROMPTS.get(use_case, _DEFAULT_PROMPT.format(use_case=use_case))

    try:
        model, tokenizer = _get_model(model_id, progress_cb)
    except Exception as e:
        raise RuntimeError(f"Could not load summary model '{model_id}': {e}") from e

    if len(transcript.split()) <= _MAP_REDUCE_WORD_THRESHOLD:
        return _generate(
            model, tokenizer, prompt_text, transcript,
            on_token=on_token, cancel_check=cancel_check,
        )

    sections = _split_sections(transcript)
    section_prompt = _SECTION_PROMPT.format(use_case=use_case)
    notes: list[str] = []
    for i, section in enumerate(sections, start=1):
        if cancel_check is not None and cancel_check():
            break
        if progress_cb is not None:
            progress_cb(i / (len(sections) + 1), f"Summarizing {i}/{len(sections)}")
        # Prefix the live preview with the sections already done so the dropdown
        # keeps growing instead of restarting on every section.
        done = "\n\n".join(notes)
        notes.append(
            _generate(
                model, tokenizer, section_prompt, section,
                on_token=(lambda t, d=done: on_token(f"{d}\n\n{t}" if d else t))
                if on_token is not None
                else None,
                cancel_check=cancel_check,
            )
        )

    merged = "\n\n".join(n for n in notes if n)
    if not merged or (cancel_check is not None and cancel_check()):
        return merged
    if progress_cb is not None:
        progress_cb(None, "Merging notes")
    return _generate(
        model, tokenizer, prompt_text,
        _REDUCE_PREFIX.format(use_case=use_case) + merged,
        on_token=on_token, cancel_check=cancel_check,
    )


@on_mlx_thread
def _generate(
    model,
    tokenizer,
    system_prompt: str,
    user_content: str,
    *,
    on_token: OnToken | None = None,
    cancel_check: CancelCheck | None = None,
) -> str:
    """One streamed mlx-lm completion, returned with any <think> block stripped.

    Runs on the shared MLX worker thread, so ``on_token`` and ``cancel_check`` are
    invoked from there. Both are plain attribute reads/writes in the menu bar;
    neither may block on the MLX executor or touch MLX itself."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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

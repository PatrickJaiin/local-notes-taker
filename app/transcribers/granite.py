from __future__ import annotations

import numpy as np

from app.transcribers.base import (
    CancelCheck,
    ProgressCb,
    Transcriber,
    load_wav_mono16k,
    strip_special_tokens,
)

# Granite Speech can't cheaply stream and is slow on Mac, so it transcribes only
# on stop. Audio is chunked into windows to stay within a sane context length.
_WINDOW_SECONDS = 30
_OVERLAP_SECONDS = 2  # carry context across window boundaries so words aren't sliced
_MAX_NEW_TOKENS_PER_WINDOW = 384


def _merge_overlap(a: str, b: str, max_words: int = 12) -> str:
    """Concatenate two segment transcripts, eliding a duplicated run of words at the
    seam (produced by the audio overlap between consecutive windows)."""
    if not a:
        return b
    if not b:
        return a
    aw, bw = a.split(), b.split()

    def norm(words: list[str]) -> list[str]:
        return [w.lower().strip(".,!?;:'\"") for w in words]

    limit = min(len(aw), len(bw), max_words)
    overlap = 0
    for k in range(limit, 0, -1):
        if norm(aw[-k:]) == norm(bw[:k]):
            overlap = k
            break
    return " ".join(aw + bw[overlap:])

_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant."
)
_USER_PROMPT = "<|audio|>can you transcribe the speech into a written format?"


_projector_patched = False


def _patch_projector_broadcast() -> None:
    """Make GraniteSpeechEncoderProjector broadcast its query over audio blocks.

    The projector reshapes the encoder output to ``(blocks, window, dim)`` but hands
    the Q-Former a query of shape ``(1, num_queries, dim)``. Up to transformers
    5.12 the Q-Former's hand-rolled matmul attention broadcast batch 1 against
    ``blocks``; from 5.13 it goes through ``ALL_ATTENTION_FUNCTIONS`` (SDPA), which
    returns a batch-1 result, and the following ``view(batch, blocks*queries, -1)``
    raises. Expanding the query to the block batch first is correct under both
    code paths (verified byte-for-byte identical transcripts on 5.12.1 and 5.16.1).
    """
    global _projector_patched
    if _projector_patched:
        return
    _projector_patched = True
    try:
        import math

        import torch.nn.functional as F
        from transformers.models.granite_speech import modeling_granite_speech as mg

        projector_cls = mg.GraniteSpeechEncoderProjector
    except Exception:
        return  # different transformers layout; leave upstream code alone

    def forward(self, hidden_states):
        batch_size, seq_len, dim = hidden_states.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad), "constant", 0)
        hidden_states = hidden_states.view(batch_size * nblocks, self.window_size, dim)
        query = self.query.expand(hidden_states.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=None,
            return_dict=True,
        )
        return self.linear(
            query_output.last_hidden_state.view(
                batch_size, nblocks * self.window_size // self.downsample_rate, -1
            )
        )

    projector_cls.forward = forward


class GraniteTranscriber(Transcriber):
    """IBM Granite Speech 3.3 8B via HuggingFace transformers + peft (PyTorch).

    This is the heavy backend: ~34 GB resident in fp32. It does not stream — the
    menu bar shows "Transcribing on stop…" and runs ``transcribe_file`` once. The
    app unloads it (``unload``) before loading the summary LLM so they never
    co-reside.

    Two platform quirks are handled here (both reproduced on 2026-09-02):

    * **fp32 on MPS.** In fp16 the conformer encoder degenerates on Apple GPUs and
      the LLM decodes runs of "1", "s", "0" or "…" instead of speech; bf16 decodes
      to nothing at all. fp32 transcribes correctly, so that is what MPS gets even
      though it doubles the footprint. CPU was always fp32.
    * **Projector query broadcast.** transformers >= 5.13 routes the Q-Former
      through the shared attention backend, which no longer broadcasts the
      projector's single query batch across the N audio blocks — the projector
      then fails with ``shape '[1, 300, -1]' is invalid for input of size 3072``.
      ``_patch_projector_broadcast`` expands the query to the block batch up
      front, which is a no-op for the older matmul path.
    """

    supports_streaming = False

    def __init__(self, model_id: str = "ibm-granite/granite-speech-3.3-8b", device: str = "mps") -> None:
        super().__init__(model_id)
        self.device = device
        self._torch = None
        self._processor = None
        self._tokenizer = None
        self._model = None

    def load(self, progress_cb: ProgressCb | None = None) -> None:
        if self._model is not None:
            return
        if progress_cb:
            progress_cb(None, "Loading Granite")

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._torch = torch

        # Fall back to CPU if MPS isn't available.
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        # fp32 everywhere — see the class docstring for why not fp16/bf16 on MPS.
        dtype = torch.float32

        _patch_projector_broadcast()

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, dtype=dtype
        )
        self._model.to(self.device)
        self._model.eval()

    def transcribe_file(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        progress_cb: ProgressCb | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        self.load()
        audio = load_wav_mono16k(audio_path)
        total = len(audio)
        window = _WINDOW_SECONDS * self.SAMPLE_RATE
        step = max(1, (_WINDOW_SECONDS - _OVERLAP_SECONDS) * self.SAMPLE_RATE)
        result = ""
        start = 0
        while start < total:
            if cancel_check is not None and cancel_check():
                break
            end = min(start + window, total)
            if progress_cb is not None:
                done_min = end // (self.SAMPLE_RATE * 60)
                total_min = max(1, round(total / (self.SAMPLE_RATE * 60)))
                progress_cb(end / total, f"Transcribed {done_min}/{total_min} min")
            seg = audio[start:end]
            if len(seg) >= self.SAMPLE_RATE // 10:  # skip <0.1s tail
                text = self._transcribe_segment(seg)
                if text:
                    result = _merge_overlap(result, text)
            if end >= total:
                break
            start += step
        return strip_special_tokens(result, context="final pass")

    def _transcribe_segment(self, seg: np.ndarray) -> str:
        torch = self._torch
        wav = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).unsqueeze(0)

        chat = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT},
        ]
        text = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text, wav, return_tensors="pt").to(self.device)
        # Match audio features to the model's dtype (the processor emits fp32).
        if "input_features" in inputs and hasattr(self._model, "dtype"):
            inputs["input_features"] = inputs["input_features"].to(self._model.dtype)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS_PER_WINDOW,
                num_beams=1,
                do_sample=False,
            )

        num_input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = outputs[:, num_input_tokens:]
        decoded = self._tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return decoded[0].strip() if decoded else ""

    # Non-streaming: these satisfy the interface but are never called (the menu bar
    # checks supports_streaming first).
    def start_stream(self, language: str | None = None) -> object:
        return None

    def feed(self, pcm: np.ndarray, session: object = None) -> str:
        return ""

    def end_stream(self, session: object = None) -> str:
        return ""

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None

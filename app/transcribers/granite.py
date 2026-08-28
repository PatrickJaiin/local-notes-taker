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


class GraniteTranscriber(Transcriber):
    """IBM Granite Speech 3.3 8B via HuggingFace transformers + peft (PyTorch).

    This is the heavy backend: ~18-22 GB resident. It does not stream — the menu
    bar shows "Transcribing on stop…" and runs ``transcribe_file`` once. The app
    unloads it (``unload``) before loading the summary LLM so they never co-reside.
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
        dtype = torch.float16 if self.device == "mps" else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=dtype
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
        # Match audio features to the model's dtype (avoids fp16/fp32 mismatch on MPS).
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

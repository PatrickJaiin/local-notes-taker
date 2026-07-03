from __future__ import annotations

import numpy as np

from app.transcribers.base import ProgressCb, Transcriber, append_text, load_wav_mono16k


class WhisperTranscriber(Transcriber):
    """Whisper Large V3 via mlx-whisper (Apple Silicon / MLX).

    mlx-whisper has no native streaming API, so the live preview is a pseudo-stream:
    every ``chunk_seconds`` the menu-bar loop hands us a window of fresh audio, we
    transcribe it independently and append the text. The final ``transcribe_file``
    pass re-transcribes the whole recording for clean, context-aware output.
    """

    def __init__(self, model_id: str = "mlx-community/whisper-large-v3-mlx", chunk_seconds: int = 10) -> None:
        super().__init__(model_id)
        self._chunk_seconds = max(3, int(chunk_seconds))
        self._language: str | None = None
        self._running = ""
        self._loaded = False

    @property
    def live_interval(self) -> float:
        return float(self._chunk_seconds)

    def load(self, progress_cb: ProgressCb | None = None) -> None:
        if self._loaded:
            return
        if progress_cb:
            progress_cb(None, "Loading Whisper")
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        # Pre-warm the singleton that transcribe() actually reads from (it caches
        # the model in ModelHolder, NOT via load_model). float16 matches the dtype
        # mlx_whisper.transcribe requests, so the first real call is a cache hit.
        ModelHolder.get_model(self.model_id, mx.float16)
        self._loaded = True

    def _transcribe(self, audio: np.ndarray, language: str | None) -> tuple[str, str | None]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.model_id,
            language=language,
        )
        return (result.get("text") or "").strip(), result.get("language")

    def transcribe_file(self, audio_path: str, language: str | None = None) -> str:
        self.load()
        audio = load_wav_mono16k(audio_path)
        text, _ = self._transcribe(audio, language)
        return text

    def start_stream(self, language: str | None = None) -> None:
        self.load()
        self._language = language
        self._running = ""

    def feed(self, pcm: np.ndarray) -> str:
        try:
            text, detected = self._transcribe(pcm, self._language)
        except Exception:
            return self._running
        # Lock onto the first chunk's detected language so a later silent/noisy
        # chunk can't flip the live preview to a different language.
        if self._language is None and detected and text.strip():
            self._language = detected
        self._running = append_text(self._running, text)
        return self._running

    def end_stream(self) -> str:
        return self._running

    def unload(self) -> None:
        self._running = ""
        self._loaded = False
        # mlx-whisper caches the loaded model in a module-level ModelHolder
        # singleton (not on this object), so reset it or the weights stay resident.
        try:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
        except Exception:
            pass

from __future__ import annotations

import numpy as np

from app.mlx_runtime import on_mlx_thread
from app.transcribers.base import (
    CancelCheck,
    ProgressCb,
    Transcriber,
    append_text,
    load_wav_mono16k,
    strip_special_tokens,
)

# Above this, disable Whisper's condition_on_previous_text. Carrying decoded text
# forward helps short clips but is the classic trigger for Whisper's repetition
# loops on long recordings — one bad segment then conditions every segment after it.
_CONDITION_ON_PREVIOUS_MAX_SECONDS = 300


class WhisperTranscriber(Transcriber):
    """Whisper Large V3 via mlx-whisper (Apple Silicon / MLX).

    mlx-whisper has no native streaming API, so the live preview is a pseudo-stream:
    every ``chunk_seconds`` the menu-bar loop hands us a window of fresh audio, we
    transcribe it independently and append the text. The final ``transcribe_file``
    pass re-transcribes the whole recording for clean, context-aware output —
    mlx-whisper does its own 30 s windowing internally, so we hand it the lot.

    Every MLX-touching method is pinned to the shared MLX worker thread (see
    app.mlx_runtime): MLX 0.31 destroys a thread's GPU streams on thread exit, so
    a model loaded on a short-lived thread is unusable afterwards. That also makes
    the session token below a guard against *logical* staleness (a loop thread
    from a finished recording) rather than a data race.
    """

    def __init__(self, model_id: str = "mlx-community/whisper-large-v3-mlx", chunk_seconds: int = 10) -> None:
        super().__init__(model_id)
        self._chunk_seconds = max(3, int(chunk_seconds))
        self._language: str | None = None
        self._running = ""
        self._loaded = False
        self._session = 0

    @property
    def live_interval(self) -> float:
        return float(self._chunk_seconds)

    @on_mlx_thread
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

    def _transcribe(
        self, audio: np.ndarray, language: str | None, *, condition_on_previous: bool = True
    ) -> tuple[str, str | None]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.model_id,
            language=language,
            condition_on_previous_text=condition_on_previous,
        )
        return (result.get("text") or "").strip(), result.get("language")

    @on_mlx_thread
    def transcribe_file(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        progress_cb: ProgressCb | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        self.load()
        if cancel_check is not None and cancel_check():
            return ""
        audio = load_wav_mono16k(audio_path)
        # mlx-whisper drives its own internal segment loop with no progress or
        # cancel hook, so this is a single indeterminate step rather than a bar.
        if progress_cb is not None:
            progress_cb(None, "Transcribing")
        seconds = len(audio) / self.SAMPLE_RATE
        text, _ = self._transcribe(
            audio,
            language,
            condition_on_previous=seconds <= _CONDITION_ON_PREVIOUS_MAX_SECONDS,
        )
        return strip_special_tokens(text, context="final pass")

    @on_mlx_thread
    def start_stream(self, language: str | None = None) -> object:
        self.load()
        self._language = language
        self._running = ""
        self._session += 1
        return self._session

    @on_mlx_thread
    def feed(self, pcm: np.ndarray, session: object = None) -> str:
        if session != self._session:
            return ""  # stale loop thread from a finished recording
        try:
            text, detected = self._transcribe(pcm, self._language)
        except Exception:
            return self._running
        # Lock onto the first chunk's detected language so a later silent/noisy
        # chunk can't flip the live preview to a different language.
        if self._language is None and detected and text.strip():
            self._language = detected
        self._running = append_text(self._running, strip_special_tokens(text))
        return self._running

    @on_mlx_thread
    def end_stream(self, session: object = None) -> str:
        if session != self._session:
            return ""
        return self._running

    @on_mlx_thread
    def unload(self) -> None:
        # Dropping ModelHolder's weights is MLX work; keep it on the worker.
        self._running = ""
        self._session += 1  # invalidate any in-flight session
        self._loaded = False
        # mlx-whisper caches the loaded model in a module-level ModelHolder
        # singleton (not on this object), so reset it or the weights stay resident.
        try:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
        except Exception:
            pass

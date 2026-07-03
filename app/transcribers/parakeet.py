from __future__ import annotations

import numpy as np

from app.transcribers.base import ProgressCb, Transcriber, load_wav_mono16k


class ParakeetTranscriber(Transcriber):
    """Default backend: NVIDIA Parakeet TDT via parakeet-mlx (Apple Silicon / MLX).

    Parakeet has a native streaming API, so it powers a true live transcript while
    recording. The model auto-detects language, so the ``language`` argument is
    accepted for interface parity but ignored.
    """

    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3") -> None:
        super().__init__(model_id)
        self._model = None
        self._mx = None
        self._stream_ctx = None
        self._streamer = None

    def load(self, progress_cb: ProgressCb | None = None) -> None:
        if self._model is not None:
            return
        if progress_cb:
            progress_cb(None, "Loading Parakeet")
        import mlx.core as mx  # noqa: F401
        from parakeet_mlx import from_pretrained

        self._mx = mx
        self._model = from_pretrained(self.model_id)

    def _to_mx(self, pcm: np.ndarray):
        return self._mx.array(np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1))

    # --- Final full pass ---

    def transcribe_file(self, audio_path: str, language: str | None = None) -> str:
        self.load()
        # parakeet-mlx's transcribe() only accepts a file path and routes file
        # loading through ffmpeg. To stay ffmpeg-free we load the WAV ourselves
        # and run it through the streaming API (which accepts mx arrays), feeding
        # it in windows so context carries across the whole recording.
        audio = load_wav_mono16k(audio_path)
        ctx = self._model.transcribe_stream(context_size=(256, 256))
        streamer = ctx.__enter__()
        try:
            step = 30 * self.SAMPLE_RATE
            for i in range(0, len(audio), step):
                chunk = audio[i:i + step]
                if len(chunk):
                    streamer.add_audio(self._to_mx(chunk))
            return (streamer.result.text or "").strip()
        finally:
            ctx.__exit__(None, None, None)

    # --- Live streaming ---

    def start_stream(self, language: str | None = None) -> None:
        self.load()
        # context_size=(left, right) attention context in encoder frames.
        self._stream_ctx = self._model.transcribe_stream(context_size=(256, 256))
        self._streamer = self._stream_ctx.__enter__()

    def feed(self, pcm: np.ndarray) -> str:
        if self._streamer is None:
            return ""
        self._streamer.add_audio(self._to_mx(pcm))
        return (self._streamer.result.text or "").strip()

    def end_stream(self) -> str:
        if self._streamer is None:
            return ""
        text = (self._streamer.result.text or "").strip()
        try:
            self._stream_ctx.__exit__(None, None, None)
        finally:
            self._stream_ctx = None
            self._streamer = None
        return text

    def unload(self) -> None:
        self._streamer = None
        self._stream_ctx = None
        self._model = None

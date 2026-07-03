from __future__ import annotations

from abc import ABC, abstractmethod
from math import gcd
from typing import Callable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000

# A progress callback: progress_cb(fraction_or_None, human_label).
ProgressCb = Callable[[float | None, str], None]


class Transcriber(ABC):
    """A pluggable speech-to-text backend.

    All backends consume 16 kHz mono float32 audio. The app keeps every backend's
    code bundled; only the model weights download on demand (in ``load``).

    Two transcription paths:
      * Streaming (``start_stream``/``feed``/``end_stream``) drives the live preview
        while recording. Backends that can't stream cheaply set
        ``supports_streaming = False`` and the UI shows "transcribing on stop".
      * ``transcribe_file`` runs a final, full-quality pass on the complete WAV when
        the user stops.
    """

    SAMPLE_RATE = SAMPLE_RATE

    # Class-level so the UI can check it per backend without instantiating
    # (and therefore loading) the model. Granite overrides this to False.
    supports_streaming: bool = True

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def load(self, progress_cb: ProgressCb | None = None) -> None:
        """Download weights (reporting progress) and instantiate the model. Idempotent."""

    @abstractmethod
    def transcribe_file(self, audio_path: str, language: str | None = None) -> str:
        """Final, full-quality transcription of a complete WAV (called on stop)."""

    @abstractmethod
    def start_stream(self, language: str | None = None) -> None:
        """Begin a streaming session for live preview."""

    @abstractmethod
    def feed(self, pcm: np.ndarray) -> str:
        """Push a 16 kHz mono float32 chunk; return the running transcript so far."""

    @abstractmethod
    def end_stream(self) -> str:
        """Flush the stream and return the final streamed transcript."""

    def unload(self) -> None:
        """Release the model from memory. Overridden by heavy backends (Granite)
        so the summary LLM can load without co-residing with the ASR model."""
        return None


def wav_to_float_mono(data: np.ndarray) -> np.ndarray:
    """Normalize a scipy-read WAV array to a 1-D float32 [-1, 1] mono signal."""
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.reshape(-1)


def load_wav_mono16k(path: str) -> np.ndarray:
    """Read a WAV into a 1-D float32 [-1, 1] array at 16 kHz mono. Resamples and
    downmixes as needed so no external ffmpeg is required."""
    rate, data = wavfile.read(path)
    audio = wav_to_float_mono(data)

    if rate != SAMPLE_RATE:
        g = gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(rate) // g)

    return np.ascontiguousarray(audio, dtype=np.float32)


def append_text(running: str, addition: str) -> str:
    """Join transcript fragments with a single separating space."""
    addition = addition.strip()
    if not addition:
        return running
    if not running:
        return addition
    return f"{running} {addition}"

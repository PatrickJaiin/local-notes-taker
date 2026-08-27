from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections import Counter
from math import gcd
from typing import Callable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000

# A progress callback: progress_cb(fraction_or_None, human_label).
ProgressCb = Callable[[float | None, str], None]
# Polled between units of work; return True to abandon the run early.
CancelCheck = Callable[[], bool]


class Transcriber(ABC):
    """A pluggable speech-to-text backend.

    All backends consume 16 kHz mono float32 audio. The app keeps every backend's
    code bundled; only the model weights download on demand (in ``load``).

    Two transcription paths:
      * Streaming (``start_stream``/``feed``/``end_stream``) drives the live preview
        while recording. Backends that can't stream cheaply set
        ``supports_streaming = False`` and the UI shows "transcribing on stop".
      * ``transcribe_file`` runs a final, full-quality pass on the complete WAV when
        the user stops. It reports per-chunk progress and polls ``cancel_check`` so
        an hour-long recording can be abandoned without waiting for it to finish.

    Streaming sessions are handed out as opaque tokens. A loop thread that outlives
    its recording holds a stale token, so its late ``feed``/``end_stream`` calls are
    ignored instead of driving — or tearing down — the *next* recording's session.
    """

    SAMPLE_RATE = SAMPLE_RATE

    # Class-level so the UI can check it per backend without instantiating
    # (and therefore loading) the model. Granite overrides this to False.
    supports_streaming: bool = True

    # Seconds of audio the live loop should accumulate between ``feed`` calls.
    # Backends override this; see ParakeetTranscriber for why it is not ~1 s.
    live_interval: float = 1.0

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def load(self, progress_cb: ProgressCb | None = None) -> None:
        """Download weights (reporting progress) and instantiate the model. Idempotent."""

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        progress_cb: ProgressCb | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        """Final, full-quality transcription of a complete WAV (called on stop)."""

    @abstractmethod
    def start_stream(self, language: str | None = None) -> object:
        """Begin a streaming session for live preview. Returns an opaque session
        token to pass back to ``feed``/``end_stream``."""

    @abstractmethod
    def feed(self, pcm: np.ndarray, session: object = None) -> str:
        """Push a 16 kHz mono float32 chunk; return the running transcript so far.
        Returns "" if ``session`` is not the currently-open session."""

    @abstractmethod
    def end_stream(self, session: object = None) -> str:
        """Flush the stream and return the final streamed transcript. A stale
        ``session`` is a no-op, so a late loop thread can't close a live session."""

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


def resample_to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Polyphase-resample a 1-D float32 signal to 16 kHz (no-op if already there).

    ``resample_poly`` is a single clean FIR pass, so the whole recording is
    converted at once rather than block-by-block — no per-block edge transients,
    and no ffmpeg."""
    if rate == SAMPLE_RATE:
        return np.ascontiguousarray(audio, dtype=np.float32)
    g = gcd(int(rate), SAMPLE_RATE)
    resampled = resample_poly(audio, SAMPLE_RATE // g, int(rate) // g)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def load_wav_mono16k(path: str) -> np.ndarray:
    """Read a WAV into a 1-D float32 [-1, 1] array at 16 kHz mono. Resamples and
    downmixes as needed so no external ffmpeg is required."""
    rate, data = wavfile.read(path)
    return resample_to_16k(wav_to_float_mono(data), int(rate))


def append_text(running: str, addition: str) -> str:
    """Join transcript fragments with a single separating space."""
    addition = addition.strip()
    if not addition:
        return running
    if not running:
        return addition
    return f"{running} {addition}"


# --- Special-token hygiene -------------------------------------------------
#
# parakeet-mlx's tokenizer.decode() is a bare vocabulary lookup with no special-
# token handling, and parakeet-tdt-0.6b-v3's vocabulary carries <unk>, <pad>,
# <|nospeech|>, <|spkchange|>, <|spk0|>..<|spk15|>, <|emo:*|> and 100+ <|xx|>
# language tags. Any of them the decoder emits would otherwise land verbatim in
# the transcript and then in the summarization prompt.

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>|</?(?:unk|pad|s|mask)>")

# The v3 model has diarization tokens in its label space. We never *ask* for them
# (TDT greedy decoding is unconditional — there is no prompt prefix to inject), but
# if the model emits them spontaneously they are a free speaker-turn signal. Log
# what we see so a future speaker-ID feature knows whether this route is viable
# before anyone builds a VAD + embedding-clustering pipeline.
_DIARIZATION_TOKEN_RE = re.compile(r"<\|(spkchange|spk\d+|diarize|nodiarize|audioseparator)\|>")

# Runs of blanks left behind by removed tokens — collapsed without touching newlines.
_BLANK_RUN_RE = re.compile(r"[ \t]{2,}")


def strip_special_tokens(text: str, *, context: str = "") -> str:
    """Remove model special tokens from decoded text, logging any diarization
    tokens seen (see the note above) before they are dropped."""
    if not text:
        return ""

    diarization = _DIARIZATION_TOKEN_RE.findall(text)
    if diarization:
        counts = dict(Counter(diarization))
        where = f" [{context}]" if context else ""
        print(f"[diarization]{where} tokens emitted: {counts}", file=sys.stderr, flush=True)

    cleaned = _SPECIAL_TOKEN_RE.sub(" ", text)
    return _BLANK_RUN_RE.sub(" ", cleaned).strip()

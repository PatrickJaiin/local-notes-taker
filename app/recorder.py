from __future__ import annotations

import tempfile
import threading

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000  # 16 kHz mono float32 — what every ASR backend expects
CHANNELS = 1


class Recorder:
    """Captures microphone audio at 16 kHz mono float32.

    Every captured chunk is retained so a final, full-quality transcription pass
    can run on the complete recording when the user stops. A separate live cursor
    lets the UI drain only the newly-captured audio for incremental/streaming
    transcription without discarding anything.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._live_cursor = 0
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._chunks.append(indata.copy())

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._chunks = []
            self._live_cursor = 0
            try:
                self._stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._callback,
                )
                self._stream.start()
                self._recording = True
            except sd.PortAudioError as e:
                msg = str(e)
                if "no" in msg.lower() and "device" in msg.lower():
                    raise RuntimeError(
                        "No microphone found. Check System Settings > Privacy & Security > Microphone."
                    ) from e
                raise RuntimeError(f"Could not access microphone: {msg}") from e

    def drain_live(self) -> np.ndarray | None:
        """Return audio captured since the last drain as a 1-D float32 array, without
        discarding it (the full recording is preserved for the final pass). Returns
        None when no new audio has arrived or recording has ended — so a live-loop
        thread that outlives stop()/cancel() can never feed the model again."""
        with self._lock:
            if not self._recording or self._live_cursor >= len(self._chunks):
                return None
            new = self._chunks[self._live_cursor:]
            self._live_cursor = len(self._chunks)
        if not new:
            return None
        return np.concatenate(new, axis=0).astype(np.float32).reshape(-1)

    def _stop_stream(self) -> None:
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None
        # Stop outside the lock — stream.stop() waits for the callback, which needs the lock.
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def stop(self) -> str:
        """Stop recording and write the complete recording to a temp WAV; return its path."""
        self._stop_stream()
        with self._lock:
            chunks = self._chunks
            self._chunks = []
            self._live_cursor = 0
        if not chunks:
            raise RuntimeError("No audio was captured. Check your microphone.")
        audio = np.concatenate(chunks, axis=0)
        return write_wav(audio)

    def cancel(self) -> None:
        """Stop recording and discard all audio."""
        self._stop_stream()
        with self._lock:
            self._chunks = []
            self._live_cursor = 0


def write_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a float32 [-1, 1] mono array to a temp 16-bit WAV; return the path.
    The caller owns the file and is responsible for deleting it."""
    audio_int16 = np.clip(audio.reshape(-1) * 32767, -32768, 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, sample_rate, audio_int16)
    tmp.close()
    return tmp.name

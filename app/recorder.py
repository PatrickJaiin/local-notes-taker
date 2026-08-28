from __future__ import annotations

import os
import sys
import tempfile
import threading

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from app.transcribers.base import resample_to_16k

SAMPLE_RATE = 16000  # 16 kHz mono float32 — what every ASR backend expects
CHANNELS = 1

# The live preview is fed to a mel front-end with a 10 ms hop (160 samples at
# 16 kHz). parakeet-mlx's StreamingParakeet.add_audio() advances its internal
# buffer by mel_frames * hop_length, which is one hop MORE than the audio it
# consumed — so any sub-hop remainder we hand it is silently discarded. Feeding
# whole multiples of the hop makes that remainder zero. At a 1 s cadence the
# unaligned loss is up to 10 ms per call: ~36 s of a one-hour lecture.
HOP_LENGTH = 160


def _log(msg: str) -> None:
    print(f"[recorder] {msg}", file=sys.stderr, flush=True)


class Recorder:
    """Captures microphone audio as mono float32.

    Capture runs at the input device's native rate (48 kHz on built-in Mac mics)
    and is resampled to 16 kHz once, on stop, with a single polyphase pass —
    rather than asking PortAudio to convert every buffer at its default quality.

    Every captured chunk is retained so a final, full-quality transcription pass
    can run on the complete recording when the user stops. A separate live cursor
    lets the UI drain only the newly-captured audio for incremental/streaming
    transcription without discarding anything.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._live_cursor = 0
        self._live_tail = np.zeros(0, dtype=np.float32)
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False
        self._rate = SAMPLE_RATE
        self._overflows = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def overflow_count(self) -> int:
        """How many input-overflow events PortAudio reported this session. Non-zero
        means the recording has holes in it — usually the first recording, when the
        model download/load is competing for I/O."""
        return self._overflows

    def _callback(self, indata, frames, time_info, status) -> None:
        # Runs on the CoreAudio realtime thread. Taking a lock here would block that
        # thread on whatever the UI or drain thread is doing (and start() used to
        # hold the lock across stream.start(), inverting priority on every record).
        # list.append is atomic under the GIL, which is all the ordering we need —
        # drain_live and stop() only ever read a prefix of the list.
        if status and status.input_overflow:
            self._overflows += 1
        self._chunks.append(indata.copy())

    @staticmethod
    def _device_rate() -> int:
        """The default input device's native sample rate, or 16 kHz if unknown."""
        try:
            info = sd.query_devices(kind="input")
            rate = int(info["default_samplerate"])
            return rate if rate > 0 else SAMPLE_RATE
        except Exception:
            return SAMPLE_RATE

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._chunks = []
            self._live_cursor = 0
            self._live_tail = np.zeros(0, dtype=np.float32)
            self._overflows = 0
            self._rate = self._device_rate()

        stream = None
        try:
            stream = self._open_stream(self._rate)
        except Exception:
            # Native rate refused: fall back to letting PortAudio convert, which
            # is what this app did before and still beats not recording at all.
            if self._rate != SAMPLE_RATE:
                _log(f"native rate {self._rate} Hz unavailable; falling back to 16 kHz")
                self._rate = SAMPLE_RATE
                stream = self._open_stream(SAMPLE_RATE)
            else:
                raise

        # Start outside the lock: the callback can fire before start() returns, and
        # it must never find the lock held by this thread.
        try:
            stream.start()
        except Exception as e:
            try:
                stream.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not start the microphone: {e}") from e
        with self._lock:
            self._stream = stream
            self._recording = True

    def _open_stream(self, rate: int) -> sd.InputStream:
        try:
            return sd.InputStream(
                samplerate=rate,
                channels=CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
        except sd.PortAudioError as e:
            msg = str(e)
            if "no" in msg.lower() and "device" in msg.lower():
                raise RuntimeError(
                    "No microphone found. Check System Settings > Privacy & Security > Microphone."
                ) from e
            raise RuntimeError(f"Could not access microphone: {msg}") from e

    def drain_live(self) -> np.ndarray | None:
        """Return audio captured since the last drain, resampled to 16 kHz and
        trimmed to a whole number of mel hops, without discarding it (the full
        recording is preserved for the final pass). Returns None when no new audio
        has arrived or recording has ended — so a live-loop thread that outlives
        stop()/cancel() can never feed the model again.

        The sub-hop remainder is carried into the next drain rather than dropped;
        see HOP_LENGTH for why that matters."""
        with self._lock:
            if not self._recording or self._live_cursor >= len(self._chunks):
                return None
            new = self._chunks[self._live_cursor:]
            self._live_cursor = len(self._chunks)
            rate = self._rate

        if not new:
            return None
        block = np.concatenate(new, axis=0).astype(np.float32).reshape(-1)
        # Per-block resampling puts a short FIR transient at each block edge (~1 ms
        # for 48k->16k). That is fine for a preview; the archived WAV is converted
        # in one pass so the transcript that actually matters never sees a seam.
        block = resample_to_16k(block, rate)

        # Single consumer (the live loop), so the tail needs no lock of its own.
        block = np.concatenate([self._live_tail, block]) if len(self._live_tail) else block
        usable = (len(block) // HOP_LENGTH) * HOP_LENGTH
        self._live_tail = np.ascontiguousarray(block[usable:], dtype=np.float32)
        if usable == 0:
            return None
        return np.ascontiguousarray(block[:usable], dtype=np.float32)

    def _stop_stream(self) -> None:
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None
        # Stop outside the lock — stream.stop() waits for the callback, which needs
        # the GIL; holding our lock here would serialize against drain_live.
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def stop(self) -> str:
        """Stop recording and write the complete recording to a temp WAV at 16 kHz;
        return its path."""
        self._stop_stream()
        with self._lock:
            chunks = self._chunks
            rate = self._rate
            self._chunks = []
            self._live_cursor = 0
            self._live_tail = np.zeros(0, dtype=np.float32)
        if not chunks:
            raise RuntimeError("No audio was captured. Check your microphone.")
        if self._overflows:
            _log(f"{self._overflows} input overflow(s) — the recording may have gaps")
        audio = np.concatenate(chunks, axis=0).astype(np.float32).reshape(-1)
        return write_wav(resample_to_16k(audio, rate))

    def cancel(self) -> None:
        """Stop recording and discard all audio."""
        self._stop_stream()
        with self._lock:
            self._chunks = []
            self._live_cursor = 0
            self._live_tail = np.zeros(0, dtype=np.float32)


def write_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a float32 [-1, 1] mono array to a temp 16-bit WAV; return the path.
    The caller owns the file and is responsible for deleting it."""
    audio_int16 = np.clip(audio.reshape(-1) * 32767, -32768, 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        wavfile.write(tmp.name, sample_rate, audio_int16)
    except Exception:
        # Don't leave a zero-byte temp behind if the write failed (a full disk is
        # the realistic case — an hour of audio is ~115 MB).
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return tmp.name

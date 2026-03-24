import tempfile
import threading

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000  # Whisper's native sample rate
CHANNELS = 1


class Recorder:
    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata.copy())

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._chunks = []
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
                        "No microphone found. Check System Settings > Privacy > Microphone."
                    ) from e
                raise RuntimeError(f"Could not access microphone: {msg}") from e

    def flush(self) -> str | None:
        """Drain accumulated chunks to a temp WAV without stopping the stream."""
        with self._lock:
            if not self._recording or not self._chunks:
                return None
            audio = np.concatenate(self._chunks, axis=0)
            self._chunks = []

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)
        tmp.close()
        return tmp.name

    def _stop_stream(self):
        """Stop and close the audio stream outside the lock to avoid deadlock."""
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None
        # Stop outside lock — stream.stop() waits for the callback which needs the lock
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def stop(self) -> str:
        """Stop recording and return the path to the WAV file."""
        self._stop_stream()

        with self._lock:
            if not self._chunks:
                raise RuntimeError("No audio was captured. Check your microphone.")

            audio = np.concatenate(self._chunks, axis=0)
            self._chunks = []

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)
        tmp.close()
        return tmp.name

    def cancel(self):
        """Stop recording and discard all audio."""
        self._stop_stream()
        with self._lock:
            self._chunks = []

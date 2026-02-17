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

    def _callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())

    def start(self):
        with self._lock:
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def flush(self) -> str | None:
        """Drain accumulated chunks to a temp WAV without stopping the stream."""
        with self._lock:
            if not self._chunks:
                return None

            audio = np.concatenate(self._chunks, axis=0)
            self._chunks = []

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)
        tmp.close()
        return tmp.name

    def stop(self) -> str:
        """Stop recording and return the path to the WAV file."""
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            if not self._chunks:
                raise RuntimeError("No audio recorded")

            audio = np.concatenate(self._chunks, axis=0)
            # Convert float32 [-1, 1] to int16 for WAV
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)
            tmp.close()
            self._chunks = []
            return tmp.name

from __future__ import annotations

import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile

from app.transcribers.base import wav_to_float_mono


def clean_audio(input_path: str) -> str:
    """Apply a high-pass filter + gentle dynamic gain to a WAV file.

    Removes low-frequency rumble/hum and boosts quieter speakers toward a target
    level without amplifying silence. Returns the path to a new temp WAV; caller
    is responsible for unlinking.

    This is OFF by default (``clean_audio`` in config.yaml). ASR models are trained
    on unprocessed speech, and the previous version of this function was actively
    harmful to transcription: it soft-clipped every sample through tanh (~16%
    peak compression, i.e. broadband harmonic distortion) and could swing the gain
    8x within a quarter second, which in a distant-mic lecture mostly amplifies
    room noise during pauses. The tanh is gone and the gain is far gentler; even
    so, prefer the raw audio unless a recording is genuinely too quiet to decode.
    """
    rate, data = wavfile.read(input_path)
    audio = wav_to_float_mono(data)

    sos = signal.butter(4, 80, btype="highpass", fs=rate, output="sos")
    audio = signal.sosfilt(sos, audio).astype(np.float32)

    audio = _dynamic_gain(audio, rate)

    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, rate, audio_int16)
    tmp.close()
    return tmp.name


def _dynamic_gain(audio: np.ndarray, rate: int) -> np.ndarray:
    """Per-window peak normalization with max-gain cap and smoothing.

    Approximates ffmpeg's dynaudnorm: quiet windows get boosted toward
    target_peak, loud windows stay put, and windows below noise_floor
    aren't amplified so silence doesn't turn into hiss.

    The window is 2 s (was 0.5 s) and the cap is 3x (was 8x): a gain curve that
    moves quickly and far is itself an amplitude modulation the acoustic model
    has never heard, which costs more accuracy than the level gain buys.
    """
    window = int(rate * 2.0)
    hop = max(window // 2, 1)
    target_peak = 0.7
    max_gain = 3.0
    noise_floor = 0.02

    if len(audio) < window:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > noise_floor:
            return audio * min(target_peak / peak, max_gain)
        return audio

    starts = np.arange(0, len(audio) - window + 1, hop)
    if starts[-1] + window < len(audio):
        starts = np.append(starts, len(audio) - window)

    gains = np.empty(len(starts), dtype=np.float32)
    for i, start in enumerate(starts):
        peak = float(np.max(np.abs(audio[start:start + window])))
        if peak > noise_floor:
            gains[i] = min(target_peak / peak, max_gain)
        else:
            gains[i] = 1.0

    centers = starts + window // 2
    gain_curve = np.interp(np.arange(len(audio)), centers, gains).astype(np.float32)
    return audio * gain_curve

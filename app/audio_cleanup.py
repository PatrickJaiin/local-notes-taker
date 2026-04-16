import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile


def clean_audio(input_path: str) -> str:
    """Apply high-pass filter + dynamic gain + soft limit to a WAV file.

    Removes low-frequency rumble/hum, boosts quieter speakers toward a target
    level without amplifying silence, and soft-clips peaks. Returns the path
    to a new temp WAV; caller is responsible for unlinking.
    """
    rate, data = wavfile.read(input_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    sos = signal.butter(4, 80, btype="highpass", fs=rate, output="sos")
    audio = signal.sosfilt(sos, audio).astype(np.float32)

    audio = _dynamic_gain(audio, rate)

    audio = np.tanh(audio * 1.1)

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
    """
    window = int(rate * 0.5)
    hop = max(window // 2, 1)
    target_peak = 0.7
    max_gain = 8.0
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

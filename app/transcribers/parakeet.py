from __future__ import annotations

import numpy as np

from app.mlx_runtime import on_mlx_thread
from app.transcribers.base import (
    CancelCheck,
    ProgressCb,
    Transcriber,
    load_wav_mono16k,
    strip_special_tokens,
)

# Final-pass chunking. These are parakeet-mlx's own CLI defaults (cli.py:
# --chunk-duration 120, --overlap-duration 15) and they suit this model:
#   * The encoder's relative-position buffer is 2T-1 long, so the attention
#     intermediate grows as O(T^2). At 120 s that is T = 120*100/8 = 1500 frames
#     — comfortable; at 300 s the same tensor is ~6x larger.
#   * parakeet-tdt-0.6b-v3 ships pos_emb_max_len = 5000 encoder frames (~400 s).
#     Exceeding it forces a full positional-encoding rebuild mid-pass.
#   * A single un-chunked hour (45,000 frames) is not merely slow, it is
#     infeasible — hence chunking is mandatory, not an optimization.
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 15.0

# Encoder attention context (left, right), in encoder frames, for the live stream.
CONTEXT_SIZE = (256, 256)

# Seconds of audio per live `feed`. Deliberately NOT ~1 s: parakeet-mlx's
# StreamingParakeet.add_audio() runs get_logmel() over each window in isolation,
# and get_logmel normalizes per-feature *over the window's own time axis*. A short
# window is acoustically homogeneous, so a quiet one gets rescaled to look exactly
# like loud speech — which is what makes the TDT decoder spray repeated tokens.
# Longer windows carry enough loud/quiet contrast to normalize sanely. add_audio
# also drops the sub-hop remainder of every call, so fewer calls means less loss.
LIVE_INTERVAL_SECONDS = 12.0

_IMPORT_ERROR_HINT = (
    "Local Notes needs parakeet-mlx >= 0.5.2, < 0.6 — it uses the chunk-merge and "
    "mel helpers from parakeet_mlx.alignment / parakeet_mlx.audio, which are not "
    "part of the package's public __all__ and moved in earlier releases. "
    "Run: pip install 'parakeet-mlx>=0.5.2,<0.6'"
)


def _internals():
    """Import the parakeet-mlx helpers the chunked pass needs.

    These live outside parakeet_mlx.__all__, so the import is version-fragile by
    construction — it is isolated here and given an actionable error rather than
    surfacing as a bare ImportError from deep inside a transcription."""
    try:
        from parakeet_mlx.alignment import (
            merge_longest_common_subsequence,
            merge_longest_contiguous,
            sentences_to_result,
            tokens_to_sentences,
        )
        from parakeet_mlx.audio import get_logmel
    except ImportError as e:
        raise RuntimeError(f"{_IMPORT_ERROR_HINT} ({e})") from e
    return (
        get_logmel,
        merge_longest_contiguous,
        merge_longest_common_subsequence,
        tokens_to_sentences,
        sentences_to_result,
    )


def _progress_label(done_samples: int, total_samples: int) -> str:
    done_min = done_samples // (Transcriber.SAMPLE_RATE * 60)
    total_min = max(1, round(total_samples / (Transcriber.SAMPLE_RATE * 60)))
    return f"Transcribed {done_min}/{total_min} min"


class ParakeetTranscriber(Transcriber):
    """Default backend: NVIDIA Parakeet TDT via parakeet-mlx (Apple Silicon / MLX).

    Parakeet has a native streaming API, so it powers a true live transcript while
    recording. The model auto-detects language, so the ``language`` argument is
    accepted for interface parity but ignored.

    The final pass does NOT use the streaming API. transcribe_stream() re-normalizes
    every window's mel independently, carries one TDT decoder state across the whole
    recording (so a single degenerate repeat loop poisons everything after it), and
    swaps the encoder into local attention the model was never trained with. Instead
    we chunk the audio ourselves and call generate(), which takes an mx.array mel
    directly — no file path, so no ffmpeg — with a clean decoder state per chunk.

    Every method that touches MLX is pinned to the shared MLX worker thread (see
    app.mlx_runtime): MLX 0.31 destroys a thread's GPU streams when that thread
    exits, so a model loaded on the pre-warm thread became unusable the moment
    that thread finished. Callers keep their own threading model — the wrappers
    block until the work completes.

    That pinning also serializes the live stream against the final pass, which is
    required for correctness independently of the stream bug: both share one model
    object, and transcribe_stream() mutates it (it replaces pos_enc and all 24
    layers' self_attn). The session token below therefore guards *logical*
    staleness — a loop thread from a finished recording — not data races.
    """

    live_interval = LIVE_INTERVAL_SECONDS

    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3") -> None:
        super().__init__(model_id)
        self._model = None
        self._mx = None
        self._stream_ctx = None
        self._streamer = None
        self._session = 0

    @on_mlx_thread
    def load(self, progress_cb: ProgressCb | None = None) -> None:
        if self._model is not None:
            return
        if progress_cb:
            progress_cb(None, "Loading Parakeet")
        import mlx.core as mx  # noqa: F401
        from parakeet_mlx import from_pretrained

        _internals()  # fail fast on a bad parakeet-mlx, not mid-transcription
        self._mx = mx
        self._model = from_pretrained(self.model_id)

    def _to_mx(self, pcm: np.ndarray):
        # float32 is not a preference here, it is the only correct dtype:
        # get_logmel reinterprets the complex STFT output via mx.view(x, dtype)
        # to take |re| + |im|, which assumes a 4-byte element. bfloat16 input
        # yields 514 bins instead of 257 and raises. parakeet-mlx's own
        # load_audio() always returns float32 for the same reason.
        return self._mx.array(np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1))

    # --- Final full pass ---

    @on_mlx_thread
    def transcribe_file(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        progress_cb: ProgressCb | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        """Transcribe a complete WAV in overlapping chunks, merging at the seams.

        Mirrors parakeet-mlx's own BaseParakeet.transcribe() chunk loop, minus its
        load_audio() (which shells out to ffmpeg) — we read the WAV with scipy and
        build each chunk's mel ourselves."""
        self.load()
        # The file pass owns the shared encoder. transcribe_stream() leaves the model
        # in local-attention mode for as long as a session is open, and generate()
        # would then run through an attention configuration this model was never
        # trained with — silently, with no error. Close any session first.
        self.close_stream()

        (
            get_logmel,
            merge_contiguous,
            merge_lcs,
            tokens_to_sentences,
            sentences_to_result,
        ) = _internals()

        audio = load_wav_mono16k(audio_path)
        total = len(audio)
        if total == 0:
            return ""

        preprocess = self._model.preprocessor_config
        chunk_samples = int(CHUNK_SECONDS * self.SAMPLE_RATE)
        overlap_samples = int(OVERLAP_SECONDS * self.SAMPLE_RATE)
        step = max(1, chunk_samples - overlap_samples)

        all_tokens: list = []
        for start in range(0, total, step):
            if cancel_check is not None and cancel_check():
                break

            end = min(start + chunk_samples, total)
            if end - start < preprocess.hop_length:
                break  # too short to produce even one mel frame

            if progress_cb is not None:
                progress_cb(end / total, _progress_label(end, total))

            mel = get_logmel(self._to_mx(audio[start:end]), preprocess)
            chunk_tokens = self._model.generate(mel)[0].tokens

            # generate() times every token from the start of its own mel, so shift
            # the whole chunk onto the recording's timeline before merging. This is
            # what makes timestamps globally monotonic — the streaming path restarts
            # them at 0 every window, which is why it can't support seek or speakers.
            offset = start / self.SAMPLE_RATE
            for token in chunk_tokens:
                token.start += offset
                token.end = token.start + token.duration

            if not all_tokens:
                all_tokens = chunk_tokens
            elif chunk_tokens:
                try:
                    # Splice on the longest run of matching token ids whose start
                    # times agree; falls back to an LCS alignment when the overlap
                    # decoded too differently for a contiguous run to be found.
                    all_tokens = merge_contiguous(
                        all_tokens, chunk_tokens, overlap_duration=OVERLAP_SECONDS
                    )
                except RuntimeError:
                    all_tokens = merge_lcs(
                        all_tokens, chunk_tokens, overlap_duration=OVERLAP_SECONDS
                    )

            # range() would keep stepping past the tail and re-decode progressively
            # shorter slices of it (parakeet-mlx's own loop does exactly that).
            if end >= total:
                break

        if not all_tokens:
            return ""

        text = sentences_to_result(tokens_to_sentences(all_tokens)).text
        return strip_special_tokens(text, context="final pass")

    # --- Live streaming ---

    @on_mlx_thread
    def start_stream(self, language: str | None = None) -> object:
        self.load()
        self._close_stream()  # drop an orphan from an abandoned session
        # context_size=(left, right) attention context in encoder frames.
        self._stream_ctx = self._model.transcribe_stream(context_size=CONTEXT_SIZE)
        self._streamer = self._stream_ctx.__enter__()
        self._session += 1
        return self._session

    @on_mlx_thread
    def feed(self, pcm: np.ndarray, session: object = None) -> str:
        # A stale token means a loop thread from a finished recording; ignore it
        # rather than letting it drive the current session.
        if self._streamer is None or session != self._session:
            return ""
        self._streamer.add_audio(self._to_mx(pcm))
        return strip_special_tokens(self._streamer.result.text or "", context="live")

    @on_mlx_thread
    def end_stream(self, session: object = None) -> str:
        if self._streamer is None or session != self._session:
            return ""  # stale: must not tear down whoever owns the session now
        text = self._streamer.result.text or ""
        self._close_stream()
        return strip_special_tokens(text, context="live")

    @on_mlx_thread
    def close_stream(self) -> None:
        """Tear down any open session, whoever owns it. Used by the final pass,
        which must not run against a model left in streaming attention mode."""
        self._close_stream()

    def _close_stream(self) -> None:
        """MLX-thread-only; callers are already pinned there."""
        ctx, self._stream_ctx, self._streamer = self._stream_ctx, None, None
        if ctx is None:
            return
        try:
            # Restores the encoder's global attention and frees the KV caches.
            ctx.__exit__(None, None, None)
        except Exception:
            pass

    @on_mlx_thread
    def unload(self) -> None:
        # Releasing the model's arrays is itself MLX work, so it belongs on the
        # worker thread too.
        self._close_stream()
        self._model = None

from __future__ import annotations

import gc
import subprocess
import sys
import threading

from app import config as cfg
from app.mlx_runtime import run_on_mlx_thread
from app.transcribers.base import ProgressCb, Transcriber

# Granite 3.3 8B (~34 GB resident in fp32) is impractical below this; we warn but proceed.
GRANITE_8B_MIN_RAM_GB = 48


def _log(msg: str) -> None:
    print(f"[model] {msg}", file=sys.stderr, flush=True)


class ModelManager:
    """Owns ASR backend lifecycle: lazy weight download (with progress), load,
    and unload. Backends stay cached once loaded; the heavy Granite backend is
    unloaded explicitly between transcription and summarization so the two large
    models never co-reside in memory."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._loaded: dict[str, Transcriber] = {}
        self._lock = threading.Lock()

    def update_config(self, config: dict) -> None:
        self._config = config

    # --- Construction ---

    @staticmethod
    def transcriber_class(backend: str) -> type[Transcriber]:
        """Resolve a backend key to its Transcriber class (imports stay lazy so
        heavy ML deps aren't pulled in at startup)."""
        if backend == cfg.PARAKEET:
            from app.transcribers.parakeet import ParakeetTranscriber

            return ParakeetTranscriber
        if backend == cfg.WHISPER:
            from app.transcribers.whisper import WhisperTranscriber

            return WhisperTranscriber
        if backend == cfg.GRANITE:
            from app.transcribers.granite import GraniteTranscriber

            return GraniteTranscriber
        raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def backend_supports_streaming(backend: str) -> bool:
        return ModelManager.transcriber_class(backend).supports_streaming

    def build_transcriber(self, backend: str) -> Transcriber:
        model_id = cfg.model_id_for_backend(self._config, backend)
        klass = self.transcriber_class(backend)
        if backend == cfg.WHISPER:
            return klass(model_id, chunk_seconds=self._config.get("chunk_seconds", 10))
        if backend == cfg.GRANITE:
            return klass(model_id, device=self._config.get("granite_device", "mps"))
        return klass(model_id)

    # --- Readiness ---

    def ensure_ready(self, backend: str, progress_cb: ProgressCb | None = None) -> Transcriber:
        """Return a loaded transcriber for ``backend``, downloading weights (with
        progress) and loading the model if necessary. Cached after first load.

        Holds the lock across the whole download+load so concurrent callers (the
        live preview loop and the final pass) dedup instead of double-downloading.
        MUST be called from a background thread — never the UI thread."""
        with self._lock:
            cached = self._loaded.get(backend)
            if cached is not None:
                return cached

            model_id = cfg.model_id_for_backend(self._config, backend)
            _log(f"ensure_ready backend={backend} model={model_id}")
            # Always run the tracked download so any missing/partial files are
            # fetched WITH a visible "Downloading …%" — never silently inside model
            # load. When the cache is already complete it's a local-only check.
            self.download(model_id, progress_cb)
            _log("loading model…")
            transcriber = self.build_transcriber(backend)
            transcriber.load(progress_cb)
            self._loaded[backend] = transcriber
            _log(f"loaded backend={backend}")
            return transcriber

    # --- Download ---

    @staticmethod
    def is_downloaded(repo_id: str) -> bool:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id, local_files_only=True)
            return True
        except Exception as e:
            _log(f"is_downloaded({repo_id}) -> False ({e!r})")
            return False

    @staticmethod
    def download(repo_id: str, progress_cb: ProgressCb | None = None, retries: int = 3) -> None:
        from huggingface_hub import snapshot_download

        # Complete local cache: skip the network entirely, so a fully-downloaded
        # model loads instantly (and works offline without 3 timed-out retries).
        if ModelManager.is_downloaded(repo_id):
            return

        name = repo_id.split("/")[-1]
        if progress_cb:
            progress_cb(None, f"Downloading {name}")  # say it up front; % follows
        last_err: Exception | None = None
        for attempt in range(retries):
            # Fresh aggregate state each attempt so a failed retry's stale bars
            # can't corrupt the reported fraction.
            tqdm_class = _make_tqdm_class(progress_cb, repo_id) if progress_cb else None
            try:
                snapshot_download(repo_id, tqdm_class=tqdm_class)
                return
            except Exception as e:  # network hiccup / partial cache — retry
                last_err = e
                if progress_cb and attempt < retries - 1:
                    progress_cb(None, f"Retrying {name} ({attempt + 2}/{retries})")
        # Offline / repeated failure: if a local copy already exists, use it.
        if ModelManager.is_downloaded(repo_id):
            return
        raise RuntimeError(f"Failed to download {repo_id}: {last_err}")

    # --- Unload (RAM handoff) ---

    def unload(self, backend: str) -> None:
        with self._lock:
            transcriber = self._loaded.pop(backend, None)
        if transcriber is not None:
            try:
                transcriber.unload()
            except Exception:
                pass
        free_memory()

    def unload_all(self) -> None:
        with self._lock:
            transcribers = list(self._loaded.values())
            self._loaded.clear()
        for t in transcribers:
            try:
                t.unload()
            except Exception:
                pass
        free_memory()

    # --- RAM helpers ---

    @staticmethod
    def total_ram_gb() -> float | None:
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out.strip()) / (1024 ** 3)
        except Exception:
            return None

    def granite_ram_warning(self) -> str | None:
        """Return a warning string if this machine is likely too small for Granite 8B."""
        ram = self.total_ram_gb()
        if ram is not None and ram < GRANITE_8B_MIN_RAM_GB:
            return (
                f"Granite Speech 8B needs ~34 GB+ RAM; this Mac has {ram:.0f} GB. "
                "Transcription will be slow and may swap heavily."
            )
        return None


def free_memory() -> None:
    """Drop cached allocations. Runs on the shared MLX worker thread: both
    mx.clear_cache() and the array deallocations that gc.collect() triggers are
    MLX work, and MLX state is thread-scoped (see app.mlx_runtime)."""
    run_on_mlx_thread(_free_memory)


def _free_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _make_tqdm_class(progress_cb: ProgressCb, repo_id: str):
    """Build a tqdm subclass that aggregates byte-progress across all of
    snapshot_download's per-file bars and reports an overall fraction."""
    from tqdm.auto import tqdm as _tqdm

    label = f"Downloading {repo_id.split('/')[-1]}"
    # Prefer byte progress (HTTP downloads); fall back to file-count progress
    # (the Xet backend reports files, not bytes).
    state = {"byte": {}, "file": {}, "lock": threading.Lock()}

    def report() -> None:
        with state["lock"]:
            byte_bars = [(n, t) for (n, t) in state["byte"].values() if t]
            file_bars = [(n, t) for (n, t) in state["file"].values() if t]
        bars = byte_bars or file_bars
        if not bars:
            return
        done = sum(min(n, t) for n, t in bars)
        total = sum(t for _, t in bars)
        if total:
            progress_cb(done / total, label)

    class _ProgressTqdm(_tqdm):  # type: ignore[misc]
        def update(self, n=1):
            ret = super().update(n)
            try:
                if self.total:
                    bucket = "byte" if getattr(self, "unit", "") == "B" else "file"
                    with state["lock"]:
                        state[bucket][id(self)] = (self.n, self.total)
                    report()
            except Exception:
                pass
            return ret

    return _ProgressTqdm

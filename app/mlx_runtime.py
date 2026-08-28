"""A single long-lived thread that owns every MLX operation in the app.

Why this exists
---------------
MLX 0.31 scopes GPU streams to the thread that created them and destroys them
when that thread exits — ``mlx/core.cpython-314-darwin.so`` carries the string
"Destroy all streams created in current thread." A model loaded on one thread is
pinned to that thread's stream, so once the loading thread dies, any later
operation on that model fails from every other thread with::

    RuntimeError: There is no Stream(gpu, 0) in current thread.

This app hit exactly that: ``menubar._warm_backend`` pre-warmed Parakeet on a
short-lived thread, which then exited, and the first live-preview ``feed`` blew
up inside ``parakeet_mlx.parakeet.add_audio``'s ``mx.eval``.

Giving each new thread a fresh default stream
(``mx.set_default_stream(mx.new_stream(...))``) does NOT rescue an
already-loaded model — the pinning happens at load time. The only reliable fix
is for the load and every subsequent operation to happen on the same thread that
stays alive for the process lifetime. Hence: one worker, forever.

Serializing all MLX work on one thread is also a correctness win here. The live
preview and the final pass share a single Parakeet model object, and
``transcribe_stream()`` mutates that shared object (it swaps every encoder
layer's attention module). Running them on one thread makes overlap impossible.

Usage
-----
Wrap any function that touches ``mlx.core`` — directly or through
parakeet-mlx / mlx-whisper / mlx-lm — with :func:`on_mlx_thread`, or call
:func:`run_on_mlx_thread` explicitly. Both block until the work completes and
re-raise its exception with the original traceback.

Callbacks passed into wrapped functions (progress, token streaming, cancel
checks) will be invoked ON the worker thread. Keep them to plain attribute
writes and queue puts — never make one block on this executor, and never let
one call back into MLX from another thread.
"""

from __future__ import annotations

import functools
import queue
import sys
import threading
from concurrent.futures import Future
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_WORKER_NAME = "mlx-runtime"

_queue: queue.SimpleQueue = queue.SimpleQueue()
_start_lock = threading.Lock()
_worker: threading.Thread | None = None
_worker_ident: int | None = None
_ready = threading.Event()


def _pump() -> None:
    global _worker_ident
    # Publish our identity before draining anything, so a reentrant call made
    # from inside the very first task can't fail to recognise this thread.
    _worker_ident = threading.get_ident()
    _ready.set()
    while True:
        future, fn, args, kwargs = _queue.get()
        if not future.set_running_or_notify_cancel():
            continue
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 — propagated to the caller
            future.set_exception(e)
        finally:
            # Drop references promptly; a queued transcription can otherwise pin
            # an hour of audio alive until the next task arrives.
            del future, fn, args, kwargs


def _ensure_worker() -> None:
    global _worker
    if _worker is not None:
        return
    with _start_lock:
        if _worker is not None:
            return
        worker = threading.Thread(target=_pump, name=_WORKER_NAME, daemon=True)
        # Daemon: quitting the app must not block on an in-flight transcription.
        worker.start()
        _ready.wait()
        _worker = worker


def is_mlx_thread() -> bool:
    """True when the caller is already running on the MLX worker."""
    return _worker_ident is not None and threading.get_ident() == _worker_ident


def run_on_mlx_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on the MLX worker thread and return its result.

    Reentrant: when the caller is already the worker, ``fn`` runs inline. Without
    that, a wrapped method calling another wrapped method (``transcribe_file``
    calling ``load``) would enqueue work behind itself and deadlock the single
    worker permanently."""
    if is_mlx_thread():
        return fn(*args, **kwargs)
    _warn_if_main_thread(fn)
    _ensure_worker()
    future: Future = Future()
    _queue.put((future, fn, args, kwargs))
    return future.result()


_warned_main_thread = False


def _warn_if_main_thread(fn: Callable) -> None:
    """MLX work can take minutes (an hour-long transcription). Blocking the main
    thread on it freezes the menu bar, so make that mistake loud instead of
    letting it look like a hang. Warns once, then stays quiet."""
    global _warned_main_thread
    if _warned_main_thread or threading.current_thread() is not threading.main_thread():
        return
    _warned_main_thread = True
    print(
        f"[mlx] WARNING: {getattr(fn, '__qualname__', fn)} was called from the main "
        "thread and will block the UI until the MLX worker is free. Move it to a "
        "background thread.",
        file=sys.stderr,
        flush=True,
    )


def on_mlx_thread(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator form of :func:`run_on_mlx_thread`, for wrapping methods at the
    public boundary so callers keep their own threading model unchanged."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_on_mlx_thread(fn, *args, **kwargs)

    return wrapper

from __future__ import annotations


def main() -> None:
    from app import config

    # Point HuggingFace/transformers caches at Application Support, and disable
    # joblib/loky process-based parallelism — BEFORE importing anything that reads
    # these at import time (mlx, transformers, torch, librosa via parakeet). The
    # joblib/loky setting matters because loky re-execs sys.executable, which in a
    # frozen app is THIS binary → a duplicate menu-bar icon.
    config.configure_hf_env()

    # Single-instance guard: if another copy already owns the menu bar (a stray
    # re-exec, a double launch, etc.), exit before creating a second status item.
    # flock is advisory and auto-released when the process dies, so a crash leaves
    # no stale lock.
    import fcntl

    config.SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    global _instance_lock  # keep the handle alive for the process lifetime
    _instance_lock = open(config.SUPPORT_DIR / ".instance.lock", "w")
    try:
        fcntl.flock(_instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        import sys

        sys.exit(0)

    from AppKit import (
        NSApp,
        NSApplication,
        NSApplicationActivationPolicyAccessory,
    )

    from app.menubar import LocalNotesApp

    NSApplication.sharedApplication()
    NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app = LocalNotesApp()
    app.run()


if __name__ == "__main__":
    # MUST run before anything else in the frozen app: when a bundled library
    # spawns a multiprocessing child, macOS 'spawn' re-execs this binary; without
    # this the child would re-run main() and paint a second menu-bar icon.
    import multiprocessing

    multiprocessing.freeze_support()
    main()

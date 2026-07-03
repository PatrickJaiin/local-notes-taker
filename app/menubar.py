from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from enum import Enum, auto

import rumps
from AppKit import (
    NSApp,
    NSApplicationActivationPolicyAccessory,
    NSApplicationActivationPolicyRegular,
    NSWorkspace,
)
from pynput import keyboard

from app import __version__ as VERSION
from app import config as cfg
from app import storage, summarizer
from app.audio_cleanup import clean_audio
from app.models.manager import ModelManager
from app.output import auto_paste, copy_to_clipboard, show_notification
from app.recorder import Recorder

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

USE_CASES = ["Meeting", "Lecture", "Brainstorm", "Interview", "Stand-up"]

# (label, backend key)
TRANSCRIBERS = [
    ("Parakeet TDT (Default)", cfg.PARAKEET),
    ("Whisper Large V3", cfg.WHISPER),
    ("Granite Speech 8B", cfg.GRANITE),
]

# (label, mlx-community model id)
SUMMARY_MODELS = [
    ("Qwen3 4B (Default)", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
    ("Granite 3.3 8B", "mlx-community/granite-3.3-8b-instruct-4bit"),
    ("Llama 3.2 3B", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
]

LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Hindi", "hi"),
    ("Malayalam", "ml"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("German", "de"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
]


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


def _backend_label(backend: str) -> str:
    for label, key in TRANSCRIBERS:
        if key == backend:
            return label
    return backend




class LocalNotesApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Local Notes", title="📝", quit_button=None)
        self.config = cfg.load_config()
        self.state = State.IDLE
        self.recorder = Recorder()
        self.model_mgr = ModelManager(self.config)

        self._backend = self.config["asr_backend"]
        self._use_case = self.config.get("use_case", USE_CASES[0])
        self._language = self.config.get("language")
        self._summary_model = self.config["summary_model"]

        # Shared state written by worker threads, rendered by _tick on the main thread.
        self._current_step = ""
        self._processing_done = False
        self._processing_cancelled = False
        self._spinner_index = 0
        # Thread-safe FIFO of UI actions to run on the main thread (via _tick).
        # A queue (not a single slot) so concurrent writers can't clobber each
        # other and state-critical actions like _reset are never dropped.
        self._action_q: queue.Queue = queue.Queue()
        self._transcript_so_far = ""
        self._summary_preview = ""
        self._last_summary = ""
        self._prep_status = ""              # e.g. "Downloading model 42%"
        self._download_fraction: float | None = None
        self._download_label = ""
        self._recording_start: float | None = None

        self._flush_stop = threading.Event()
        self._loop_thread: threading.Thread | None = None

        self._build_menu()
        self._start_hotkey_listener()

        # Single poll timer on the main thread for all UI updates.
        self._poll_timer = rumps.Timer(self._tick, 0.15)
        self._poll_timer.start()

        # Pre-warm the active transcriber so the first recording is instant.
        threading.Thread(target=self._warm_backend, daemon=True).start()

    # ------------------------------------------------------------------ menu

    def _build_menu(self) -> None:
        self._record_btn = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self._cancel_btn = rumps.MenuItem("Cancel", callback=self._cancel)
        self._transcript_preview = rumps.MenuItem("Transcript Preview", callback=self._copy_transcript)
        self._paste_last_btn = rumps.MenuItem("Paste Last Summary", callback=self._paste_last_summary)
        self._redo_last_btn = rumps.MenuItem("Redo Last Recording", callback=self._redo_last_recording)

        self._status_item = rumps.MenuItem("Status: …")
        self._status_item.set_callback(None)

        # Live model download/load progress bar (shown only while active).
        self._progress_item = rumps.MenuItem("")
        self._progress_item.set_callback(None)
        self._progress_item.hidden = True

        transcriber_menu = rumps.MenuItem("Transcriber")
        for label, key in TRANSCRIBERS:
            item = rumps.MenuItem(label, callback=self._select_backend)
            if key == self._backend:
                item.state = 1
            transcriber_menu.add(item)

        summary_menu = rumps.MenuItem("Summary Model")
        for label, model_id in SUMMARY_MODELS:
            item = rumps.MenuItem(label, callback=self._select_summary_model)
            if model_id == self._summary_model:
                item.state = 1
            summary_menu.add(item)
        if self._summary_model not in {m for _, m in SUMMARY_MODELS}:
            custom = rumps.MenuItem(self._summary_model, callback=self._select_summary_model_custom)
            custom.state = 1
            summary_menu.add(custom)
        summary_menu.add(rumps.separator)
        summary_menu.add(rumps.MenuItem("Custom…", callback=self._custom_summary_model))

        use_case_menu = rumps.MenuItem("Use Case")
        for uc in USE_CASES:
            item = rumps.MenuItem(uc, callback=self._select_use_case)
            if uc == self._use_case:
                item.state = 1
            use_case_menu.add(item)
        if self._use_case not in USE_CASES:
            custom = rumps.MenuItem(self._use_case, callback=self._select_use_case)
            custom.state = 1
            use_case_menu.add(custom)
        use_case_menu.add(rumps.separator)
        use_case_menu.add(rumps.MenuItem("Custom…", callback=self._custom_use_case))

        lang_menu = rumps.MenuItem("Language")
        for label, code in LANGUAGES:
            item = rumps.MenuItem(label, callback=self._select_language)
            if code == self._language:
                item.state = 1
            lang_menu.add(item)
        if self._language is not None and self._language not in {c for _, c in LANGUAGES}:
            custom = rumps.MenuItem(self._language, callback=self._select_custom_language)
            custom.state = 1
            lang_menu.add(custom)
        lang_menu.add(rumps.separator)
        lang_menu.add(rumps.MenuItem("Other…", callback=self._custom_language))

        self.menu = [
            self._record_btn,
            self._cancel_btn,
            self._progress_item,
            self._transcript_preview,
            self._paste_last_btn,
            self._redo_last_btn,
            rumps.MenuItem("Open Transcripts Folder", callback=self._open_transcripts_folder),
            None,
            transcriber_menu,
            summary_menu,
            use_case_menu,
            lang_menu,
            None,
            self._status_item,
            rumps.MenuItem("Reload Config", callback=self._reload_config),
            None,
            rumps.MenuItem(f"About Local Notes v{VERSION}", callback=self._show_about),
            rumps.MenuItem("Quit", callback=self._quit),
        ]
        self._cancel_btn.hidden = True

    # ----------------------------------------------------------- model prep

    def _progress_cb(self, fraction: float | None, label: str) -> None:
        """Called from background threads during weight download / model load.
        Stores raw data only; _tick (main thread) draws the bar."""
        self._download_fraction = fraction
        self._download_label = label or ""
        self._prep_status = label or ""

    def _clear_prep(self) -> None:
        self._download_fraction = None
        self._download_label = ""
        self._prep_status = ""

    def _notify_async(self, message: str) -> None:
        """Queue a notification to fire on the main thread (safe from any thread)."""
        self._action_q.put(lambda: show_notification("Local Notes", message))

    def _ensure_backend_ready(self):
        """Blocking (background-thread only): download+load the active backend with
        progress, then return the ready transcriber. Returns None on failure."""
        try:
            transcriber = self.model_mgr.ensure_ready(self._backend, self._progress_cb)
            return transcriber
        except Exception as e:
            traceback.print_exc()
            self._notify_async(f"Model: {e}")
            return None
        finally:
            self._clear_prep()

    def _warm_backend(self) -> None:
        """Blocking (background-thread only): get the active backend ready for a fast
        first recording. Granite only gets its weights fetched — the ~20 GB model is
        loaded just-in-time while processing so it never sits resident at idle."""
        if self._backend != cfg.GRANITE:
            self._ensure_backend_ready()
            return
        try:
            self.model_mgr.download(
                cfg.model_id_for_backend(self.config, cfg.GRANITE), self._progress_cb
            )
        except Exception as e:
            traceback.print_exc()
            self._notify_async(f"Model: {e}")
        finally:
            self._clear_prep()

    def _download_summary_model(self) -> None:
        """Ensure the summary LLM weights are present, showing a "Downloading …%"
        status — mlx_lm's own load() would otherwise fetch them silently."""
        try:
            self.model_mgr.download(self._summary_model, self._progress_cb)
        except Exception:
            pass  # summarize() surfaces a clearer error if it truly can't load

    def _render_prep(self):
        """Render active download/load status into the dropdown item and return a
        compact title suffix (or None when idle). Main thread only."""
        label = self._prep_status  # "Downloading X" or "Loading X"
        frac = self._download_fraction
        if not label and frac is None:
            self._progress_item.hidden = True
            return None
        self._progress_item.hidden = False
        if frac is not None:
            pct = int(frac * 100)
            self._progress_item.title = f"{label} — {pct}%"
            verb = label.split(" ", 1)[0]  # "Downloading"
            return f"⬇ {verb} {pct}%"
        # Indeterminate: model loading, or a download just starting up.
        self._progress_item.title = f"{label}…"
        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        return f"{frame} {label}…"

    # ---------------------------------------------------------------- hotkey

    def _start_hotkey_listener(self) -> None:
        # Stop any prior listener so a reload can rebind without two listeners
        # both firing the hotkey.
        existing = getattr(self, "_hotkey_listener", None)
        if existing is not None:
            try:
                existing.stop()
            except Exception:
                pass
        hotkey_str = self.config.get("hotkey", cfg.DEFAULT_CONFIG["hotkey"])
        try:
            hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._on_hotkey)
        except ValueError:
            hotkey_str = cfg.DEFAULT_CONFIG["hotkey"]
            hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._on_hotkey)
            show_notification("Local Notes", f"Invalid hotkey in config. Using default: {hotkey_str}")
        listener = keyboard.Listener(
            on_press=lambda k: hotkey.press(listener.canonical(k)),
            on_release=lambda k: hotkey.release(listener.canonical(k)),
        )
        listener.daemon = True
        listener.start()
        self._hotkey_listener = listener

    def _on_hotkey(self) -> None:
        self._action_q.put(lambda: self.toggle_recording(None))

    # ----------------------------------------------------------- main tick

    def _tick(self, _) -> None:
        """Runs on the main thread: dispatches deferred actions, animates the
        spinner, and renders live preview + download progress."""
        while True:
            try:
                action = self._action_q.get_nowait()
            except queue.Empty:
                break
            action()

        self._spinner_index += 1  # drives both the spinner and the indeterminate bar

        if self.state == State.RECORDING:
            self._render_recording()
            return

        if self.state == State.PROCESSING:
            if self._processing_done:
                self._reset()
                return
            self._render_processing()
            return

        # IDLE — show pre-warm/download progress bar if any.
        prep = self._render_prep()
        self.title = prep if prep else "📝"
        self._status_item.title = self._status_text()

    def _render_recording(self) -> None:
        elapsed = time.time() - self._recording_start if self._recording_start else 0
        duration = storage.format_duration(elapsed)
        word_count = len(self._transcript_so_far.split()) if self._transcript_so_far else 0

        prep = self._render_prep()
        if prep:
            self.title = f"🔴 {duration} {prep}"
            self._transcript_preview.title = self._prep_status
            return

        self.title = f"🔴 {duration} • {word_count}w" if word_count else f"🔴 {duration}"

        if not self._streaming_backend():
            self._transcript_preview.title = "Transcribing on stop…"
        elif self._transcript_so_far:
            snippet = self._transcript_so_far.strip()[-40:]
            if len(self._transcript_so_far.strip()) > 40:
                snippet = "…" + snippet
            self._transcript_preview.title = f"{word_count}w: {snippet}"
        else:
            self._transcript_preview.title = "Listening…"

    def _render_processing(self) -> None:
        prep = self._render_prep()
        if prep:
            self.title = prep
        else:
            frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            self.title = f"{frame} {self._current_step}"
        self._record_btn.title = f"{self._current_step}…"
        if self._summary_preview:
            snippet = self._summary_preview.strip()[-50:]
            self._transcript_preview.title = "Summary: …" + snippet

    def _streaming_backend(self) -> bool:
        return self.model_mgr.backend_supports_streaming(self._backend)

    def _status_text(self) -> str:
        return f"Status: {_backend_label(self._backend)} · idle"

    # ------------------------------------------------------- recording flow

    def toggle_recording(self, sender) -> None:
        if self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()

    def _start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as e:
            show_notification("Local Notes", str(e))
            self._reset()
            return

        self.state = State.RECORDING
        self._recording_start = time.time()
        self.title = "🔴 0s"
        self._record_btn.title = "Stop Recording"
        self._cancel_btn.hidden = False
        self._transcript_so_far = ""
        self._summary_preview = ""
        self._processing_cancelled = False
        # Fresh Event per session: a stale loop thread that outlived its join keeps
        # its own (permanently set) event, so it can never wake into a new session.
        self._flush_stop = threading.Event()

        self._loop_thread = threading.Thread(
            target=self._live_loop, args=(self._flush_stop,), daemon=True
        )
        self._loop_thread.start()

    def _live_loop(self, stop_event: threading.Event) -> None:
        """Background: ensure the model is ready, then drive the live transcript for
        streaming backends. Granite (non-streaming) just waits for stop."""
        transcriber = self._ensure_backend_ready()
        if transcriber is None or stop_event.is_set():
            return
        if not transcriber.supports_streaming:
            return  # Granite: no live preview; transcribed on stop.

        try:
            transcriber.start_stream(self._language)
        except Exception:
            return

        interval = getattr(transcriber, "live_interval", 1.0)
        try:
            while not stop_event.wait(interval):
                pcm = self.recorder.drain_live()
                if pcm is None or len(pcm) == 0:
                    continue
                try:
                    running = transcriber.feed(pcm)
                    if running and not stop_event.is_set():
                        self._transcript_so_far = running
                except Exception:
                    pass
            # Stop requested. No final drain: the recorder is already stopped (mic
            # off first), and the full-quality file pass covers the audio tail.
        finally:
            try:
                transcriber.end_stream()
            except Exception:
                pass

    def _stop_recording(self) -> None:
        self.state = State.PROCESSING
        self._current_step = "Processing"
        self._processing_done = False
        self._spinner_index = 0
        self._record_btn.title = "Processing…"
        threading.Thread(target=self._process, daemon=True).start()

    # ------------------------------------------------------------- process

    def _process(self) -> None:
        audio_path = None
        archived = False
        try:
            self._current_step = "Saving audio"
            self._flush_stop.set()
            # Mic off FIRST — before waiting on the live loop — so nothing said
            # after Stop can end up in the transcript.
            audio_path = self.recorder.stop()
            # Archive immediately: whatever fails later, the recording survives
            # and "Redo Last Recording" can retry it.
            dest = storage.archive_recording(cfg.RECORDINGS_DIR, audio_path, self._use_case)
            if dest is not None:
                audio_path = str(dest)
                archived = True

            if self._loop_thread is not None:
                self._loop_thread.join(timeout=30)
            if self._processing_cancelled:
                return

            duration = (time.time() - self._recording_start) if self._recording_start else None
            self._run_pipeline(audio_path, duration_s=duration)
        except Exception as e:
            if not self._processing_cancelled:
                show_notification("Local Notes", str(e))
        finally:
            # Cancel/exception before stop() completed: make sure the mic is off.
            if self.recorder.is_recording:
                self.recorder.cancel()
            self._processing_done = True
            self._clear_prep()
            if not archived and audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def _run_pipeline(self, audio_path: str, *, duration_s: float | None) -> None:
        """Clean → transcribe → summarize → save → clipboard. Blocking; run from a
        worker thread. The caller owns ``audio_path``; this never deletes it."""
        cleaned_path = None
        try:
            self._current_step = "Cleaning audio"
            try:
                cleaned_path = clean_audio(audio_path)
            except Exception:
                cleaned_path = None

            if self._processing_cancelled:
                return

            self._current_step = "Transcribing"
            transcriber = self._ensure_backend_ready()
            if transcriber is None:
                return
            # Full-quality pass on the cleaned audio. For streaming backends this
            # re-runs audio the live preview already saw — intentionally: the file
            # pass gets the cleaned signal, full context, and the post-drain tail.
            transcript = transcriber.transcribe_file(
                cleaned_path or audio_path, language=self._language
            ).strip()
            if not transcript:
                transcript = self._transcript_so_far.strip()
            if not transcript:
                show_notification("Local Notes", "No speech detected.")
                return
            self._transcript_so_far = transcript

            if self._processing_cancelled:
                return

            self._unload_granite()

            self._current_step = "Summarizing"
            self._summary_preview = ""
            self._download_summary_model()
            try:
                summary = summarizer.summarize(
                    transcript,
                    model_id=self._summary_model,
                    use_case=self._use_case,
                    on_token=lambda t: setattr(self, "_summary_preview", t),
                    cancel_check=lambda: self._processing_cancelled,
                    progress_cb=self._progress_cb,
                )
            except Exception:
                # Never lose the transcript because summarization failed.
                self._save_transcript(transcript, "(summarization failed)", duration_s)
                raise
            self._clear_prep()

            if self._processing_cancelled:
                return

            self._current_step = "Saving"
            self._save_transcript(transcript, summary, duration_s)

            if self._processing_cancelled:
                return

            self._current_step = "Done"
            self._last_summary = summary
            self._deliver_summary(summary)
        finally:
            # RAM handoff: free the heavy Granite model even on cancel/exception,
            # so it can't linger resident while idle.
            self._unload_granite()
            if cleaned_path and os.path.exists(cleaned_path):
                try:
                    os.unlink(cleaned_path)
                except Exception:
                    pass

    def _save_transcript(self, transcript: str, summary: str, duration_s: float | None) -> None:
        storage.save_transcript(
            cfg.TRANSCRIPTS_DIR,
            use_case=self._use_case,
            transcript=transcript,
            summary=summary,
            backend=self._backend,
            model=cfg.model_id_for_backend(self.config, self._backend),
            duration_s=duration_s,
        )

    def _deliver_summary(self, summary: str) -> None:
        """Copy to the clipboard and (if enabled) paste — reporting a paste failure
        honestly instead of claiming success."""
        copy_to_clipboard(summary)
        if not self.config.get("auto_paste", True):
            show_notification("Local Notes", "Summary copied to clipboard!")
            return
        if auto_paste():
            show_notification("Local Notes", "Summary copied and pasted!")
        else:
            show_notification(
                "Local Notes",
                "Summary copied — paste failed. Allow Local Notes under System "
                "Settings > Privacy & Security > Accessibility.",
            )

    def _unload_granite(self) -> None:
        # RAM handoff: the heavy Granite model must never co-reside with the summarizer.
        if self._backend == cfg.GRANITE:
            self.model_mgr.unload(cfg.GRANITE)

    # ----------------------------------------------------- redo / reprocess

    def _redo_last_recording(self, sender) -> None:
        if self.state != State.IDLE:
            show_notification("Local Notes", "Finish the current recording first.")
            return
        recording = storage.newest_recording(cfg.RECORDINGS_DIR)
        if recording is None:
            show_notification("Local Notes", "No saved recording to redo.")
            return

        self.state = State.PROCESSING
        self._current_step = "Cleaning audio"
        self._processing_done = False
        self._processing_cancelled = False
        self._spinner_index = 0
        self._transcript_so_far = ""
        self._summary_preview = ""
        self._record_btn.title = "Processing…"
        self._cancel_btn.hidden = False
        threading.Thread(target=self._reprocess, args=(str(recording),), daemon=True).start()

    def _reprocess(self, audio_path: str) -> None:
        try:
            if not self._processing_cancelled:
                self._run_pipeline(audio_path, duration_s=None)
        except Exception as e:
            if not self._processing_cancelled:
                show_notification("Local Notes", str(e))
        finally:
            self._processing_done = True
            self._clear_prep()

    # -------------------------------------------------------------- cancel

    def _cancel(self, sender) -> None:
        self._processing_cancelled = True
        self._flush_stop.set()
        if self.state == State.RECORDING:
            threading.Thread(target=self._do_cancel_recording, daemon=True).start()
        elif self.state == State.PROCESSING:
            self._current_step = "Cancelling"
            self._cancel_btn.hidden = True

    def _do_cancel_recording(self) -> None:
        # Mic off and audio discarded immediately; only then wait for the live
        # loop, so a slow model download can't keep the mic hot for 10 more seconds.
        self.recorder.cancel()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
        self._action_q.put(self._reset)

    # ----------------------------------------------------- menu callbacks

    def _select_backend(self, sender) -> None:
        for label, key in TRANSCRIBERS:
            if label == sender.title:
                backend = key
                break
        else:
            return
        if backend == self._backend:
            return
        if self.state != State.IDLE:
            # Switching now would unload the model the live loop / final pass is using.
            show_notification("Local Notes", "Finish the current recording first.")
            return

        if backend == cfg.GRANITE:
            warning = self.model_mgr.granite_ram_warning()
            if warning:
                show_notification("Local Notes", warning)

        self._uncheck(self.menu["Transcriber"])
        sender.state = 1
        prev = self._backend
        self._backend = backend
        self._persist({"asr_backend": backend})

        # Free the previously-loaded backend and pre-warm the new one.
        threading.Thread(target=self._switch_backend, args=(prev,), daemon=True).start()

    def _switch_backend(self, prev: str) -> None:
        self.model_mgr.unload(prev)
        self._warm_backend()

    def _select_summary_model(self, sender) -> None:
        for label, model_id in SUMMARY_MODELS:
            if label == sender.title:
                self._set_summary_model(model_id, sender)
                return

    def _check_custom_item(self, menu_title: str, value: str, callback) -> None:
        """Check ``value`` in a submenu, adding it as a new custom item if needed."""
        submenu = self.menu[menu_title]
        self._uncheck(submenu)
        if value not in submenu:
            submenu.add(rumps.MenuItem(value, callback=callback))
        submenu[value].state = 1

    def _custom_summary_model(self, sender) -> None:
        text = self._prompt("Enter an mlx-community model id:", "mlx-community/…")
        if not text:
            return
        self._check_custom_item("Summary Model", text, self._select_summary_model_custom)
        self._set_summary_model(text, None)

    def _select_summary_model_custom(self, sender) -> None:
        self._set_summary_model(sender.title, sender)

    def _set_summary_model(self, model_id: str, sender) -> None:
        if sender is not None:
            self._uncheck(self.menu["Summary Model"])
            sender.state = 1
        if model_id == self._summary_model:
            return
        self._summary_model = model_id
        self._persist({"summary_model": model_id})
        summarizer.unload()  # drop the previously-loaded summary model

    def _select_use_case(self, sender) -> None:
        self._uncheck(self.menu["Use Case"])
        sender.state = 1
        self._use_case = sender.title
        self._persist({"use_case": self._use_case})

    def _custom_use_case(self, sender) -> None:
        text = self._prompt("Enter a custom use case:", "")
        if not text:
            return
        self._check_custom_item("Use Case", text, self._select_use_case)
        self._use_case = text
        self._persist({"use_case": text})

    def _select_language(self, sender) -> None:
        self._uncheck(self.menu["Language"])
        sender.state = 1
        code = next((c for label, c in LANGUAGES if label == sender.title), None)
        self._language = code
        self._persist({"language": code})

    def _custom_language(self, sender) -> None:
        text = self._prompt("Enter a language code (e.g. hi, ml, fr, ta, ko):", "")
        if not text:
            return
        code = text.lower()
        self._check_custom_item("Language", code, self._select_custom_language)
        self._language = code
        self._persist({"language": code})

    def _select_custom_language(self, sender) -> None:
        self._uncheck(self.menu["Language"])
        sender.state = 1
        self._language = sender.title
        self._persist({"language": sender.title})

    def _open_transcripts_folder(self, sender) -> None:
        cfg.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        NSWorkspace.sharedWorkspace().openFile_(str(cfg.TRANSCRIPTS_DIR))

    def _copy_transcript(self, sender) -> None:
        # During summarization the preview row shows the streaming summary, so copy
        # that to match what the user sees; otherwise copy the transcript.
        if self.state == State.PROCESSING and self._summary_preview:
            copy_to_clipboard(self._summary_preview)
            show_notification("Local Notes", "Summary copied to clipboard!")
        elif self._transcript_so_far:
            copy_to_clipboard(self._transcript_so_far)
            show_notification("Local Notes", "Transcript copied to clipboard!")

    def _paste_last_summary(self, sender) -> None:
        if not self._last_summary:
            show_notification("Local Notes", "No summary available yet.")
            return
        self._deliver_summary(self._last_summary)

    def _reload_config(self, sender) -> None:
        old_hotkey = self.config.get("hotkey")
        self.config = cfg.load_config()
        self.model_mgr.update_config(self.config)
        self._backend = self.config["asr_backend"]
        self._use_case = self.config.get("use_case", USE_CASES[0])
        self._language = self.config.get("language")
        self._summary_model = self.config["summary_model"]
        # Rebuild cached models so new model ids / device / chunk size take effect
        # — only when idle, so we never yank a model out mid-recording. In a
        # background thread: unload_all waits on the manager lock, which a
        # mid-download prewarm can hold for minutes.
        if self.state == State.IDLE:
            def _drop_models() -> None:
                self.model_mgr.unload_all()
                summarizer.unload()

            threading.Thread(target=_drop_models, daemon=True).start()
        self._sync_menu_state()
        if self.config.get("hotkey") != old_hotkey:
            self._start_hotkey_listener()  # stops the old listener, binds the new key
        show_notification("Local Notes", "Config reloaded.")

    def _show_about(self, sender) -> None:
        self._run_modal(
            lambda: rumps.alert(
                title="Local Notes",
                message=(
                    f"Version {VERSION}\n\n"
                    "Record, transcribe, and summarize audio locally on Apple Silicon.\n"
                    "Transcription: Parakeet / Whisper / Granite (MLX). Summary: mlx-lm.\n\n"
                    f"Transcriber: {_backend_label(self._backend)}\n"
                    f"Summary model: {self._summary_model}\n"
                    f"Hotkey: {self.config.get('hotkey', 'N/A')}"
                ),
                ok="OK",
            )
        )

    def _quit(self, sender) -> None:
        if self.recorder.is_recording:
            self.recorder.cancel()
        rumps.quit_application()

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _run_modal(fn):
        """Run a modal dialog as a temporarily 'Regular' app, then restore the
        menu-bar-only policy and refocus the previously active app."""
        prev_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        NSApp.activateIgnoringOtherApps_(True)
        try:
            return fn()
        finally:
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            if prev_app:
                prev_app.activateWithOptions_(1 << 1)

    def _prompt(self, message: str, default_text: str) -> str:
        """Show a modal text-entry dialog and return the stripped response (or "")."""
        window = rumps.Window(
            message=message, title="Local Notes", default_text=default_text, ok="Set", cancel="Cancel"
        )
        response = self._run_modal(window.run)
        if not response.clicked or not response.text.strip():
            return ""
        return response.text.strip()

    def _persist(self, updates: dict) -> None:
        self.config.update(updates)
        self.model_mgr.update_config(self.config)
        try:
            cfg.save_config(self.config)
        except Exception:
            pass

    @staticmethod
    def _uncheck(submenu) -> None:
        for item in submenu.values():
            if isinstance(item, rumps.MenuItem):
                item.state = 0

    def _sync_menu_state(self) -> None:
        """Refresh all submenu checkmarks to match the active config values
        (re-listing a custom value as a checked item if it isn't a built-in)."""
        self._check_title(self.menu["Transcriber"], _backend_label(self._backend))
        sm_label = next((l for l, m in SUMMARY_MODELS if m == self._summary_model), self._summary_model)
        self._check_title(self.menu["Summary Model"], sm_label, self._select_summary_model_custom)
        self._check_title(self.menu["Use Case"], self._use_case, self._select_use_case)
        lang_label = next((l for l, c in LANGUAGES if c == self._language), self._language or "Auto-detect")
        self._check_title(
            self.menu["Language"],
            lang_label,
            self._select_custom_language if self._language else None,
        )

    @staticmethod
    def _check_title(submenu, title, custom_callback=None) -> None:
        """Check the item whose title matches; uncheck the rest. If none matches and
        a callback is given, add the value as a new checked custom item."""
        found = False
        for item in submenu.values():
            if isinstance(item, rumps.MenuItem):
                if item.title == title:
                    item.state = 1
                    found = True
                else:
                    item.state = 0
        if not found and title and custom_callback is not None:
            extra = rumps.MenuItem(title, callback=custom_callback)
            extra.state = 1
            submenu.add(extra)

    def _reset(self) -> None:
        self.state = State.IDLE
        self.title = "📝"
        self._record_btn.title = "Start Recording"
        self._cancel_btn.hidden = True
        self._transcript_preview.title = "Transcript Preview"
        self._recording_start = None
        self._summary_preview = ""
        self._clear_prep()

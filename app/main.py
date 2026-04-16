import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import rumps
import yaml
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSApplicationActivationPolicyRegular,
    NSWorkspace,
)
from pynput import keyboard

from app.audio_cleanup import clean_audio
from app.ollama_manager import OllamaManager
from app.output import auto_paste, copy_to_clipboard, show_notification
from app.recorder import Recorder
from app.summarizer import summarize
from app.transcriber import transcribe

VERSION = "0.1.0"

if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys._MEIPASS)
    _DATA_DIR = Path.home() / "Library" / "Application Support" / "Local Notes"
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent
    _DATA_DIR = _BASE_DIR

DEFAULT_CONFIG = {
    "hotkey": "<cmd>+<shift>+i",
    "whisper_model": "large-v3-turbo",
    "ollama_model": "qwen3:8b",
    "ollama_mode": "external",
    "language": None,
    "auto_paste": True,
}

CONFIG_PATH = _DATA_DIR / "config.yaml" if getattr(sys, "frozen", False) else _BASE_DIR / "config.yaml"
TRANSCRIPTS_DIR = _DATA_DIR / "transcripts"
RECORDINGS_DIR = _DATA_DIR / "recordings"
KEEP_RECORDINGS = 2

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

USE_CASES = ["Meeting", "Lecture", "Brainstorm", "Interview", "Stand-up"]

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


def _ensure_config_file() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return

    if not getattr(sys, "frozen", False):
        CONFIG_PATH.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
        return

    bundled_config = _BASE_DIR / "config.yaml"
    if bundled_config.exists():
        CONFIG_PATH.write_text(bundled_config.read_text(encoding="utf-8"), encoding="utf-8")
        return

    CONFIG_PATH.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")


def load_config() -> dict:
    _ensure_config_file()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    config = {**DEFAULT_CONFIG, **loaded}
    mode = config.get("ollama_mode")
    if mode not in {"external", "bundled"}:
        config["ollama_mode"] = DEFAULT_CONFIG["ollama_mode"]
    if config.get("language") == "":
        config["language"] = None
    return config


def _slugify_filename(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    slug = slug.strip("-")
    return slug or "note"


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class LocalNotesApp(rumps.App):
    def __init__(self):
        super().__init__("Local Notes", title="📝", quit_button=None)
        self.config = load_config()
        self.state = State.IDLE
        self.recorder = Recorder()

        # Ollama lifecycle
        ollama_mode = self.config.get("ollama_mode", "external")
        self._ollama = OllamaManager(mode=ollama_mode)
        self._ollama_host: str | None = None

        self._use_case = USE_CASES[0]
        self._language = self.config.get("language")
        self._current_step = ""
        self._processing_done = False
        self._processing_cancelled = False
        self._spinner_index = 0
        self._pending_action = None
        self._transcript_so_far = ""
        self._last_summary = ""
        self._flush_stop = threading.Event()
        self._incremental_temps: list[str] = []
        self._recording_start: float | None = None

        # --- Build menu ---

        # Record button
        self._record_btn = rumps.MenuItem("Start Recording", callback=self.toggle_recording)

        # Cancel button (hidden until recording/processing)
        self._cancel_btn = rumps.MenuItem("Cancel", callback=self._cancel_processing)

        # Transcript preview
        self._transcript_preview = rumps.MenuItem(
            "Transcript Preview", callback=self._copy_transcript,
        )

        # Paste last summary
        self._paste_last_btn = rumps.MenuItem("Paste Last Summary", callback=self._paste_last_summary)

        # Redo last recording (clean + re-transcribe + re-summarize)
        self._redo_last_btn = rumps.MenuItem("Redo Last Recording", callback=self._redo_last_recording)

        # Ollama status
        self._ollama_status = rumps.MenuItem("Ollama: Checking...")
        self._ollama_status.set_callback(None)

        # Build use-case submenu with checkmarks
        use_case_menu = rumps.MenuItem("Use Case")
        for uc in USE_CASES:
            item = rumps.MenuItem(uc, callback=self._select_use_case)
            if uc == self._use_case:
                item.state = 1
            use_case_menu.add(item)
        use_case_menu.add(rumps.separator)
        use_case_menu.add(rumps.MenuItem("Custom...", callback=self._custom_use_case))

        # Build language submenu with checkmarks
        lang_menu = rumps.MenuItem("Language")
        for label, code in LANGUAGES:
            item = rumps.MenuItem(label, callback=self._select_language)
            if code == self._language:
                item.state = 1
            lang_menu.add(item)
        lang_menu.add(rumps.separator)
        lang_menu.add(rumps.MenuItem("Other...", callback=self._custom_language))

        self.menu = [
            self._record_btn,
            self._cancel_btn,
            self._transcript_preview,
            self._paste_last_btn,
            self._redo_last_btn,
            rumps.MenuItem("Open Transcripts Folder", callback=self._open_transcripts_folder),
            None,
            use_case_menu,
            lang_menu,
            None,
            self._ollama_status,
            rumps.MenuItem("Reload Config", callback=self._reload_config),
            None,
            rumps.MenuItem(f"About Local Notes v{VERSION}", callback=self._show_about),
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        # Hide cancel button initially
        self._cancel_btn.hidden = True

        self._start_hotkey_listener()

        # Single poll timer on main thread for all UI updates
        self._poll_timer = rumps.Timer(self._tick, 0.15)
        self._poll_timer.start()

        # Start Ollama in background
        threading.Thread(target=self._start_ollama, daemon=True).start()

    # --- Ollama lifecycle ---

    def _start_ollama(self):
        try:
            self._ollama_host = self._ollama.start()
        except Exception as e:
            show_notification("Local Notes", f"Ollama: {e}")

        # Retry connection check a few times — Ollama may still be starting
        for _ in range(5):
            if self._ollama.check_connection():
                break
            time.sleep(2)

        self._pending_action = self._update_ollama_status

    def _update_ollama_status(self):
        if self._ollama.is_ready:
            model = self.config.get("ollama_model", "qwen3:8b")
            self._ollama_status.title = f"Ollama: Connected ({model})"
        else:
            self._ollama_status.title = "Ollama: Not connected"

    # --- Hotkey ---

    def _start_hotkey_listener(self):
        hotkey_str = self.config.get("hotkey", "<cmd>+<shift>+i")
        try:
            hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._on_hotkey)
        except ValueError:
            hotkey_str = DEFAULT_CONFIG["hotkey"]
            hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._on_hotkey)
            show_notification("Local Notes", f"Invalid hotkey in config. Using default: {hotkey_str}")
        listener = keyboard.Listener(
            on_press=lambda k: hotkey.press(listener.canonical(k)),
            on_release=lambda k: hotkey.release(listener.canonical(k)),
        )
        listener.daemon = True
        listener.start()

    def _on_hotkey(self):
        self._pending_action = lambda: self.toggle_recording(None)

    # --- Menu callbacks ---

    def _select_use_case(self, sender):
        self._uncheck_all_use_cases()
        sender.state = 1
        self._use_case = sender.title

    def _custom_use_case(self, sender):
        prev_app = NSWorkspace.sharedWorkspace().frontmostApplication()

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        NSApp.activateIgnoringOtherApps_(True)

        window = rumps.Window(
            message="Enter a custom use case:",
            title="Local Notes",
            default_text="",
            ok="Set",
            cancel="Cancel",
        )
        response = window.run()

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        if prev_app:
            prev_app.activateWithOptions_(1 << 1)

        if not response.clicked or not response.text.strip():
            return

        custom_text = response.text.strip()
        self._uncheck_all_use_cases()
        self._use_case = custom_text

        submenu = self.menu["Use Case"]
        if custom_text not in submenu:
            submenu.add(rumps.MenuItem(custom_text, callback=self._select_use_case))
        submenu[custom_text].state = 1

    def _select_language(self, sender):
        self._uncheck_all_languages()
        sender.state = 1
        code = None
        for label, c in LANGUAGES:
            if label == sender.title:
                code = c
                break
        self._language = code

    def _custom_language(self, sender):
        prev_app = NSWorkspace.sharedWorkspace().frontmostApplication()

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        NSApp.activateIgnoringOtherApps_(True)

        window = rumps.Window(
            message="Enter a language code (e.g. hi, ml, fr, ta, ko):",
            title="Local Notes",
            default_text="",
            ok="Set",
            cancel="Cancel",
        )
        response = window.run()

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        if prev_app:
            prev_app.activateWithOptions_(1 << 1)

        if not response.clicked or not response.text.strip():
            return

        code = response.text.strip().lower()
        self._uncheck_all_languages()
        self._language = code

        submenu = self.menu["Language"]
        if code not in submenu:
            submenu.add(rumps.MenuItem(code, callback=self._select_custom_language))
        submenu[code].state = 1

    def _select_custom_language(self, sender):
        self._uncheck_all_languages()
        sender.state = 1
        self._language = sender.title

    def _open_transcripts_folder(self, sender):
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        NSWorkspace.sharedWorkspace().openFile_(str(TRANSCRIPTS_DIR))

    def _reload_config(self, sender):
        self.config = load_config()
        show_notification("Local Notes", "Config reloaded.")
        self._update_ollama_status()

    def _show_about(self, sender):
        prev_app = NSWorkspace.sharedWorkspace().frontmostApplication()

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        NSApp.activateIgnoringOtherApps_(True)

        rumps.alert(
            title="Local Notes",
            message=(
                f"Version {VERSION}\n\n"
                "Record, transcribe, and summarize audio locally.\n"
                "Powered by Whisper and Ollama.\n\n"
                f"Hotkey: {self.config.get('hotkey', 'N/A')}\n"
                f"Whisper model: {self.config.get('whisper_model', 'N/A')}\n"
                f"Ollama model: {self.config.get('ollama_model', 'N/A')}"
            ),
            ok="OK",
        )

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        if prev_app:
            prev_app.activateWithOptions_(1 << 1)

    def _copy_transcript(self, sender):
        if self._transcript_so_far:
            copy_to_clipboard(self._transcript_so_far)
            show_notification("Local Notes", "Transcript copied to clipboard!")

    def _redo_last_recording(self, sender):
        if self.state != State.IDLE:
            show_notification("Local Notes", "Finish the current recording first.")
            return

        recording = self._newest_recording()
        if recording is None:
            show_notification("Local Notes", "No saved recording to redo.")
            return

        self.state = State.PROCESSING
        self._current_step = "Cleaning audio"
        self._processing_done = False
        self._processing_cancelled = False
        self._spinner_index = 0
        self._transcript_so_far = ""
        self._record_btn.title = "Processing..."
        self._cancel_btn.hidden = False

        threading.Thread(
            target=self._reprocess, args=(str(recording),), daemon=True
        ).start()

    def _newest_recording(self) -> Path | None:
        if not RECORDINGS_DIR.exists():
            return None
        wavs = sorted(
            RECORDINGS_DIR.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return wavs[0] if wavs else None

    def _reprocess(self, audio_path: str):
        cleaned_path = None
        try:
            if self._processing_cancelled:
                return

            self._current_step = "Cleaning audio"
            try:
                cleaned_path = clean_audio(audio_path)
            except Exception:
                cleaned_path = None

            if self._processing_cancelled:
                return

            self._current_step = "Transcribing"
            whisper_model = self.config.get("whisper_model", "base")
            transcript = transcribe(
                cleaned_path or audio_path,
                model_size=whisper_model,
                language=self._language,
            ).strip()

            if not transcript:
                show_notification("Local Notes", "No speech detected.")
                return

            self._transcript_so_far = transcript

            if self._processing_cancelled:
                return

            self._current_step = "Connecting to Ollama"
            ollama_model = self.config.get("ollama_model", "qwen3:8b")
            self._ollama.ensure_model(ollama_model)

            if self._processing_cancelled:
                return

            self._current_step = "Summarizing"
            summary = summarize(
                transcript,
                model=ollama_model,
                use_case=self._use_case,
                host=self._ollama_host,
            )

            if self._processing_cancelled:
                return

            self._current_step = "Saving"
            self._save_transcript(transcript, summary)

            self._current_step = "Done"
            self._last_summary = summary
            copy_to_clipboard(summary)

            if self.config.get("auto_paste", True):
                show_notification("Local Notes", "Summary copied and pasted!")
                auto_paste()
            else:
                show_notification("Local Notes", "Summary copied to clipboard!")

        except Exception as e:
            if not self._processing_cancelled:
                show_notification("Local Notes", str(e))
        finally:
            self._processing_done = True
            if cleaned_path:
                try:
                    os.unlink(cleaned_path)
                except Exception:
                    pass

    def _paste_last_summary(self, sender):
        if not self._last_summary:
            show_notification("Local Notes", "No summary available yet.")
            return
        copy_to_clipboard(self._last_summary)
        if self.config.get("auto_paste", True):
            auto_paste()
            show_notification("Local Notes", "Last summary pasted!")
        else:
            show_notification("Local Notes", "Last summary copied to clipboard!")

    def _cancel_processing(self, sender):
        """Cancel an in-progress recording or processing.
        Sets flags only — actual cleanup happens on the background/tick threads
        to avoid blocking the main thread."""
        self._processing_cancelled = True
        self._flush_stop.set()
        if self.state == State.RECORDING:
            threading.Thread(target=self._do_cancel_recording, daemon=True).start()
        elif self.state == State.PROCESSING:
            self._current_step = "Cancelling"
            self._cancel_btn.hidden = True

    def _do_cancel_recording(self):
        """Background cleanup for a cancelled recording."""
        self.recorder.cancel()
        self._cleanup_temps()
        self._pending_action = self._reset

    def _quit(self, sender):
        if self.recorder.is_recording:
            self.recorder.cancel()
        rumps.quit_application()

    # --- Helpers ---

    def _uncheck_all_languages(self):
        for item in self.menu["Language"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 0

    def _uncheck_all_use_cases(self):
        for item in self.menu["Use Case"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 0

    # --- Main tick loop ---

    def _tick(self, _):
        """Runs on main thread: dispatches hotkey actions, animates spinner,
        updates live preview, and detects processing completion."""
        action = self._pending_action
        if action is not None:
            self._pending_action = None
            action()

        if self.state == State.RECORDING:
            elapsed = time.time() - self._recording_start if self._recording_start else 0
            word_count = len(self._transcript_so_far.split()) if self._transcript_so_far else 0
            duration = _format_duration(elapsed)

            if word_count:
                self.title = f"🔴 {duration} \u2022 {word_count}w"
            else:
                self.title = f"🔴 {duration}"

            if self._transcript_so_far:
                snippet = self._transcript_so_far.strip()[-40:]
                if len(self._transcript_so_far.strip()) > 40:
                    snippet = "\u2026" + snippet
                self._transcript_preview.title = f"{word_count}w: {snippet}"
            else:
                self._transcript_preview.title = "Listening..."
            return

        if self.state != State.PROCESSING:
            return

        if self._processing_done:
            self._reset()
            return

        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        self.title = f"{frame} {self._current_step}"
        self._record_btn.title = f"{self._current_step}..."
        self._spinner_index += 1

    # --- Recording flow ---

    def toggle_recording(self, sender):
        if self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()

    def _start_recording(self):
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
        self._flush_stop.clear()
        self._processing_cancelled = False
        self._incremental_temps = []
        threading.Thread(target=self._incremental_transcribe_loop, daemon=True).start()

    def _stop_recording(self):
        self.state = State.PROCESSING
        self._current_step = "Processing"
        self._processing_done = False
        self._spinner_index = 0
        self._record_btn.title = "Processing..."

        thread = threading.Thread(target=self._process, daemon=True)
        thread.start()

    def _incremental_transcribe_loop(self):
        whisper_model = self.config.get("whisper_model", "base")
        while not self._flush_stop.wait(10):
            path = self.recorder.flush()
            if path is None:
                continue
            try:
                chunk_text = transcribe(path, model_size=whisper_model, language=self._language)
                if chunk_text.strip():
                    self._transcript_so_far += (" " + chunk_text if self._transcript_so_far else chunk_text)
            except Exception:
                pass
            finally:
                self._incremental_temps.append(path)

    # --- Processing ---

    def _process(self):
        audio_path = None
        cleaned_path = None
        try:
            self._flush_stop.set()

            if self._processing_cancelled:
                return

            self._current_step = "Saving audio"
            audio_path = self.recorder.stop()

            if self._processing_cancelled:
                return

            self._current_step = "Cleaning audio"
            try:
                cleaned_path = clean_audio(audio_path)
            except Exception:
                cleaned_path = None

            if self._processing_cancelled:
                return

            self._current_step = "Transcribing"
            whisper_model = self.config.get("whisper_model", "base")
            tail_text = transcribe(
                cleaned_path or audio_path,
                model_size=whisper_model,
                language=self._language,
            )
            if tail_text.strip():
                self._transcript_so_far += (" " + tail_text if self._transcript_so_far else tail_text)

            transcript = self._transcript_so_far.strip()
            if not transcript:
                show_notification("Local Notes", "No speech detected.")
                return

            if self._processing_cancelled:
                return

            self._current_step = "Connecting to Ollama"
            ollama_model = self.config.get("ollama_model", "qwen3:8b")
            self._ollama.ensure_model(ollama_model)

            if self._processing_cancelled:
                return

            self._current_step = "Summarizing"
            summary = summarize(transcript, model=ollama_model, use_case=self._use_case, host=self._ollama_host)

            if self._processing_cancelled:
                return

            self._current_step = "Saving"
            self._save_transcript(transcript, summary)
            self._archive_recording(audio_path)

            self._current_step = "Done"
            self._last_summary = summary
            copy_to_clipboard(summary)

            if self.config.get("auto_paste", True):
                show_notification("Local Notes", "Summary copied and pasted!")
                auto_paste()
            else:
                show_notification("Local Notes", "Summary copied to clipboard!")

        except Exception as e:
            if not self._processing_cancelled:
                show_notification("Local Notes", str(e))
        finally:
            self._processing_done = True
            self._cleanup_temps()
            if cleaned_path:
                try:
                    os.unlink(cleaned_path)
                except Exception:
                    pass
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def _archive_recording(self, audio_path: str) -> str | None:
        """Move the WAV into recordings/ and prune to the newest KEEP_RECORDINGS."""
        try:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            slug = _slugify_filename(self._use_case)
            dest = RECORDINGS_DIR / f"{ts}_{slug}.wav"
            shutil.move(audio_path, dest)
        except Exception:
            return None

        try:
            recordings = sorted(
                RECORDINGS_DIR.glob("*.wav"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in recordings[KEEP_RECORDINGS:]:
                try:
                    old.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        return str(dest)

    def _save_transcript(self, transcript: str, summary: str):
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        use_case_slug = _slugify_filename(self._use_case)
        filepath = TRANSCRIPTS_DIR / f"{ts}_{use_case_slug}.txt"

        duration = ""
        if self._recording_start:
            elapsed = time.time() - self._recording_start
            duration = f"Duration: {_format_duration(elapsed)}\n"

        word_count = len(transcript.split())
        filepath.write_text(
            f"Use Case: {self._use_case}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{duration}"
            f"Words: {word_count}\n"
            f"{'=' * 40}\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"{'=' * 40}\n\n"
            f"SUMMARY:\n{summary}\n",
            encoding="utf-8",
        )

    def _cleanup_temps(self):
        for tmp in self._incremental_temps:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        self._incremental_temps = []

    def _reset(self):
        self.state = State.IDLE
        self.title = "📝"
        self._record_btn.title = "Start Recording"
        self._cancel_btn.hidden = True
        self._transcript_preview.title = "Transcript Preview"
        self._recording_start = None


def main():
    NSApplication.sharedApplication()
    NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app = LocalNotesApp()
    try:
        app.run()
    finally:
        app._ollama.stop()


if __name__ == "__main__":
    main()

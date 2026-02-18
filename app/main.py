import os
import threading
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import rumps
import yaml
from AppKit import NSApp, NSApplicationActivationPolicyAccessory, NSApplicationActivationPolicyRegular, NSWorkspace
from pynput import keyboard

from app.ollama_manager import OllamaManager
from app.output import auto_paste, copy_to_clipboard, show_notification
from app.recorder import Recorder
from app.summarizer import summarize
from app.transcriber import transcribe

import sys as _sys

if getattr(_sys, "frozen", False):
    _BASE_DIR = Path(_sys._MEIPASS)
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = _BASE_DIR / "config.yaml"
TRANSCRIPTS_DIR = _BASE_DIR / "transcripts"

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


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class LocalNotesApp(rumps.App):
    def __init__(self):
        super().__init__("Local Notes", title="📝")
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
        self._spinner_index = 0
        self._pending_action = None
        self._transcript_so_far = ""
        self._flush_stop = threading.Event()
        self._incremental_temps: list[str] = []

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

        self._transcript_preview = rumps.MenuItem(
            "Transcript Preview", callback=self._copy_transcript,
        )

        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            self._transcript_preview,
            None,
            use_case_menu,
            lang_menu,
            None,
        ]
        self._start_hotkey_listener()
        # Single poll timer created on main thread — guarantees all UI
        # updates happen on the main thread, avoiding AppKit crashes.
        self._poll_timer = rumps.Timer(self._tick, 0.15)
        self._poll_timer.start()

        # Start bundled Ollama in background so it's ready when needed
        threading.Thread(target=self._start_ollama, daemon=True).start()

    def _start_ollama(self):
        try:
            self._ollama_host = self._ollama.start()
        except Exception as e:
            show_notification("Local Notes", f"Ollama start failed: {e}")

    def _start_hotkey_listener(self):
        hotkey_str = self.config.get("hotkey", "<cmd>+<shift>+n")
        hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._on_hotkey)
        listener = keyboard.Listener(
            on_press=lambda k: hotkey.press(listener.canonical(k)),
            on_release=lambda k: hotkey.release(listener.canonical(k)),
        )
        listener.daemon = True
        listener.start()

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

        # Add to menu if it's new, and check it
        submenu = self.menu["Use Case"]
        if custom_text not in submenu:
            submenu.add(rumps.MenuItem(custom_text, callback=self._select_use_case))
        submenu[custom_text].state = 1

    def _select_language(self, sender):
        self._uncheck_all_languages()
        sender.state = 1
        # Look up the code for this label
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

        # Add to menu if it's new, and check it
        submenu = self.menu["Language"]
        if code not in submenu:
            submenu.add(rumps.MenuItem(code, callback=self._select_custom_language))
        submenu[code].state = 1

    def _select_custom_language(self, sender):
        self._uncheck_all_languages()
        sender.state = 1
        self._language = sender.title

    def _uncheck_all_languages(self):
        for item in self.menu["Language"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 0

    def _uncheck_all_use_cases(self):
        for item in self.menu["Use Case"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 0

    def _on_hotkey(self):
        self._pending_action = lambda: self.toggle_recording(None)

    def _copy_transcript(self, sender):
        if self._transcript_so_far:
            copy_to_clipboard(self._transcript_so_far)
            show_notification("Local Notes", "Transcript copied to clipboard!")

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
            finally:
                self._incremental_temps.append(path)

    def _tick(self, _):
        """Runs on main thread: dispatches hotkey actions, animates spinner,
        updates live preview, and detects processing completion."""
        action = self._pending_action
        if action is not None:
            self._pending_action = None
            action()

        if self.state == State.RECORDING:
            word_count = len(self._transcript_so_far.split()) if self._transcript_so_far else 0
            self.title = f"🔴 ({word_count} words)" if word_count else "🔴"
            preview = self._transcript_so_far[-200:] if self._transcript_so_far else "(listening...)"
            self._transcript_preview.title = f"Preview: {preview}"
            return

        if self.state != State.PROCESSING:
            return

        if self._processing_done:
            self._reset()
            return

        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        self.title = f"{frame} {self._current_step}"
        self.menu["Start Recording"].title = f"{self._current_step}..."
        self._spinner_index += 1

    def toggle_recording(self, sender):
        if self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()

    def _start_recording(self):
        self.state = State.RECORDING
        self.title = "🔴"
        self.menu["Start Recording"].title = "Stop Recording"
        self._transcript_so_far = ""
        self._flush_stop.clear()
        self._incremental_temps = []
        self.recorder.start()
        threading.Thread(target=self._incremental_transcribe_loop, daemon=True).start()

    def _stop_recording(self):
        self.state = State.PROCESSING
        self._current_step = "Processing"
        self._processing_done = False
        self._spinner_index = 0

        thread = threading.Thread(target=self._process, daemon=True)
        thread.start()

    def _save_transcript(self, transcript: str, summary: str):
        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        use_case_slug = self._use_case.lower().replace(" ", "-")
        filepath = TRANSCRIPTS_DIR / f"{ts}_{use_case_slug}.txt"
        filepath.write_text(
            f"Use Case: {self._use_case}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 40}\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"{'=' * 40}\n\n"
            f"SUMMARY:\n{summary}\n"
        )

    def _process(self):
        audio_path = None
        try:
            # Stop the incremental loop and flush the tail
            self._flush_stop.set()

            self._current_step = "Saving audio"
            audio_path = self.recorder.stop()

            # Transcribe only the remaining tail audio
            self._current_step = "Transcribing tail"
            whisper_model = self.config.get("whisper_model", "base")
            tail_text = transcribe(audio_path, model_size=whisper_model, language=self._language)
            if tail_text.strip():
                self._transcript_so_far += (" " + tail_text if self._transcript_so_far else tail_text)

            transcript = self._transcript_so_far.strip()
            if not transcript:
                show_notification("Local Notes", "No speech detected.")
                return

            self._current_step = "Summarizing"
            ollama_model = self.config.get("ollama_model", "qwen3:8b")
            summary = summarize(transcript, model=ollama_model, use_case=self._use_case, host=self._ollama_host)

            self._current_step = "Saving"
            self._save_transcript(transcript, summary)

            self._current_step = "Copying"
            copy_to_clipboard(summary)
            show_notification("Local Notes", "Summary copied to clipboard!")
            auto_paste()

        except Exception as e:
            show_notification("Local Notes - Error", str(e))
        finally:
            self._processing_done = True
            for tmp in self._incremental_temps:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
            if audio_path:
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def _reset(self):
        self.state = State.IDLE
        self.title = "📝"
        self.menu["Start Recording"].title = "Start Recording"
        self._transcript_preview.title = "Transcript Preview"


def main():
    app = LocalNotesApp()
    try:
        app.run()
    finally:
        app._ollama.stop()


if __name__ == "__main__":
    main()

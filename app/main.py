import os
import threading
import tkinter as tk
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from tkinter import simpledialog

import pystray
import yaml
from PIL import Image, ImageDraw
from pynput import keyboard

from app.ollama_manager import OllamaManager
from app.output import auto_paste, copy_to_clipboard, show_notification
from app.recorder import Recorder
from app.summarizer import summarize
from app.transcriber import transcribe

import sys as _sys

if getattr(_sys, "frozen", False):
    _BASE_DIR = Path(_sys._MEIPASS)
    _DATA_DIR = Path.home() / "AppData" / "Local" / "Local Notes"
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent
    _DATA_DIR = _BASE_DIR

CONFIG_PATH = _BASE_DIR / "config.yaml"
TRANSCRIPTS_DIR = _DATA_DIR / "transcripts"

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

ICON_SIZE = 64


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


def _make_icon(color: str) -> Image.Image:
    """Generate a colored circle icon for the system tray."""
    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, ICON_SIZE - 4, ICON_SIZE - 4], fill=color)
    return img


ICON_IDLE = _make_icon("#4A90D9")      # blue
ICON_RECORDING = _make_icon("#D94A4A")  # red
ICON_PROCESSING = _make_icon("#D9A04A") # orange


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class LocalNotesApp:
    def __init__(self):
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
        self._pending_action = None
        self._custom_use_cases: list[str] = []
        self._custom_languages: list[tuple[str, str]] = []
        self._stop_event = threading.Event()
        self._transcript_so_far = ""
        self._flush_stop = threading.Event()
        self._incremental_temps: list[str] = []

        self._icon = pystray.Icon(
            "local-notes",
            icon=ICON_IDLE,
            title="Local Notes - Idle",
            menu=pystray.Menu(self._build_menu),
        )

        self._start_hotkey_listener()

        # Start bundled Ollama in background so it's ready when needed
        threading.Thread(target=self._start_ollama, daemon=True).start()

    def _start_ollama(self):
        try:
            self._ollama_host = self._ollama.start()
        except Exception as e:
            show_notification("Local Notes", f"Ollama start failed: {e}")

    def _build_menu(self) -> list:
        """Dynamically build the tray menu (called each time the menu opens)."""
        # Toggle recording item - default=True makes left-click trigger it
        if self.state == State.IDLE:
            rec_label = "Start Recording"
        elif self.state == State.RECORDING:
            rec_label = "Stop Recording"
        else:
            rec_label = f"{self._current_step}..."

        # Transcript preview as a submenu with chunked lines
        preview_items = []
        if self._transcript_so_far:
            words = self._transcript_so_far.split()
            word_count = len(words)
            # Show last ~300 chars, split into ~60-char lines
            tail = self._transcript_so_far[-300:]
            lines = [tail[i:i + 60] for i in range(0, len(tail), 60)]
            for line in lines:
                preview_items.append(pystray.MenuItem(line, None, enabled=False))
            preview_items.append(pystray.Menu.SEPARATOR)
            preview_items.append(pystray.MenuItem(
                f"Copy transcript ({word_count} words)",
                self._on_copy_transcript,
            ))
            preview_label = f"Preview ({word_count} words)"
        elif self.state == State.RECORDING:
            preview_items.append(pystray.MenuItem("(listening...)", None, enabled=False))
            preview_label = "Preview"
        else:
            preview_items.append(pystray.MenuItem("No transcript yet", None, enabled=False))
            preview_label = "Preview"

        items = [
            pystray.MenuItem(rec_label, self._on_toggle_recording, default=True),
            pystray.MenuItem(preview_label, pystray.Menu(*preview_items)),
            pystray.Menu.SEPARATOR,
        ]

        # Use Case submenu
        uc_items = []
        all_use_cases = USE_CASES + self._custom_use_cases
        for uc in all_use_cases:
            uc_items.append(pystray.MenuItem(
                uc,
                self._make_use_case_callback(uc),
                checked=lambda item, u=uc: self._use_case == u,
            ))
        uc_items.append(pystray.Menu.SEPARATOR)
        uc_items.append(pystray.MenuItem("Custom...", self._on_custom_use_case))
        items.append(pystray.MenuItem("Use Case", pystray.Menu(*uc_items)))

        # Language submenu
        lang_items = []
        all_languages = list(LANGUAGES) + self._custom_languages
        for label, code in all_languages:
            lang_items.append(pystray.MenuItem(
                label,
                self._make_language_callback(code),
                checked=lambda item, c=code: self._language == c,
            ))
        lang_items.append(pystray.Menu.SEPARATOR)
        lang_items.append(pystray.MenuItem("Other...", self._on_custom_language))
        items.append(pystray.MenuItem("Language", pystray.Menu(*lang_items)))

        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Quit", self._on_quit))

        return items

    def _make_use_case_callback(self, uc: str):
        def callback(icon, item):
            self._use_case = uc
        return callback

    def _make_language_callback(self, code):
        def callback(icon, item):
            self._language = code
        return callback

    def _on_copy_transcript(self, icon, item):
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
                    # Update tooltip with word count
                    word_count = len(self._transcript_so_far.split())
                    self._icon.title = f"Local Notes - Recording ({word_count} words)"
            finally:
                self._incremental_temps.append(path)

    def _on_toggle_recording(self, icon, item):
        if self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()

    def _on_custom_use_case(self, icon, item):
        thread = threading.Thread(target=self._show_custom_use_case_dialog, daemon=True)
        thread.start()

    def _show_custom_use_case_dialog(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = simpledialog.askstring(
            "Local Notes",
            "Enter a custom use case:",
            parent=root,
        )
        root.destroy()
        if result and result.strip():
            custom_text = result.strip()
            self._use_case = custom_text
            if custom_text not in USE_CASES and custom_text not in self._custom_use_cases:
                self._custom_use_cases.append(custom_text)

    def _on_custom_language(self, icon, item):
        thread = threading.Thread(target=self._show_custom_language_dialog, daemon=True)
        thread.start()

    def _show_custom_language_dialog(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = simpledialog.askstring(
            "Local Notes",
            "Enter a language code (e.g. hi, ml, fr, ta, ko):",
            parent=root,
        )
        root.destroy()
        if result and result.strip():
            code = result.strip().lower()
            self._language = code
            existing_codes = [c for _, c in LANGUAGES] + [c for _, c in self._custom_languages]
            if code not in existing_codes:
                self._custom_languages.append((code, code))

    def _on_quit(self, icon, item):
        self._stop_event.set()
        icon.stop()

    def _start_hotkey_listener(self):
        hotkey_str = self.config.get("hotkey", "<ctrl>+<shift>+i")
        hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._on_hotkey)
        listener = keyboard.Listener(
            on_press=lambda k: hotkey.press(listener.canonical(k)),
            on_release=lambda k: hotkey.release(listener.canonical(k)),
        )
        listener.daemon = True
        listener.start()

    def _on_hotkey(self):
        self._pending_action = self._toggle_from_hotkey

    def _toggle_from_hotkey(self):
        if self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()

    def _start_recording(self):
        self.state = State.RECORDING
        self._icon.icon = ICON_RECORDING
        self._icon.title = "Local Notes - Recording"
        self._transcript_so_far = ""
        self._flush_stop.clear()
        self._incremental_temps = []
        self.recorder.start()
        threading.Thread(target=self._incremental_transcribe_loop, daemon=True).start()

    def _stop_recording(self):
        self.state = State.PROCESSING
        self._current_step = "Processing"
        self._processing_done = False
        self._icon.icon = ICON_PROCESSING
        self._icon.title = "Local Notes - Processing"

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
            f"SUMMARY:\n{summary}\n",
            encoding="utf-8",
        )

    def _process(self):
        audio_path = None
        try:
            # Stop the incremental loop and flush the tail
            self._flush_stop.set()

            self._current_step = "Saving audio"
            self._icon.title = "Local Notes - Saving audio"
            audio_path = self.recorder.stop()

            # Transcribe only the remaining tail audio
            self._current_step = "Transcribing tail"
            self._icon.title = "Local Notes - Transcribing tail"
            whisper_model = self.config.get("whisper_model", "base")
            tail_text = transcribe(audio_path, model_size=whisper_model, language=self._language)
            if tail_text.strip():
                self._transcript_so_far += (" " + tail_text if self._transcript_so_far else tail_text)

            transcript = self._transcript_so_far.strip()
            if not transcript:
                show_notification("Local Notes", "No speech detected.")
                return

            self._current_step = "Preparing model"
            self._icon.title = "Local Notes - Preparing model"
            ollama_model = self.config.get("ollama_model", "qwen3:8b")
            self._ollama.ensure_model(ollama_model)

            self._current_step = "Summarizing"
            self._icon.title = "Local Notes - Summarizing"
            summary = summarize(transcript, model=ollama_model, use_case=self._use_case, host=self._ollama_host)

            self._current_step = "Saving"
            self._icon.title = "Local Notes - Saving"
            self._save_transcript(transcript, summary)

            self._current_step = "Copying"
            self._icon.title = "Local Notes - Copying"
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
            self._reset()

    def _reset(self):
        self.state = State.IDLE
        self._icon.icon = ICON_IDLE
        self._icon.title = "Local Notes - Idle"

    def _poll_loop(self):
        """Background thread that dispatches hotkey actions."""
        while not self._stop_event.is_set():
            action = self._pending_action
            if action is not None:
                self._pending_action = None
                action()
            self._stop_event.wait(0.15)

    def run(self):
        poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        poll_thread.start()
        self._icon.run()


def main():
    app = LocalNotesApp()
    try:
        app.run()
    finally:
        app._ollama.stop()


if __name__ == "__main__":
    main()

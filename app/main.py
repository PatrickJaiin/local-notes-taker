import os
import threading
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import rumps
import yaml
from AppKit import NSApp, NSApplicationActivationPolicyAccessory, NSApplicationActivationPolicyRegular
from pynput import keyboard

from app.output import auto_paste, copy_to_clipboard, show_notification
from app.recorder import Recorder
from app.summarizer import summarize
from app.transcriber import transcribe

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transcripts"

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

USE_CASES = ["Meeting", "Lecture", "Brainstorm", "Interview", "Stand-up"]


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
        self._use_case = USE_CASES[0]
        self._current_step = ""
        self._processing_done = False
        self._spinner_index = 0
        self._pending_action = None

        # Build use-case submenu with checkmarks
        use_case_menu = rumps.MenuItem("Use Case")
        for uc in USE_CASES:
            item = rumps.MenuItem(uc, callback=self._select_use_case)
            if uc == self._use_case:
                item.state = 1
            use_case_menu.add(item)
        use_case_menu.add(rumps.separator)
        use_case_menu.add(rumps.MenuItem("Custom...", callback=self._custom_use_case))

        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            None,
            use_case_menu,
            None,
        ]
        self._start_hotkey_listener()
        # Single poll timer created on main thread — guarantees all UI
        # updates happen on the main thread, avoiding AppKit crashes.
        self._poll_timer = rumps.Timer(self._tick, 0.15)
        self._poll_timer.start()

    def _start_hotkey_listener(self):
        hotkey_str = self.config.get("hotkey", "<cmd>+<shift>+n")
        hotkeys = keyboard.GlobalHotKeys({hotkey_str: self._on_hotkey})
        hotkeys.daemon = True
        hotkeys.start()

    def _select_use_case(self, sender):
        self._uncheck_all_use_cases()
        sender.state = 1
        self._use_case = sender.title

    def _custom_use_case(self, sender):
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

    def _uncheck_all_use_cases(self):
        for item in self.menu["Use Case"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 0

    def _on_hotkey(self):
        self._pending_action = lambda: self.toggle_recording(None)

    def _tick(self, _):
        """Runs on main thread: dispatches hotkey actions, animates spinner,
        and detects processing completion."""
        action = self._pending_action
        if action is not None:
            self._pending_action = None
            action()

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
        self.recorder.start()

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
            self._current_step = "Saving audio"
            audio_path = self.recorder.stop()

            self._current_step = "Transcribing"
            whisper_model = self.config.get("whisper_model", "base")
            transcript = transcribe(audio_path, model_size=whisper_model)

            if not transcript.strip():
                show_notification("Local Notes", "No speech detected.")
                return

            self._current_step = "Summarizing"
            ollama_model = self.config.get("ollama_model", "qwen3:8b")
            summary = summarize(transcript, model=ollama_model, use_case=self._use_case)

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
            if audio_path:
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def _reset(self):
        self.state = State.IDLE
        self.title = "📝"
        self.menu["Start Recording"].title = "Start Recording"


def main():
    app = LocalNotesApp()
    app.run()


if __name__ == "__main__":
    main()

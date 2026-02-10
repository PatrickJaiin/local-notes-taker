import subprocess

import pyperclip


def copy_to_clipboard(text: str):
    pyperclip.copy(text)


def show_notification(title: str, message: str):
    # Truncate message for notification display
    display_msg = message[:200] + "..." if len(message) > 200 else message
    script = (
        f'display notification "{_escape(display_msg)}" '
        f'with title "{_escape(title)}"'
    )
    subprocess.run(["osascript", "-e", script], check=False)


def auto_paste():
    """Simulate Cmd+V to paste into the focused application."""
    script = """
    tell application "System Events"
        keystroke "v" using command down
    end tell
    """
    subprocess.run(["osascript", "-e", script], check=False)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')

from __future__ import annotations

import subprocess


def copy_to_clipboard(text: str) -> None:
    # Use pbcopy directly — more reliable than pyperclip on macOS.
    process = subprocess.Popen(
        ["pbcopy"], stdin=subprocess.PIPE, env={"LANG": "en_US.UTF-8"}
    )
    process.communicate(text.encode("utf-8"))


def show_notification(title: str, message: str) -> None:
    display_msg = message[:200] + "…" if len(message) > 200 else message
    script = (
        f"display notification {_applescript_string(display_msg)} "
        f"with title {_applescript_string(title)}"
    )
    subprocess.run(
        ["osascript", "-e", script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def auto_paste() -> bool:
    """Simulate Cmd+V to paste into the focused application. Returns False when
    the keystroke was refused (typically a missing Automation/Accessibility
    permission) so the caller can tell the user instead of claiming success."""
    script = """
    tell application "System Events"
        keystroke "v" using command down
    end tell
    """
    result = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"[paste] osascript failed: {result.stderr.decode(errors='replace').strip()}")
        return False
    return True


def _applescript_string(text: str) -> str:
    """Safely encode a string as an AppleScript string literal by escaping
    backslashes, quotes, and control characters."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return f'"{text}"'

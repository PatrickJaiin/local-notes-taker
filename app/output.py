import subprocess


def copy_to_clipboard(text: str):
    # Use pbcopy directly - more reliable than pyperclip on macOS
    process = subprocess.Popen(
        ["pbcopy"], stdin=subprocess.PIPE, env={"LANG": "en_US.UTF-8"}
    )
    process.communicate(text.encode("utf-8"))


def show_notification(title: str, message: str):
    display_msg = message[:200] + "\u2026" if len(message) > 200 else message
    # Use -s flag with properly escaped arguments to avoid shell injection
    script = (
        f'display notification {_applescript_string(display_msg)} '
        f'with title {_applescript_string(title)}'
    )
    subprocess.run(["osascript", "-e", script], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def auto_paste():
    """Simulate Cmd+V to paste into the focused application."""
    script = """
    tell application "System Events"
        keystroke "v" using command down
    end tell
    """
    subprocess.run(["osascript", "-e", script], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _applescript_string(text: str) -> str:
    """Safely encode a string for AppleScript using 'quoted form'
    approach: replace backslashes, quotes, and control characters."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return f'"{text}"'

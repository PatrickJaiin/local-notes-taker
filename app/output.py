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
    """Paste into the focused app by synthesizing Cmd+V as a native CGEvent.
    Unlike the old osascript/System Events route, only THIS app needs the
    Accessibility permission — no Automation consent, and no stray osascript/
    Script Editor entries appear in the TCC lists. The system permission prompt
    is raised automatically on first use. Returns False when the permission is
    missing (or the event can't be posted) so the caller can tell the user."""
    try:
        import Quartz
        from ApplicationServices import (
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )

        if not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
            return False
        V_KEY = 9  # kVK_ANSI_V
        for key_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, V_KEY, key_down)
            Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True
    except Exception as e:
        print(f"[paste] CGEvent paste failed: {e}")
        return False


def _applescript_string(text: str) -> str:
    """Safely encode a string as an AppleScript string literal by escaping
    backslashes, quotes, and control characters."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return f'"{text}"'

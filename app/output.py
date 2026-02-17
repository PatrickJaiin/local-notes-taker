import time

import pyautogui
import pyperclip
from plyer import notification


def copy_to_clipboard(text: str):
    pyperclip.copy(text)


def show_notification(title: str, message: str):
    display_msg = message[:200] + "..." if len(message) > 200 else message
    notification.notify(
        title=title,
        message=display_msg,
        app_name="Local Notes",
        timeout=5,
    )


def auto_paste():
    """Simulate Ctrl+V to paste into the focused application."""
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "v")

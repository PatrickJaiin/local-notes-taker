"""Manages a bundled Ollama server lifecycle.

In 'bundled' mode, starts an Ollama binary shipped with the app on a free
port and points it at the bundled model directory.  In 'external' mode,
assumes a system Ollama is already running on the default port.
"""

import atexit
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _app_root() -> Path:
    """Return the application root, handling both dev and frozen (py2app/PyInstaller) layouts."""
    if getattr(sys, "frozen", False):
        # py2app: .app/Contents/Resources
        # PyInstaller: _MEIPASS or exe directory
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent.parent / "Resources"
    return Path(__file__).resolve().parent.parent


class OllamaManager:
    """Start / stop a bundled Ollama server."""

    def __init__(self, mode: str = "external"):
        self.mode = mode
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._host: str | None = None

        if mode == "bundled":
            self._port = _find_free_port()
            self._host = f"http://127.0.0.1:{self._port}"
        else:
            self._host = "http://127.0.0.1:11434"

    @property
    def host(self) -> str:
        return self._host

    def _find_binary(self) -> Path:
        root = _app_root()
        # Check bundled assets locations
        candidates = [
            root / "assets" / "ollama",
            root / "assets" / "ollama.exe",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Bundled Ollama binary not found. Searched: {[str(c) for c in candidates]}"
        )

    def _find_models_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            # Use writable user directory for models (app bundle may be read-only)
            support = Path.home() / "Library" / "Application Support" / "Local Notes"
            models = support / "models"
        else:
            root = _app_root()
            models = root / "assets" / "models"
        models.mkdir(parents=True, exist_ok=True)
        return models

    def start(self) -> str:
        """Start the Ollama server and return the host URL.

        In external mode, just returns the default host without starting anything.
        """
        if self.mode != "bundled":
            return self._host

        binary = self._find_binary()
        binary.chmod(0o755)
        models_dir = self._find_models_dir()

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{self._port}"
        env["OLLAMA_MODELS"] = str(models_dir)

        self._process = subprocess.Popen(
            [str(binary), "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.stop)
        self._wait_ready()
        return self._host

    def _wait_ready(self, timeout: float = 30.0):
        """Poll Ollama's API until it responds or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self._host}/api/tags", timeout=2)
                if r.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Ollama did not become ready within {timeout}s on {self._host}")

    def ensure_model(self, model: str):
        """Pull the model if it's not already available."""
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Check both exact name and name without tag
                if any(model in m or m in model for m in models):
                    return
        except Exception:
            pass

        binary = self._find_binary()
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{self._port}" if self._port else "127.0.0.1:11434"
        env["OLLAMA_MODELS"] = str(self._find_models_dir())
        subprocess.run([str(binary), "pull", model], env=env, check=True)

    def stop(self):
        """Terminate the Ollama subprocess if running."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
        self._process = None

    @property
    def running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

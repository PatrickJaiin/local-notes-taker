#!/usr/bin/env python3
"""Download and stage Ollama binary + Qwen3 model for bundled packaging.

Usage:
    python build/download_assets.py [--platform macos-arm64|macos-amd64|windows-amd64]
                                    [--model qwen3:8b]

Run this ONCE on the developer's machine before building the app.
It creates an `assets/` directory at the project root containing:
    assets/ollama          (or ollama.exe on Windows)
    assets/models/          (Ollama model blobs)
"""

import argparse
import os
import platform
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

OLLAMA_VERSION = "v0.6.2"

DOWNLOAD_URLS = {
    "macos-arm64": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-darwin-arm64.tgz",
    "macos-amd64": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-darwin-amd64.tgz",
    "windows-amd64": f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-windows-amd64.zip",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def detect_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        return "macos-arm64" if machine == "arm64" else "macos-amd64"
    elif system == "windows":
        return "windows-amd64"
    else:
        sys.exit(f"Unsupported platform: {system} {machine}")


def download_ollama(plat: str):
    url = DOWNLOAD_URLS[plat]
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Ollama {OLLAMA_VERSION} for {plat}...")
    with tempfile.NamedTemporaryFile(suffix=".archive", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path)

        if url.endswith(".tgz"):
            with tarfile.open(tmp_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("/ollama") or member.name == "ollama":
                        member.name = "ollama"
                        tar.extract(member, ASSETS_DIR)
                        break
                else:
                    tar.extractall(ASSETS_DIR)
            binary = ASSETS_DIR / "ollama"
            if binary.exists():
                binary.chmod(0o755)
        elif url.endswith(".zip"):
            with zipfile.ZipFile(tmp_path) as zf:
                for name in zf.namelist():
                    if name.endswith("ollama.exe"):
                        data = zf.read(name)
                        dest = ASSETS_DIR / "ollama.exe"
                        dest.write_bytes(data)
                        break
                else:
                    zf.extractall(ASSETS_DIR)
    finally:
        os.unlink(tmp_path)

    binary_name = "ollama.exe" if "windows" in plat else "ollama"
    binary = ASSETS_DIR / binary_name
    if not binary.exists():
        sys.exit(f"Error: expected binary at {binary} but not found")
    print(f"  -> {binary}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pull_model(model: str, plat: str):
    """Start a temporary Ollama server, pull the model, then copy blobs."""
    binary_name = "ollama.exe" if "windows" in plat else "ollama"
    binary = ASSETS_DIR / binary_name
    models_dir = ASSETS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = str(models_dir)

    print(f"Starting temporary Ollama server on port {port}...")
    proc = subprocess.Popen(
        [str(binary), "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        import requests
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=2)
                if r.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        else:
            sys.exit("Ollama server did not start in time")

        print(f"Pulling model {model} (this may take a while)...")
        result = subprocess.run(
            [str(binary), "pull", model],
            env=env,
            capture_output=False,
        )
        if result.returncode != 0:
            sys.exit(f"Failed to pull model {model}")

        print(f"  -> Model blobs stored in {models_dir}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    parser = argparse.ArgumentParser(description="Download Ollama + model for bundled packaging")
    parser.add_argument("--platform", choices=list(DOWNLOAD_URLS.keys()), default=None,
                        help="Target platform (auto-detected if omitted)")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model to pull")
    parser.add_argument("--skip-model", action="store_true", help="Only download binary, skip model pull")
    args = parser.parse_args()

    plat = args.platform or detect_platform()
    print(f"Target platform: {plat}")

    download_ollama(plat)

    if not args.skip_model:
        pull_model(args.model, plat)

    print("\nDone! Assets are in:", ASSETS_DIR)
    print("You can now run the build script.")


if __name__ == "__main__":
    main()

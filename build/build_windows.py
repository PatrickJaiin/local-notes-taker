#!/usr/bin/env python3
"""Build a Windows executable using PyInstaller.

Prerequisites:
    pip install pyinstaller
    python build/download_assets.py --platform windows-amd64

Usage:
    python build/build_windows.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build" / "pyinstaller_build"
APP_NAME = "LocalNotes"


def check_assets():
    binary = ASSETS_DIR / "ollama.exe"
    models = ASSETS_DIR / "models"
    if not binary.exists():
        sys.exit(f"Missing bundled Ollama binary at {binary}.\nRun: python build/download_assets.py --platform windows-amd64")
    if not models.exists() or not any(models.iterdir()):
        sys.exit(f"Missing model blobs at {models}.\nRun: python build/download_assets.py --platform windows-amd64")


def build_exe():
    add_data = [
        f"{ASSETS_DIR / 'ollama.exe'}{os.pathsep}assets",
        f"{ASSETS_DIR / 'models'}{os.pathsep}assets/models",
        f"{PROJECT_ROOT / 'config.yaml'}{os.pathsep}.",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
    ]

    for item in add_data:
        cmd.extend(["--add-data", item])

    for mod in ["ollama", "faster_whisper", "ctranslate2", "yaml", "requests", "pynput"]:
        cmd.extend(["--hidden-import", mod])

    cmd.append(str(PROJECT_ROOT / "app" / "main.py"))

    print("Running PyInstaller...")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    exe_dir = DIST_DIR / APP_NAME
    if not exe_dir.exists():
        sys.exit(f"Build failed - {exe_dir} not found")
    print(f"Built: {exe_dir}")
    return exe_dir


def create_zip(exe_dir: Path):
    zip_path = DIST_DIR / f"{APP_NAME}"
    print("Creating zip archive...")
    shutil.make_archive(str(zip_path), "zip", str(DIST_DIR), APP_NAME)
    final = Path(f"{zip_path}.zip")
    print(f"Archive created: {final}")
    return final


def main():
    check_assets()

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    exe_dir = build_exe()
    zip_path = create_zip(exe_dir)

    print(f"\nDone! Distribute: {zip_path}")


if __name__ == "__main__":
    main()

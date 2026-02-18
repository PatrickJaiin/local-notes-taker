#!/usr/bin/env python3
"""Build a macOS .app bundle and .dmg using py2app.

Prerequisites:
    pip install py2app
    python build/download_assets.py   # populates assets/

Usage:
    python build/build_macos.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build" / "py2app_build"
APP_NAME = "Local Notes"

SETUP_PY = PROJECT_ROOT / "setup_macos.py"


def check_assets():
    binary = ASSETS_DIR / "ollama"
    models = ASSETS_DIR / "models"
    if not binary.exists():
        sys.exit(f"Missing bundled Ollama binary at {binary}.\nRun: python build/download_assets.py")
    if not models.exists() or not any(models.iterdir()):
        sys.exit(f"Missing model blobs at {models}.\nRun: python build/download_assets.py")


def write_setup_py():
    """Generate a temporary setup.py for py2app."""
    assets_files = []
    for p in ASSETS_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(PROJECT_ROOT)
            assets_files.append(str(rel))

    setup_content = f'''\
from setuptools import setup

APP = ["app/main.py"]
DATA_FILES = []
OPTIONS = {{
    "argv_emulation": False,
    "iconfile": None,
    "plist": {{
        "CFBundleName": "{APP_NAME}",
        "CFBundleDisplayName": "{APP_NAME}",
        "CFBundleIdentifier": "com.localnotes.app",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": "Local Notes needs microphone access to record audio for transcription.",
    }},
    "packages": ["app", "ollama", "faster_whisper", "ctranslate2", "yaml", "rumps", "pynput"],
    "includes": ["requests"],
    "resources": {assets_files!r} + ["config.yaml"],
}}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={{"py2app": OPTIONS}},
)
'''
    SETUP_PY.write_text(setup_content)
    return SETUP_PY


def build_app():
    setup = write_setup_py()
    try:
        subprocess.run(
            [sys.executable, str(setup), "py2app"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
    finally:
        setup.unlink(missing_ok=True)

    app_path = DIST_DIR / f"{APP_NAME}.app"
    if not app_path.exists():
        sys.exit(f"Build failed - {app_path} not found")
    print(f"Built: {app_path}")
    return app_path


def create_dmg(app_path: Path):
    dmg_path = DIST_DIR / f"{APP_NAME}.dmg"
    dmg_path.unlink(missing_ok=True)

    print("Creating DMG...")
    subprocess.run(
        [
            "hdiutil", "create",
            "-volname", APP_NAME,
            "-srcfolder", str(app_path),
            "-ov",
            "-format", "UDZO",
            str(dmg_path),
        ],
        check=True,
    )
    print(f"DMG created: {dmg_path}")
    return dmg_path


def main():
    check_assets()

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    app_path = build_app()
    dmg_path = create_dmg(app_path)

    print(f"\nDone! Distribute: {dmg_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a macOS .app bundle and .dmg using PyInstaller.

Prerequisites:
    pip install pyinstaller
    python build/download_assets.py   # populates assets/

Usage:
    python build/build_macos.py
"""

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build" / "pyinstaller_build"
APP_NAME = "Local Notes"


def check_assets():
    binary = ASSETS_DIR / "ollama"
    if not binary.exists():
        sys.exit(f"Missing bundled Ollama binary at {binary}.\nRun: python build/download_assets.py")


def build_app():
    models = ASSETS_DIR / "models"
    add_data = [
        f"{ASSETS_DIR / 'ollama'}{os.pathsep}assets",
        f"{PROJECT_ROOT / 'config.yaml'}{os.pathsep}.",
    ]
    if models.exists() and any(models.iterdir()):
        add_data.append(f"{models}{os.pathsep}assets/models")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onedir",
        "--windowed",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--osx-bundle-identifier", "com.localnotes.app",
    ]

    for item in add_data:
        cmd.extend(["--add-data", item])

    for mod in ["ollama", "faster_whisper", "ctranslate2", "yaml", "rumps", "requests", "pynput"]:
        cmd.extend(["--hidden-import", mod])

    cmd.append(str(PROJECT_ROOT / "app" / "main.py"))

    print("Running PyInstaller...")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    app_path = DIST_DIR / f"{APP_NAME}.app"
    if not app_path.exists():
        sys.exit(f"Build failed - {app_path} not found")

    # Update Info.plist with macOS-specific settings
    plist_path = app_path / "Contents" / "Info.plist"
    if plist_path.exists():
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        plist["LSUIElement"] = True
        plist["NSMicrophoneUsageDescription"] = (
            "Local Notes needs microphone access to record audio for transcription."
        )
        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)

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
    skip_dmg = "--no-dmg" in sys.argv

    check_assets()

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    app_path = build_app()

    if skip_dmg:
        print(f"\nDone! App bundle: {app_path}")
    else:
        dmg_path = create_dmg(app_path)
        print(f"\nDone! Distribute: {dmg_path}")


if __name__ == "__main__":
    main()

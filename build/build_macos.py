#!/usr/bin/env python3
"""Build the Local Notes macOS .app and .dmg with PyInstaller (Apple Silicon).

Prerequisites:
    pip install -e .
    pip install "pyinstaller>=6.14.0"

Usage:
    python build/build_macos.py            # build .app + .dmg (ad-hoc signed)
    python build/build_macos.py --no-dmg   # build .app only

Model weights are NOT bundled — they download on first use. Developer ID signing
and notarization are done in CI (see .github/workflows/release.yml); this script
ad-hoc signs so the app runs locally during development.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC = PROJECT_ROOT / "build" / "LocalNotes.spec"
ENTITLEMENTS = PROJECT_ROOT / "build" / "entitlements.plist"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build" / "pyinstaller_build"
APP_NAME = "Local Notes"


def build_app() -> Path:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean", "--noconfirm",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        str(SPEC),
    ]
    print("Running PyInstaller...")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    app_path = DIST_DIR / f"{APP_NAME}.app"
    if not app_path.exists():
        sys.exit(f"Build failed — {app_path} not found")
    print(f"Built: {app_path}")
    return app_path


def adhoc_sign(app_path: Path) -> None:
    """Ad-hoc sign with the hardened-runtime entitlements so the app launches in
    development. (CI re-signs with a Developer ID cert + notarizes for release.)"""
    print("Ad-hoc signing...")
    result = subprocess.run(
        [
            "codesign", "--force", "--sign", "-",
            "--options", "runtime",
            "--entitlements", str(ENTITLEMENTS),
            "--timestamp=none",
            str(app_path),
        ],
        check=False,
    )
    if result.returncode != 0:
        print("WARNING: ad-hoc signing failed — the app may not launch under the hardened runtime.")


def create_dmg(app_path: Path) -> Path:
    dmg_path = DIST_DIR / f"{APP_NAME}.dmg"
    dmg_path.unlink(missing_ok=True)
    print("Creating DMG...")
    subprocess.run(
        [
            "hdiutil", "create",
            "-volname", APP_NAME,
            "-srcfolder", str(app_path),
            "-ov", "-format", "UDZO",
            str(dmg_path),
        ],
        check=True,
    )
    print(f"DMG created: {dmg_path}")
    return dmg_path


def main() -> None:
    skip_dmg = "--no-dmg" in sys.argv
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            shutil.rmtree(d)

    app_path = build_app()
    adhoc_sign(app_path)

    if skip_dmg:
        print(f"\nDone! App bundle: {app_path}")
    else:
        dmg = create_dmg(app_path)
        print(f"\nDone! Distribute: {dmg}")


if __name__ == "__main__":
    main()

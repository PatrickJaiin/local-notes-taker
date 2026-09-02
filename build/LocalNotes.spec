# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Local Notes (Apple Silicon, --onedir).

Bundles every inference backend (MLX + PyTorch). Only model WEIGHTS download on
demand at runtime, so they are NOT bundled here.

Build:  pyinstaller --clean --noconfirm build/LocalNotes.spec
"""

import os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [(os.path.join(PROJECT_ROOT, "config.yaml"), ".")]
binaries = []
hiddenimports = []

# MLX packages ship compiled Metal shader libs (.metallib) + dylibs that a plain
# analysis misses; collect_all is required or the packaged app dies with
# "Failed to load the default metallib".
for pkg in ("mlx", "mlx_whisper", "parakeet_mlx", "mlx_lm"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# parakeet-mlx pulls in librosa, which imports lazy_loader and numba/llvmlite
# through lazy/conditional imports PyInstaller's static analysis misses.
# Collect the whole chain explicitly, or `import parakeet_mlx` dies at runtime
# with ModuleNotFoundError: lazy_loader.
for pkg in ("librosa", "lazy_loader", "numba", "llvmlite", "soxr", "audioread", "pooch", "soundfile", "hf_xet"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# hf_xet is the fast HuggingFace download backend (native Rust); bundling it
# avoids the slow plain-HTTP fallback on first model download.
hiddenimports += ["hf_xet"]

# auto_paste synthesizes Cmd+V via CGEvent; these pyobjc frameworks are
# imported lazily inside the function so name them explicitly.
hiddenimports += ["Quartz", "ApplicationServices"]

hiddenimports += collect_submodules("mlx_lm")

# Many libraries read their own package metadata at import time.
for dist in (
    "torch", "torchaudio", "transformers", "tokenizers", "safetensors",
    "regex", "tqdm", "huggingface-hub", "filelock", "packaging", "numpy",
    "pyyaml", "peft", "sentencepiece", "soundfile", "librosa", "numba",
):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

hiddenimports += [
    "sentencepiece", "sounddevice", "soundfile", "peft",
    "scipy.io.wavfile", "scipy.signal", "sklearn.utils._typedefs",
]

# transformers and peft resolve concrete model architectures via runtime
# string imports (e.g. transformers.models.granite_speech) that PyInstaller's
# static analysis can't follow. Without these the Granite backend dies with
# ModuleNotFoundError in the packaged app (the default Parakeet path is fine).
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("peft")
datas += collect_data_files("transformers")

a = Analysis(
    [os.path.join(PROJECT_ROOT, "app", "main.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Local Notes",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Local Notes",
)

app = BUNDLE(
    coll,
    name="Local Notes.app",
    icon=os.path.join(PROJECT_ROOT, "build", "AppIcon.icns"),
    bundle_identifier="com.localnotes.app",
    info_plist={
        "LSUIElement": True,  # menu-bar only, no Dock icon
        "LSMultipleInstancesProhibited": True,
        "NSMicrophoneUsageDescription": "Local Notes needs microphone access to record audio for transcription.",
        "CFBundleShortVersionString": "0.2.1",
        "CFBundleVersion": "0.2.1",
        "LSMinimumSystemVersion": "14.0",
    },
)

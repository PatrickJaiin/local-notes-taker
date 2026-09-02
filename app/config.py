from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

APP_NAME = "Local Notes"

# --- Backend identifiers ---
PARAKEET = "parakeet"
WHISPER = "whisper"
GRANITE = "granite"
VALID_BACKENDS = (PARAKEET, WHISPER, GRANITE)

# --- Default configuration ---
DEFAULT_CONFIG: dict = {
    "hotkey": "<cmd>+<shift>+i",
    "asr_backend": PARAKEET,
    # HuggingFace model ids per backend (weights download on demand):
    "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v3",
    "whisper_model": "mlx-community/whisper-large-v3-mlx",
    "granite_model": "ibm-granite/granite-speech-3.3-8b",
    "granite_device": "mps",  # "mps" (Apple GPU, fp32) or "cpu" (slow, most compatible)
    # Summary LLM (mlx-lm):
    "summary_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "language": None,  # Whisper/Parakeet language code; auto-detect if empty
    "use_case": "Meeting",
    "auto_paste": True,
    "chunk_seconds": 10,  # live-transcription cadence for the Whisper pseudo-stream
    # Pre-process audio before transcribing (high-pass + gentle levelling). Off by
    # default: ASR models want the raw signal, and the cleanup mostly amplifies
    # room noise during pauses. Turn on only for recordings too quiet to decode.
    "clean_audio": False,
}

# --- Filesystem layout ---
# Model weights are machine-global and large, so they always live under
# Application Support (never inside the repo, even in dev).
SUPPORT_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
HF_DIR = SUPPORT_DIR / "huggingface"


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# In a packaged (frozen) app, user data lives in Application Support; in dev it
# lives in the repo root (transcripts/ and *.wav are gitignored).
DATA_DIR = SUPPORT_DIR if _is_frozen() else _repo_root()
CONFIG_PATH = DATA_DIR / "config.yaml"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
RECORDINGS_DIR = DATA_DIR / "recordings"


def configure_hf_env() -> None:
    """Point HuggingFace + transformers caches at Application Support. MUST be
    called before importing huggingface_hub / mlx / transformers / torch, since
    those read these env vars at import time."""
    HF_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_DIR / "hub"))
    # Granite/torch: let unsupported MPS ops fall back to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Avoid noisy tokenizers fork warnings in the menu-bar process.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Stop joblib/loky (pulled in via librosa) from re-exec'ing the frozen app
    # binary as worker processes — that spawns duplicate menu-bar icons. Force
    # in-process threading; our hot paths don't rely on joblib parallelism.
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _ensure_config_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return
    # Seed from a bundled config.yaml if present (frozen app), else from defaults.
    bundled = _repo_root() / "config.yaml"
    if _is_frozen() and bundled.exists():
        CONFIG_PATH.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        CONFIG_PATH.write_text(
            yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8"
        )


def load_config() -> dict:
    _ensure_config_file()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _normalize({**DEFAULT_CONFIG, **loaded})


def save_config(config: dict) -> None:
    """Persist config, preserving comments/layout by patching changed scalar values
    in the existing file line-by-line. Falls back to a full dump if the file is new
    or the patch wouldn't round-trip correctly."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_CONFIG, **config}
    text = yaml.safe_dump(merged, sort_keys=False)
    try:
        if CONFIG_PATH.exists():
            patched = _patch_yaml_text(CONFIG_PATH.read_text(encoding="utf-8"), merged)
            reparsed = yaml.safe_load(patched) or {}
            if all(reparsed.get(k) == merged[k] for k in merged):
                text = patched  # round-trips correctly; keep comments
    except Exception:
        pass  # keep the full-dump fallback
    CONFIG_PATH.write_text(text, encoding="utf-8")


def _scalar(value) -> str:
    """Serialize a Python scalar to its YAML representation (empty for None)."""
    if value is None:
        return ""
    inner = yaml.safe_dump({"v": value}, default_flow_style=True, sort_keys=False).strip()
    return inner[1:-1].split(":", 1)[1].strip()  # "{v: X}" -> "X"


def _patch_yaml_text(text: str, values: dict) -> str:
    """Replace top-level scalar key values in YAML text in place, keeping comments
    and layout; append any keys not already present."""
    import re

    key_re = re.compile(r"^([A-Za-z0-9_]+):\s?(.*)$")
    lines = text.splitlines()
    seen: set[str] = set()
    for i, line in enumerate(lines):
        m = key_re.match(line)
        if not m or m.group(1) not in values:
            continue
        key, rest = m.group(1), m.group(2)
        seen.add(key)
        ci = _comment_start(rest)
        comment = (" " + rest[ci:]) if ci != -1 else ""
        val = _scalar(values[key])
        lines[i] = f"{key}: {val}{comment}" if val else f"{key}:{comment}"
    missing = {k: values[k] for k in values if k not in seen}
    if missing:
        lines.append(yaml.safe_dump(missing, sort_keys=False).rstrip("\n"))
    return "\n".join(lines).rstrip("\n") + "\n"


def _comment_start(rest: str) -> int:
    """Index of a trailing YAML comment in ``rest``, or -1. Quote-aware, so a '#'
    inside a quoted value (e.g. use_case: 'Project #12') isn't treated as one."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            return i
    return -1


# v0.1.0 (faster-whisper) stored bare size names; map the known ones onto the
# MLX HuggingFace repos this stack uses.
_LEGACY_WHISPER_MODELS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def _normalize(config: dict) -> dict:
    if config.get("asr_backend") not in VALID_BACKENDS:
        config["asr_backend"] = DEFAULT_CONFIG["asr_backend"]
    for key in ("parakeet_model", "whisper_model", "granite_model", "summary_model"):
        model = config.get(key)
        if not model or not isinstance(model, str):
            config[key] = DEFAULT_CONFIG[key]
        elif "/" not in model:
            # Not a HuggingFace repo id — a leftover from an old config version.
            config[key] = _LEGACY_WHISPER_MODELS.get(model, DEFAULT_CONFIG[key])
    if config.get("granite_device") not in ("mps", "cpu"):
        config["granite_device"] = DEFAULT_CONFIG["granite_device"]
    if config.get("language") == "":
        config["language"] = None
    try:
        config["chunk_seconds"] = max(3, int(config.get("chunk_seconds", 10)))
    except (TypeError, ValueError):
        config["chunk_seconds"] = DEFAULT_CONFIG["chunk_seconds"]
    config["auto_paste"] = bool(config.get("auto_paste", True))
    # Absent from every config.yaml written before this key existed, so read it
    # through the default rather than assuming it is present.
    config["clean_audio"] = bool(config.get("clean_audio", DEFAULT_CONFIG["clean_audio"]))
    return config


def model_id_for_backend(config: dict, backend: str) -> str:
    return {
        PARAKEET: config.get("parakeet_model", DEFAULT_CONFIG["parakeet_model"]),
        WHISPER: config.get("whisper_model", DEFAULT_CONFIG["whisper_model"]),
        GRANITE: config.get("granite_model", DEFAULT_CONFIG["granite_model"]),
    }[backend]

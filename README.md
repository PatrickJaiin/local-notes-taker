# Local Notes

A macOS **menu-bar app** that records audio, shows a **live transcript** as you
speak, and — when you stop — uses a **local LLM** to write notes formatted for
your use case. Everything runs **locally on Apple Silicon**. No cloud, no API keys.

Transcription ships with **NVIDIA Parakeet TDT** by default and lets you switch to
**Whisper Large V3** or **IBM Granite Speech 3.3 8B** — those download their
weights the first time you pick them. Summaries are generated with a local
**`mlx-lm`** model (Qwen3 4B by default).

## Requirements

- **Apple Silicon Mac** (M-series), macOS 14+
- For running from source: **Python 3.11 or 3.12**

> Apple Silicon only — the models run on Apple's MLX/Metal (and PyTorch/MPS for
> Granite). Intel Macs and Windows are not supported.

## Download

Grab the latest `.dmg` from the [Releases page](https://github.com/PatrickJaiin/local-notes-taker/releases).
On first use of a transcriber the app downloads that model's weights into
`~/Library/Application Support/Local Notes/huggingface` (Parakeet ≈ 2.5 GB,
Whisper ≈ 1.6–3 GB, Granite 8B ≈ 17 GB). The menu bar shows download progress.

## Features

- **Menu-bar app** — always one click (or hotkey) away
- **Global hotkey** — start/stop recording from any app (default **⌘⇧I**)
- **Live transcript** — Parakeet/Whisper stream a running transcript while you record
- **Switchable transcribers** — Parakeet TDT (default) · Whisper Large V3 · Granite Speech 8B
- **Local summaries** — `mlx-lm`; pick the summary model and the use-case format
- **Use-case presets** — Meeting, Lecture, Brainstorm, Interview, Stand-up, or a custom one you define
- **Auto-paste** — the summary is copied and pasted into your active app
- **History** — full transcripts + summaries saved under `transcripts/`

## Install from source

```bash
git clone https://github.com/PatrickJaiin/local-notes-taker.git
cd local-notes-taker
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
local-notes
```

A **📝** icon appears in your menu bar.

| Action | How |
|---|---|
| Start / stop recording | Menu-bar icon → Start/Stop Recording, or the global hotkey |
| Switch transcriber | Menu → **Transcriber** → Parakeet / Whisper / Granite |
| Choose summary model | Menu → **Summary Model** |
| Change format | Menu → **Use Case** → preset or **Custom…** |
| Set language | Menu → **Language** |

When you stop, the app cleans the audio, runs the final transcription pass,
summarizes it locally, copies the summary to your clipboard, and pastes it into
the focused app. The menu-bar icon shows a spinner with the current step.

## Transcribers

| Backend | Model (default) | Notes |
|---|---|---|
| **Parakeet** (default) | `mlx-community/parakeet-tdt-0.6b-v3` | Fast, true streaming live transcript, 25 European languages |
| **Whisper** | `mlx-community/whisper-large-v3-mlx` | ~99 languages (use this for e.g. Hindi/Malayalam); pseudo-streaming |
| **Granite** | `ibm-granite/granite-speech-3.3-8b` | Heavy (~34 GB RAM, fp32); transcribes when you stop (no live preview) |

> Granite 8B is large. The app warns on machines with limited RAM, loads it only
> for transcription, and frees it before summarizing so the two models don't
> co-reside in memory.

## Configuration

All settings live in `config.yaml` (in the repo when running from source, or in
`~/Library/Application Support/Local Notes/` when installed). The menu writes to
it automatically; **Reload Config** re-reads it.

```yaml
asr_backend: parakeet                                # parakeet | whisper | granite
parakeet_model: mlx-community/parakeet-tdt-0.6b-v3
whisper_model: mlx-community/whisper-large-v3-mlx
granite_model: ibm-granite/granite-speech-3.3-8b
granite_device: mps                                  # mps | cpu
summary_model: mlx-community/Qwen3-4B-Instruct-2507-4bit
language:                                            # blank = auto-detect
use_case: Meeting
auto_paste: true
chunk_seconds: 10                                    # Whisper live cadence
hotkey: <cmd>+<shift>+i
```

## Building a .dmg

See [`build/README.md`](build/README.md).

## License

[GPL-3.0](LICENSE)

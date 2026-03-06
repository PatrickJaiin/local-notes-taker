# Local Notes Taker

A macOS menu bar app that records audio, transcribes it with Whisper, and generates structured notes using Ollama, all running locally on your machine. No cloud services, no API keys, complete privacy.

## Download

Grab the latest standalone `.dmg` from the [Releases page](https://github.com/PatrickJaiin/local-notes-taker/releases). The app bundles Ollama and automatically downloads the Qwen3 model on first launch — no manual installation needed.

> A Windows build (`.zip`) is also available on the Releases page, built from the [`windows`](https://github.com/PatrickJaiin/local-notes-taker/tree/windows) branch.

## Features

- **Menu bar app** — lives in your macOS menu bar, always one click (or hotkey) away
- **Global hotkey** — start/stop recording from any app with a keyboard shortcut
- **Local transcription** — uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs on CPU, no GPU required)
- **Local summarization** — generates notes via [Ollama](https://ollama.com) with any model you choose
- **Use case presets** — Meeting, Lecture, Brainstorm, Interview, Stand-up, or custom
- **Auto-paste** — summary is copied to clipboard and pasted into your active app
- **Transcript history** — full transcripts and summaries saved to `transcripts/`
- **Open transcripts quickly** — menu action opens the transcripts folder directly

## Install from Source

### Prerequisites

- **macOS** (uses native menu bar and AppleScript)
- **Python 3.9+**
- **[Ollama](https://ollama.com)** installed and running

### Installation

```bash
git clone https://github.com/PatrickJaiin/local-notes-taker.git
cd local-notes-taker
pip install -e .
ollama pull qwen3:8b
```

> You can swap `qwen3:8b` for any Ollama model — just update `config.yaml`.

## Usage

Start the app:

```bash
local-notes
```

A **pencil icon** (📝) appears in your menu bar.

| Action | How |
|---|---|
| Start recording | Click the menu bar icon → "Start Recording", or press the global hotkey |
| Stop recording | Click "Stop Recording" or press the hotkey again |
| Change use case | Menu bar icon → "Use Case" → pick a preset or enter a custom one |

Once you stop recording, the app will:

1. Transcribe the audio with Whisper
2. Summarize the transcript with Ollama
3. Copy the summary to your clipboard
4. Auto-paste it into the focused app

The menu bar icon shows a spinner with progress during processing.

### Global Hotkey

Default: **Cmd + Shift + I**

Change it in `config.yaml`:

```yaml
hotkey: <cmd>+<shift>+i
```

## Configuration

All settings live in `config.yaml`:

```yaml
hotkey: <cmd>+<shift>+i
whisper_model: large-v3-turbo   # tiny, base, small, medium, large-v3, large-v3-turbo
ollama_model: qwen3:8b          # any model pulled in Ollama
```

| Option | Description |
|---|---|
| `hotkey` | Global keyboard shortcut to toggle recording |
| `whisper_model` | Whisper model size — smaller is faster, larger is more accurate |
| `ollama_model` | Ollama model used for summarization |
| `ollama_mode` | `external` (default, uses system Ollama) or `bundled` (uses packaged binary) |
| `language` | Whisper language code (`en`, `hi`, `fr`, etc.) — empty for auto-detect |

## Transcripts

Every recording saves a timestamped file in the `transcripts/` directory:

```
transcripts/2025-06-15_14-30-00_meeting.txt
```

Each file contains the use case, date, full transcript, and generated summary.

## License

[GPL-3.0](LICENSE)

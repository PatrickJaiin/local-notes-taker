# Local Notes Taker

A macOS menu bar app that records audio, transcribes it with Whisper, and generates structured notes using Ollama — all running locally on your machine. No cloud services, no API keys, complete privacy.

## Features

- **Menu bar app** — lives in your macOS menu bar, always one click (or hotkey) away
- **Global hotkey** — start/stop recording from any app with a keyboard shortcut
- **Local transcription** — uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs on CPU, no GPU required)
- **Local summarization** — generates notes via [Ollama](https://ollama.com) with any model you choose
- **Use case presets** — Meeting, Lecture, Brainstorm, Interview, Stand-up, or custom
- **Auto-paste** — summary is copied to clipboard and pasted into your active app
- **Transcript history** — full transcripts and summaries saved to `transcripts/`

## Prerequisites

- **macOS** (uses native menu bar and AppleScript)
- **Python 3.9+**
- **[Ollama](https://ollama.com)** installed and running

## Installation

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
whisper_model: base        # tiny, base, small, medium, large-v3
ollama_model: qwen3:8b     # any model pulled in Ollama
```

| Option | Description |
|---|---|
| `hotkey` | Global keyboard shortcut to toggle recording |
| `whisper_model` | Whisper model size — smaller is faster, larger is more accurate |
| `ollama_model` | Ollama model used for summarization |

## Transcripts

Every recording saves a timestamped file in the `transcripts/` directory:

```
transcripts/2025-06-15_14-30-00_meeting.txt
```

Each file contains the use case, date, full transcript, and generated summary.

## License

[GPL-3.0](LICENSE)

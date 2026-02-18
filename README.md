# Local Notes Taker (Windows)

A Windows system tray app that records audio, transcribes it with Whisper, and generates structured notes using Ollama — all running locally on your machine. No cloud services, no API keys, complete privacy.

## Download

Grab the latest standalone `.zip` from the [Releases page](https://github.com/PatrickJaiin/local-notes-taker/releases). The app bundles Ollama and automatically downloads the Qwen3 model on first launch — no manual installation needed.

## How It Works

1. You press a hotkey (or right-click the tray icon) to **start recording**
2. Press it again to **stop** — the app transcribes your audio using Whisper
3. The transcript is summarized into structured notes by an Ollama LLM
4. The summary is **copied to your clipboard and auto-pasted** into whatever app you're in

Everything runs on your machine. Audio never leaves your computer.

## Features

- **System tray app** — sits in the bottom-right of your taskbar, out of the way
- **Global hotkey** (`Ctrl+Shift+I`) — toggle recording from any app without switching windows
- **Local transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs on CPU, no GPU needed
- **Local summarization** — [Ollama](https://ollama.com) generates notes with any model you choose
- **Use case presets** — Meeting, Lecture, Brainstorm, Interview, Stand-up, or define your own
- **Language support** — auto-detect, or pick from English, Hindi, Malayalam, French, Spanish, German, Japanese, Chinese (or enter any language code)
- **Auto-paste** — summary goes straight into your active app after processing
- **Transcript history** — raw transcripts + summaries saved to `transcripts/` for later reference

## Prerequisites

You need four things installed before setting up the app:

### 1. Python 3.9+

Download from [python.org/downloads](https://www.python.org/downloads/).

**Important:** During installation, check the box that says **"Add Python to PATH"** — otherwise commands like `pip` and `python` won't work in your terminal.

### 2. Git

Download from [git-scm.com/download/win](https://git-scm.com/download/win). Use the default installation options.

### 3. Ollama

Download from [ollama.com](https://ollama.com) and run the Windows installer.

After installing, Ollama runs in the background — you should see the llama icon in your system tray (bottom-right, near the clock). If you don't see it, open the Ollama app from the Start menu.

### 4. Microsoft Visual C++ Redistributable (if not already installed)

Some dependencies (`faster-whisper`, `sounddevice`) need the C++ runtime. Most Windows machines already have it. If you get errors during install, download it from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Installation

Open **PowerShell** (search "PowerShell" in the Start menu) and run these commands one by one:

```powershell
# Clone the repository
git clone https://github.com/PatrickJaiin/local-notes-taker.git

# Navigate into the project folder
cd local-notes-taker

# Switch to the Windows branch
git checkout windows

# Install the app and all its dependencies
pip install -e .

# Download the default Ollama model (about 4.7 GB)
ollama pull qwen3:8b
```

> You can swap `qwen3:8b` for any Ollama model (e.g. `llama3:8b`, `mistral`) — just update `ollama_model` in `config.yaml` to match.

### If something goes wrong

| Problem | Fix |
|---|---|
| `pip` is not recognized | Try `python -m pip install -e .` instead |
| Build errors mentioning C++ | Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), restart PowerShell, try again |
| `ollama pull` hangs or fails | Make sure Ollama is running (check system tray), then retry |

## Usage

### Starting the app

In PowerShell, navigate to the project folder and run:

```powershell
cd local-notes-taker
local-notes
```

A **blue circle** appears in your system tray (bottom-right of your taskbar, near the clock). If you don't see it, click the **^** arrow to expand hidden tray icons.

### Recording

**Option A — Hotkey:** Press **Ctrl + Shift + I** to start recording. Press it again to stop.

**Option B — Tray menu:** Right-click the blue circle → click **"Start Recording"**. Right-click again → **"Stop Recording"**.

### What happens after you stop recording

The tray icon turns **orange** while the app processes your audio:

1. **Transcribing** — converts your speech to text using Whisper
2. **Summarizing** — generates structured notes using Ollama
3. **Copying** — puts the summary on your clipboard
4. **Auto-pasting** — simulates Ctrl+V into whatever app is focused

When done, the icon turns back to **blue** and you get a Windows notification.

### Tray icon colors

| Color | Meaning |
|---|---|
| Blue | Idle — ready to record |
| Red | Recording in progress |
| Orange | Processing (transcribing/summarizing) |

Hover over the icon to see a tooltip with the current status.

### Changing the use case

Right-click tray icon → **Use Case** → pick a preset (Meeting, Lecture, Brainstorm, Interview, Stand-up) or click **Custom...** to type your own. The use case affects how Ollama formats the summary.

### Changing the language

Right-click tray icon → **Language** → pick a language or click **Other...** to enter a language code (e.g. `ta` for Tamil, `ko` for Korean). Set to **Auto-detect** to let Whisper figure it out.

## Configuration

All settings are in `config.yaml` in the project root. Open it with any text editor (Notepad works fine):

```yaml
hotkey: <ctrl>+<shift>+i      # keyboard shortcut to toggle recording
whisper_model: base            # tiny, base, small, medium, large-v3
ollama_model: qwen3:8b         # any model you've pulled with ollama
language:                      # leave empty for auto-detect, or set: en, hi, fr, etc.
```

| Option | What it does | Examples |
|---|---|---|
| `hotkey` | Global keyboard shortcut to toggle recording | `<ctrl>+<shift>+i`, `<ctrl>+<alt>+r` |
| `whisper_model` | Whisper model size — smaller = faster, larger = more accurate | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `ollama_model` | Which Ollama model generates summaries | `qwen3:8b`, `llama3:8b`, `mistral` |
| `language` | Force a transcription language (empty = auto-detect) | `en`, `hi`, `ml`, `fr` |
| `ollama_mode` | `external` (default, uses system Ollama) or `bundled` (uses packaged binary) | `external`, `bundled` |

After editing, restart the app for changes to take effect.

## Transcripts

Every recording saves a file in the `transcripts/` folder:

```
transcripts/2025-06-15_14-30-00_meeting.txt
transcripts/2025-06-15_15-00-00_lecture.txt
```

Each file contains the use case, timestamp, full transcript, and the generated summary.

## License

[GPL-3.0](LICENSE)

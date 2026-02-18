# Building Local Notes

## Prerequisites

- Python 3.9+
- `requests` (`pip install requests`)
- **macOS**: `py2app` (`pip install py2app`)
- **Windows**: `PyInstaller` (`pip install pyinstaller`)

## Step 1: Download Assets

This downloads the Ollama binary and pulls the model into `assets/`:

```bash
# Auto-detect platform
python build/download_assets.py

# Or specify explicitly
python build/download_assets.py --platform macos-arm64
python build/download_assets.py --platform windows-amd64

# Custom model
python build/download_assets.py --model qwen3:8b

# Binary only (skip model pull)
python build/download_assets.py --skip-model
```

This creates:
```
assets/
├── ollama          # (or ollama.exe)
└── models/         # Ollama model blobs
```

## Step 2: Build

### macOS (.dmg)

```bash
python build/build_macos.py
```

Output: `dist/Local Notes.dmg`

### Windows (.exe)

```bash
python build/build_windows.py
```

Output: `dist/LocalNotes.zip` containing the executable folder.

## CI / Automated Releases

The GitHub Actions workflow (`.github/workflows/release.yml`) builds and publishes release artifacts automatically. To trigger a release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This will:
1. Build the macOS `.dmg` from the `main` branch
2. Build the Windows `.zip` from the `windows` branch
3. Create a GitHub Release with both artifacts attached

Releases appear at https://github.com/PatrickJaiin/local-notes-taker/releases.

## Configuration

Set `ollama_mode` in `config.yaml`:

- `external` (default) — uses system Ollama on port 11434
- `bundled` — starts the bundled Ollama binary on a random port

The bundled app ships with `config.yaml` set to `bundled` mode. For development, keep it as `external`.

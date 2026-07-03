# Building Local Notes (macOS, Apple Silicon)

The app is Apple-Silicon-only. All inference backends are bundled; model **weights**
download on demand at runtime, so they are not part of the build.

## Prerequisites

- Apple Silicon Mac, macOS 14+
- Python 3.11 or 3.12 (best wheel coverage for `torch` + the MLX stack)
- `pip install -e .` then `pip install "pyinstaller>=6.14.0"`

## Build

```bash
python build/build_macos.py            # dist/Local Notes.app + dist/Local Notes.dmg (ad-hoc signed)
python build/build_macos.py --no-dmg   # app only
```

This invokes `build/LocalNotes.spec`, which uses `collect_all` for the MLX
packages so the compiled Metal shader libraries (`.metallib`) are bundled —
without that the packaged app fails at launch with *"Failed to load the default
metallib."* Expected DMG size: ~0.7–1.3 GB (weights excluded).

## Signing & notarization (release)

CI (`.github/workflows/release.yml`) builds on an arm64 runner and, when the
secrets below are set, signs with a **Developer ID Application** certificate
(inside-out via `build/sign_app.sh`, never `--deep`), applies the hardened runtime
with `build/entitlements.plist`, and notarizes + staples both the `.app` and the
`.dmg`. Without the secrets it falls back to an ad-hoc signature (Gatekeeper will
warn on other machines).

Required repository secrets:

| Secret | Purpose |
|---|---|
| `MACOS_CERT_P12_BASE64` | base64 of your Developer ID Application `.p12` |
| `MACOS_CERT_PASSWORD` | password for the `.p12` |
| `MACOS_SIGN_IDENTITY` | e.g. `Developer ID Application: Your Name (TEAMID)` |
| `APPLE_ID` | Apple ID email for notarization |
| `APPLE_TEAM_ID` | your Apple Developer Team ID |
| `APPLE_APP_PASSWORD` | app-specific password for `notarytool` |

## Release

```bash
git tag v0.2.0
git push origin v0.2.0
```

The tagged build attaches `Local Notes.dmg` to a GitHub Release.

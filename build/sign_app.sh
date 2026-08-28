#!/usr/bin/env bash
# Sign a Local Notes .app inside-out with a Developer ID Application identity and
# the hardened runtime. Signs every nested Mach-O (dylib/so/metallib + the main
# executable) before the bundle — never use `codesign --deep`, which mis-signs a
# PyInstaller layout.
#
# Usage: SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" build/sign_app.sh "dist/Local Notes.app"
set -euo pipefail

APP="${1:?usage: sign_app.sh <path-to-.app>}"
IDENTITY="${SIGN_IDENTITY:?set SIGN_IDENTITY to your Developer ID Application identity}"
ENTITLEMENTS="$(cd "$(dirname "$0")" && pwd)/entitlements.plist"

echo "Signing nested binaries in $APP ..."
# Sign deepest-first so containing bundles are signed after their contents.
find "$APP/Contents" \
  \( -name "*.dylib" -o -name "*.so" -o -name "*.metallib" \) -type f -print0 \
  | while IFS= read -r -d '' f; do
      codesign --force --timestamp --options runtime --sign "$IDENTITY" "$f"
    done

# Sign any other Mach-O executables (e.g. nested helper binaries).
find "$APP/Contents/MacOS" -type f -perm -111 -print0 \
  | while IFS= read -r -d '' f; do
      codesign --force --timestamp --options runtime \
        --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$f" || true
    done

echo "Signing app bundle..."
codesign --force --timestamp --options runtime \
  --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"

echo "Verifying signature..."
codesign --verify --strict --verbose=2 "$APP"
echo "Done."

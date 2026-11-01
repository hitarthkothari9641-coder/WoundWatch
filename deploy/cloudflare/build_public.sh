#!/usr/bin/env bash
# Copy the web UI (and performance data) into deploy/cloudflare/public for Workers assets.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"; pub="$root/deploy/cloudflare/public"
rm -rf "$pub"; mkdir -p "$pub/assets"
cp -R "$root/service/web/." "$pub/assets/"
mv "$pub/assets/index.html" "$pub/index.html"
# Same React Native app as iOS/Android, exported for browsers: scan the QR code, no Expo account needed.
(cd "$root/mobile" && npx expo export --platform web --output-dir "$pub/app" >/dev/null)
cp "$root/deploy/cloudflare/_headers" "$pub/_headers"
echo "Built $pub"

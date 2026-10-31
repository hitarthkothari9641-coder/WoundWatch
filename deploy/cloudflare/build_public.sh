#!/usr/bin/env bash
# Copy the web UI (and performance data) into deploy/cloudflare/public for Workers assets.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"; pub="$root/deploy/cloudflare/public"
rm -rf "$pub"; mkdir -p "$pub/assets"
cp -R "$root/service/web/." "$pub/assets/"
mv "$pub/assets/index.html" "$pub/index.html"
cp "$root/deploy/cloudflare/_headers" "$pub/_headers"
echo "Built $pub"

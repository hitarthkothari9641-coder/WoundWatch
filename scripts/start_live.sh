#!/usr/bin/env bash
# Start the backend + a free Cloudflare quick tunnel, then point the deployed Worker at the new tunnel host.
# Run in a terminal tab and leave it open; Ctrl+C stops everything. Quick-tunnel hosts change on every start.
# shortcut: laptop-hosted demo backend, replace with a small always-on VM/container service for real use.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; cd "$root"
mkdir -p dist/runtime
[ -f dist/origin_secret.txt ] || { openssl rand -hex 24 > dist/origin_secret.txt; chmod 600 dist/origin_secret.txt; }
secret="$(cat dist/origin_secret.txt)"
pkill -f "uvicorn service.app:app --host 127.0.0.1 --port 8000" || true
pkill -f "cloudflared tunnel" || true
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

dist/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 > dist/tunnel.log 2>&1 &
host=""
for _ in $(seq 40); do host="$(grep -o '[a-z0-9-]*\.trycloudflare\.com' dist/tunnel.log | head -1 || true)"; [ -n "$host" ] && break; sleep 1; done
[ -n "$host" ] || { echo "Tunnel did not start; see dist/tunnel.log"; exit 1; }

PYTHONPATH=python WOUNDWATCH_ENV=production WOUNDWATCH_DATA_DIR="$root/dist/runtime" \
  WOUNDWATCH_ALLOWED_HOSTS="$host,127.0.0.1,localhost" WOUNDWATCH_CLASSIFIER_PATH="$root/models/classifier_v2.pt" \
  WOUNDWATCH_AREA_GATE=advisory WOUNDWATCH_ORIGIN_SECRET="$secret" \
  .venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8000 --no-access-log --proxy-headers > dist/backend.log 2>&1 &
for _ in $(seq 60); do curl -fs -m 2 http://127.0.0.1:8000/live >/dev/null && break; sleep 1; done

(cd deploy/cloudflare && ./build_public.sh >/dev/null \
  && npx wrangler secret put ORIGIN_SECRET < ../../dist/origin_secret.txt >/dev/null \
  && npx wrangler deploy --var "BACKEND_HOST:$host" >/dev/null)
sleep 3
echo "Tunnel: https://$host"
curl -s -m 10 -w "  direct without secret: HTTP %{http_code} (expect 403)\n" "https://$host/api/auth/me" -o /dev/null
curl -s -m 10 -w "  through Worker:        HTTP %{http_code} (expect 401 = up, not signed in)\n" https://wound-trace.dvpatel.workers.dev/api/auth/me -o /dev/null
echo "LIVE at https://wound-trace.dvpatel.workers.dev  (keep this tab open; Ctrl+C to stop)"
wait

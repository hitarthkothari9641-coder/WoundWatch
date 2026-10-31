# Free hosting: Cloudflare edge + Hugging Face Space

Architecture: browser -> Cloudflare Worker (static UI from the edge, security headers, rate limit) -> Hugging Face Space
(FastAPI + PyTorch, CPU). The Space only answers requests carrying `X-Origin-Secret`. Demo data is synthetic and lives on the
Space's ephemeral disk; do not upload real patient photographs.

1. Space: create a Docker Space on huggingface.co, then
   `EXTRA_MODELS=classifier_v2.pt deploy/hf/build_space.sh`, `cd dist/hf-space`, `git lfs install`, push to the Space remote.
   Space variables: `WOUNDWATCH_ALLOWED_HOSTS=<owner>-<space>.hf.space,127.0.0.1`, `WOUNDWATCH_CLASSIFIER_PATH=/app/models/classifier_v2.pt`,
   `WOUNDWATCH_AREA_GATE=advisory`. Space secret: `WOUNDWATCH_ORIGIN_SECRET` (random 32+ chars).
2. Worker: `deploy/cloudflare/build_public.sh`; set `BACKEND_HOST` in `wrangler.toml`; `cd deploy/cloudflare`;
   `npx wrangler login`; `npx wrangler secret put ORIGIN_SECRET` (same value as above); `npx wrangler deploy`.
3. Accounts: patients register in the UI. Clinician accounts: `python -m service.manage create-user ...` (run inside the Space terminal or a restart script).

Dataset and weight redistribution rights are unverified (see DEPLOYMENT.md): confirm before making weights public.

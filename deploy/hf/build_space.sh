#!/usr/bin/env bash
# Assemble a Hugging Face Space directory from this repo: dist/hf-space
# Usage: deploy/hf/build_space.sh  (then cd dist/hf-space, git lfs track "*.pt" "*.onnx", push to the Space remote)
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"; out="$root/dist/hf-space"
rm -rf "$out"; mkdir -p "$out/models" "$out/python"
cp "$root/deploy/hf/Dockerfile" "$root/deploy/hf/README.md" "$out/"
cp -R "$root/service" "$out/service"; rm -rf "$out/service/__pycache__"
cp -R "$root/python/woundwatch" "$out/python/woundwatch"; rm -rf "$out/python/woundwatch/experiments" "$out/python/woundwatch/__pycache__"
# .pt only: the service loads PyTorch checkpoints; ONNX is not used at runtime.
for f in classifier.pt segmenter.pt ${EXTRA_MODELS:-}; do cp "$root/models/$f" "$out/models/$f"; done
printf '*.pt filter=lfs diff=lfs merge=lfs -text\n' > "$out/.gitattributes"
echo "Built $out"

#!/usr/bin/env bash
# Re-download the Wuji asset bundle and refresh data/ and assets/wuji_policies/.
# Only needed if you want to update the vendored clips/policies.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.10.22/wuji-mjlab-v2026.10.22-assets.zip
TMP=$(mktemp -d)
curl -L -o "$TMP/assets.zip" "$URL"
unzip -q "$TMP/assets.zip" -d "$TMP"
B="$TMP/wuji-mjlab-v2026.10.22-assets"
rm -rf data/wuji_pen_spin_clips data/wuji_recipes
cp -r "$B/pen_spin/data/rawdata/clips" data/wuji_pen_spin_clips
cp -r "$B/pen_spin/data/recipe" data/wuji_recipes
for g in "$B"/pen_spin/checkpoints/*/; do
  name=$(basename "$g"); mkdir -p "assets/wuji_policies/$name"
  cp "$g/policy.onnx" "$g/config.json" "assets/wuji_policies/$name/"
done
echo "refreshed data/ and assets/wuji_policies/ from $URL"

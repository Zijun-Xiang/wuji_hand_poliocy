#!/usr/bin/env bash
# Set up Wuji's OFFICIAL pen-spin environment (wuji-mjlab, mjlab + MuJoCo Warp).
# A CPU is enough for evaluation: all 40 clips take about 4 minutes.
#
#   bash scripts/official_env/setup.sh [target_dir]     (default: ../wuji-mjlab)
set -euo pipefail
ROOT=$(realpath -m "${1:-$(dirname "$0")/../../../wuji-mjlab}")
RELEASE=v2026.10.22
COMMIT=b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9   # "Public release v2026.10.22"

if [ ! -d "$ROOT/.git" ]; then
  git clone https://github.com/wuji-technology/wuji-mjlab.git "$ROOT"
fi
git -C "$ROOT" checkout "$COMMIT"
git -C "$ROOT" submodule update --init --depth 1        # wuji-description: the Hand 2 MJCF

# released clips, recipes and checkpoints go where wuji-mjlab looks for them
TMP=$(mktemp -d)
curl -L -o "$TMP/assets.zip" "https://github.com/wuji-technology/wuji-mjlab/releases/download/$RELEASE/wuji-mjlab-$RELEASE-assets.zip"
unzip -q "$TMP/assets.zip" -d "$TMP"
SRC="$TMP/wuji-mjlab-$RELEASE-assets/pen_spin"
DST="$ROOT/src/wuji_mjlab/tasks/pen_spin/data"
mkdir -p "$DST"
cp -r "$SRC/data/rawdata" "$SRC/data/recipe" "$SRC/checkpoints" "$DST/"

# CPU-only Python environment with the versions pinned in wuji-mjlab's pixi.toml
# (mjlab 1.6.0, rsl-rl-lib 5.4.2, torch 2.7.x); tested with Python 3.13.
python3 -m venv "$ROOT/.venv-cpu"
# shellcheck disable=SC1091
. "$ROOT/.venv-cpu/bin/activate"
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
# tensordict 0.9.1 is the last release that works with torch 2.7; without the pin pip upgrades torch
pip install mjlab==1.6.0 rsl-rl-lib==5.4.2 tensordict==0.9.1 torch==2.7.1 torchvision==0.22.1 \
  --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-deps -e "$ROOT"
python -c "import warp as wp; wp.init(); print('warp devices:', wp.get_devices())"
echo "official environment ready in $ROOT (activate with: . $ROOT/.venv-cpu/bin/activate)"

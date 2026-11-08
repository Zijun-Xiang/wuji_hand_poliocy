#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 figs.py
node render.js note.src.html note.html
google-chrome --headless --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=5000 \
  --print-to-pdf=raw.pdf "file://$PWD/note.html" 2>/dev/null
python3 footer.py raw.pdf penspin_graph_step_by_step.pdf

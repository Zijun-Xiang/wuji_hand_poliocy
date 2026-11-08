#!/usr/bin/env bash
# End-to-end run. Usage: scripts/run_pipeline.sh runs/demo [workers]
set -euo pipefail
RUN=${1:-runs/demo}; W=${2:-4}
penspin-graph --run "$RUN" build
penspin-graph --run "$RUN" validate --workers "$W" --follow-frames 50
penspin-graph --run "$RUN" loops --min-rotation-deg 180
penspin-graph --run "$RUN" replay --loop 0 --repeats 3 --save
penspin-graph --run "$RUN" eval --n-starts 20 --horizon 250
penspin-graph --run "$RUN" export --seconds 20 --n-walks 20
penspin-graph --run "$RUN" train-bc --n-walks 200 --epochs 40
penspin-graph --run "$RUN" viz
penspin-graph --run "$RUN" report

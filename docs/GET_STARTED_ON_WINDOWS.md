# Getting this repo onto your laptop and GitHub (Windows)

Two one-time steps that needed your login, so they were left for you.

## A. Put the code in `C:\Users\xiang\Documents\GitHub\wuji_hand_poliocy`

Option 1 - from the zip (`penspin-graph.zip`):

```powershell
Expand-Archive penspin-graph.zip -DestinationPath C:\Users\xiang\Documents\GitHub\
Rename-Item C:\Users\xiang\Documents\GitHub\penspin-graph wuji_hand_poliocy
cd C:\Users\xiang\Documents\GitHub\wuji_hand_poliocy
git init -b main; git add -A; git commit -m "penspin-graph: initial import"
```

Option 2 - from the git bundle (`penspin-graph.bundle`, keeps the commit history):

```powershell
cd C:\Users\xiang\Documents\GitHub
git clone penspin-graph.bundle wuji_hand_poliocy
```

## B. Create the GitHub repo and push

1. https://github.com/new -> name `wuji_hand_poliocy` (or `penspin-graph`), **empty** (no README/.gitignore/licence).
2. Then:

```powershell
cd C:\Users\xiang\Documents\GitHub\wuji_hand_poliocy
git remote remove origin 2>$null
git remote add origin https://github.com/Zijun-Xiang/wuji_hand_poliocy.git
git push -u origin main
```

## C. Run it (Windows, no GPU needed)

```powershell
cd C:\Users\xiang\Documents\GitHub\wuji_hand_poliocy
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest                                   # ~1-2 min
penspin-graph --run runs/full3 replay --loop 0 --repeats 3      # best stitched loop, 45 s of spinning in sim
penspin-graph --run runs/full3 eval --n-starts 10 --horizon 500 --restrict-scc
penspin-graph --run runs/full3 render --loop 0 --format gif     # GLFW rendering works on the desktop
```

(`runs/full3/graph/graph.npz` already contains the 38,624 validated jumps, so
nothing has to be re-validated to play with the results.)

## D. PPO on the stitched clips (WSL2 + wuji-mjlab, RTX 4060)

```bash
penspin-graph --run runs/full3 export --seconds 20 --n-walks 20
WUJI=~/wuji-mjlab
cp -r runs/full3/export/clips/06_graph_stitched $WUJI/src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/
cp runs/full3/export/recipe/06_graph_stitched.json $WUJI/src/wuji_mjlab/tasks/pen_spin/data/recipe/
cd $WUJI && pixi run train --task WujiHand2_PenSpin \
  env.commands.motion.recipe_path=src/wuji_mjlab/tasks/pen_spin/data/recipe/06_graph_stitched.json \
  env.scene.num_envs=2048 agent.max_iterations=10000 agent.run_name=PenSpin_06_graph_stitched
```

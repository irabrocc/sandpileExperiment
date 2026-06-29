# Sandpile Statistical Experiments

Abelian sandpile model experiments — 2D and 3D — for studying patterns,
symmetries, and avalanche statistics.

## Experiment Paradigms

### A Perturbation Mode (per-trial reset)
Each trial: fill grid → perturb 3 random sites → stabilise → record.
Repeated many times to study typical patterns.

## Quick Start

```bash
pip install numpy matplotlib tqdm pytest

# ---- Perturbation experiments ----
python main.py --mode 2d --N 200 --trials 200
python main.py --mode 3d --n 30 --trials 200
python main.py --all
python main.py --analyze 2d      # stats + pattern analysis (JSON)
python main.py --analyze 3d      # stats + pattern analysis (JSON)

# ---- Interactive viewers ----
python main.py --view2d              # browse 2D grid snapshots
python main.py --view2d --trial 100  # start from trial 100
python main.py --view3d              # browse 3D grid z-slices
python main.py --view3d --trial 42   # start from trial 42

# ---- Tests ----
pytest test_sandpile_2d.py -v
```

## Interactive Viewers

Interactive matplotlib windows for browsing sandpile grid snapshots in real time.

### 2D Viewer (`--view2d`)

| Control        | Action                           |
|---             |---                               |
| Trial slider   | Switch between trials (0 to N−1) |
| `←` / `A`      | Previous trial                   |
| `→` / `D`      | Next trial                       |
| Mouse hover    | Live 18×18 patch preview         |
| Left-click / S | Save 18×18 patch as annotated PNG |

- Red ★ markers show the 3 perturbation sites for each trial.
- Title bar displays trial index, total topple count, and distinct site count.
- Colorbar shows the fixed 7-colour grain-count map.

#### Patch Feature (18×18 Local Inspection)

Hover the mouse over any grid cell to see a live **18×18 preview inset**
(top-left corner of the window), showing the local neighbourhood around the
cursor.  Left-click or press `S` to **save** the patch as an annotated PNG:

- Every cell is labelled with its exact grain count (black text for ≤2,
  white for ≥3).
- Files are named `patch_trial_NNNNNN_xX_yY.png` and saved to
  `outputs/2d/local_patches/`.
- If no mouse position is available, the `S` key falls back to the grid
  centre.

This is the primary tool for examining local periodic colour structures
along directional lines, inspecting grain-count patterns at intersection
points, and capturing precise configurations for inclusion in the report.

### 3D Viewer (`--view3d`)

| Control        | Action                           |
|---             |---                               |
| Trial slider   | Switch between trials            |
| z-slice slider | Scroll through depth layers      |
| `←` / `A`      | z-slice up one layer             |
| `→` / `D`      | z-slice down one layer           |
| `↑` / `W`      | Previous trial                   |
| `↓` / `S`      | Next trial                       |

- Displays a 2D slice of the 3D grid at the selected z-depth.
- Grid is cached per trial — switching z within the same trial is instant.

## Experiment Description

|                  | 2D                          | 3D                         |
|---               |---                          |---                         |
| Grid             | N × N (default 200)         | n × n × n (default 100)    |
| Initial grains   | 3                           | 5                          |
| Topple threshold | 4                           | 6                          |
| Perturbation     | +1 grain to 3 random sites  | +1 grain to 3 random sites |
| Boundary         | Open (grains lost at edges) | Open                       |

## Project Structure

```
statisticsTask/
├── config.py              # Centralized parameters
├── sandpile_2d.py         # 2D sandpile engine
├── sandpile_3d.py         # 3D sandpile engine
├── experiment_2d.py       # 2D perturbation experiment runner
├── experiment_3d.py       # 3D perturbation experiment runner
├── visualize.py           # Interactive viewers + colour map
├── analyze.py             # Statistical & pattern analysis
├── main.py                # CLI entry point
├── test_sandpile_2d.py    # Unit tests
├── main.tex               # Report: main document
├── basicSetting.tex       # Report: experimental setup
├── classification.tex     # Report: pattern classification
├── observedRegularities.tex # Report: observed regularities
├── observed_patterns.md   # Raw observation notes
├── images/                # Figures for the report
└── outputs/
    ├── 2d/
    │   ├── grids/          # .npy final grids
    │   ├── local_patches/  # saved 18×18 patches (PNG)
    │   ├── images/         # .png visualizations
    │   └── stats/          # .json statistics
    └── 3d/
        ├── grids/
        ├── images/
        └── stats/
```

## Configuration

Edit `config.py` to adjust grid sizes, trial counts, and thresholds.

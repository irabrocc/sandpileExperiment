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
python main.py --analyze 2d
python main.py --analyze 3d

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

| Control      | Action                           |
|---           |---                               |
| Trial slider | Switch between trials (0 to N−1) |
| `←` / `A`    | Previous trial                   |
| `→` / `D`    | Next trial                       |

- Red ★ markers show the 3 perturbation sites for each trial.
- Title bar displays trial index, total topple count, and distinct site count.
- Colorbar auto-adjusts to the grain range.

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
├── continuous_drive.py    # Continuous-drive SOC experiments + power-law fit
├── visualize.py           # Plotting utilities (2D, 3D, power-law)
├── analyze.py             # Statistical & pattern analysis
├── main.py                # CLI entry point
├── test_sandpile_2d.py    # Unit tests
├── notes.md               # Observations log
└── outputs/
    ├── 2d/
    │   ├── grids/          # .npy final grids
    │   ├── images/         # .png visualizations
    │   └── stats/          # .json statistics
    └── 3d/
        ├── grids/
        ├── images/
        └── stats/
```

## Configuration

Edit `config.py` to adjust grid sizes, trial counts, and thresholds.

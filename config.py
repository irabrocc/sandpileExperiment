"""
Centralized configuration for sandpile experiments.
"""

import os

# ---- Grid sizes ----
N_2D = 200          # 2D grid side length
N_3D = 100           # 3D grid side length (keep modest: O(n^3) memory)

# ---- Sandpile parameters ----
TOPPLE_THRESHOLD_2D = 4
TOPPLE_THRESHOLD_3D = 6
INITIAL_GRAINS_2D = 3
INITIAL_GRAINS_3D = 5
NUM_PERTURB_SITES = 3   # number of random sites to perturb per trial
PERTURB_AMOUNT = 1      # grains added to each perturbed site

# ---- Experiment ----
NUM_TRIALS_2D = 2000
NUM_TRIALS_3D = 2000
SAVE_INTERVAL = 100     # save intermediate results every N trials

# ---- Output directories ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_2D_DIR = os.path.join(OUTPUT_DIR, "2d")
OUTPUT_3D_DIR = os.path.join(OUTPUT_DIR, "3d")
GRIDS_2D_DIR = os.path.join(OUTPUT_2D_DIR, "grids")
IMAGES_2D_DIR = os.path.join(OUTPUT_2D_DIR, "images")
STATS_2D_DIR = os.path.join(OUTPUT_2D_DIR, "stats")
GRIDS_3D_DIR = os.path.join(OUTPUT_3D_DIR, "grids")
IMAGES_3D_DIR = os.path.join(OUTPUT_3D_DIR, "images")
STATS_3D_DIR = os.path.join(OUTPUT_3D_DIR, "stats")

# ---- Random seed ----
BASE_SEED = 42          # base seed for reproducibility; trial seed = BASE_SEED + trial_index


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [OUTPUT_DIR, OUTPUT_2D_DIR, OUTPUT_3D_DIR,
              GRIDS_2D_DIR, IMAGES_2D_DIR, STATS_2D_DIR,
              GRIDS_3D_DIR, IMAGES_3D_DIR, STATS_3D_DIR]:
        os.makedirs(d, exist_ok=True)

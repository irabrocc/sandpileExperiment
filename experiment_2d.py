"""
2D Sandpile Experiment Runner.

For each trial:
  1. Create an N x N grid filled with INITIAL_GRAINS_2D (default 3).
  2. Pick NUM_PERTURB_SITES (default 3) random distinct sites.
  3. Add PERTURB_AMOUNT (default 1) grain to each.
  4. Stabilise.
  5. Record the final grid, toppling statistics, and seed.

Results are saved incrementally (every SAVE_INTERVAL trials) for crash
recovery.
"""

import json
import os
import time
import numpy as np
from tqdm import tqdm

import config
from sandpile_2d import Sandpile2D


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(N: int, seed: int | None = None,
                     threshold: int | None = None,
                     initial_grains: int | None = None,
                     num_perturb: int | None = None,
                     perturb_amount: int | None = None) -> dict:
    """Run one sandpile trial and return a result dictionary.

    Parameters
    ----------
    N : int
        Grid side length.
    seed : int | None
        Random seed for reproducibility.
    threshold : int | None
        Toppling threshold; defaults to ``config.TOPPLE_THRESHOLD_2D``.
    initial_grains : int | None
        Grains on every site before perturbation;
        defaults to ``config.INITIAL_GRAINS_2D``.
    num_perturb : int | None
        Number of random sites to perturb;
        defaults to ``config.NUM_PERTURB_SITES``.
    perturb_amount : int | None
        Grains added to each perturbed site;
        defaults to ``config.PERTURB_AMOUNT``.

    Returns
    -------
    dict with keys:
        grid        – np.ndarray, final stable grid (N x N)
        topples     – int, total toppling count
        distinct    – int, number of distinct sites that toppled
        max_grains  – int
        min_grains  – int
        distribution – dict {grains: count}
        seed        – int, the seed used
        perturb_sites – list of (x,y), the perturbed positions
    """
    if threshold is None:
        threshold = config.TOPPLE_THRESHOLD_2D
    if initial_grains is None:
        initial_grains = config.INITIAL_GRAINS_2D
    if num_perturb is None:
        num_perturb = config.NUM_PERTURB_SITES
    if perturb_amount is None:
        perturb_amount = config.PERTURB_AMOUNT

    rng = np.random.default_rng(seed)
    sp = Sandpile2D(N, threshold=threshold)
    sp.fill(initial_grains)

    # Pick random distinct perturbation sites
    all_indices = list(np.ndindex(N, N))
    chosen_idx = rng.choice(len(all_indices), size=num_perturb, replace=False)
    perturb_sites = [all_indices[i] for i in chosen_idx]

    sp.add_grains_at(perturb_sites, amount=perturb_amount)
    stats = sp.stabilize()

    return {
        "grid": sp.copy_grid(),
        "topples": stats["total_topples"],
        "distinct": stats["distinct_sites"],
        "max_grains": stats["max_grains"],
        "min_grains": stats["min_grains"],
        "distribution": sp.grain_distribution(),
        "seed": seed if seed is not None else -1,
        "perturb_sites": perturb_sites,
    }


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

def run_batch(N: int, num_trials: int,
              output_dir: str | None = None,
              threshold: int | None = None,
              initial_grains: int | None = None,
              base_seed: int | None = None,
              num_perturb: int | None = None,
              perturb_amount: int | None = None) -> list[dict]:
    """Run *num_trials* sandpile experiments.

    Any parameter left as ``None`` falls back to the value defined in
    :mod:`config`, which is the single source of truth. Pass explicit
    values (as ``main.py`` does) to override per-run.

    Parameters
    ----------
    N : int
        Grid side length.
    num_trials : int
        Number of trials to run.
    output_dir : str | None
        Directory to save grids and stats. If None, uses config.
    threshold : int | None
        Toppling threshold.
    initial_grains : int | None
        Initial grains on every site.
    base_seed : int | None
        Base seed; trial i uses base_seed + i.
    num_perturb : int | None
        Number of random sites to perturb per trial.
    perturb_amount : int | None
        Grains added to each perturbed site.

    Returns
    -------
    list of result dicts (one per trial, without the 'grid' key to save
    memory; grids are saved to disk separately).
    """
    if output_dir is None:
        output_dir = config.OUTPUT_2D_DIR
    if threshold is None:
        threshold = config.TOPPLE_THRESHOLD_2D
    if initial_grains is None:
        initial_grains = config.INITIAL_GRAINS_2D
    if base_seed is None:
        base_seed = config.BASE_SEED
    if num_perturb is None:
        num_perturb = config.NUM_PERTURB_SITES
    if perturb_amount is None:
        perturb_amount = config.PERTURB_AMOUNT

    grids_dir = os.path.join(output_dir, "grids")
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    all_results = []       # metadata only (no grids in memory)

    start_time = time.time()

    for trial_idx in tqdm(range(num_trials), desc=f"2D N={N}"):
        seed = base_seed + trial_idx
        result = run_single_trial(
            N, seed=seed,
            threshold=threshold,
            initial_grains=initial_grains,
            num_perturb=num_perturb,
            perturb_amount=perturb_amount,
        )

        # Save grid to disk (as .npy), keep metadata in memory
        grid_path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
        np.save(grid_path, result["grid"])

        meta = {k: v for k, v in result.items() if k != "grid"}
        all_results.append(meta)

    # Save raw metadata
    elapsed = time.time() - start_time
    meta_path = os.path.join(stats_dir, "all_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. {num_trials} trials in {elapsed:.1f}s "
          f"({num_trials / elapsed:.1f} trials/s)")

    return all_results




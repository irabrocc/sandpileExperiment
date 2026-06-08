"""
3D Sandpile Experiment Runner.

For each trial:
  1. Create an n x n x n cube filled with INITIAL_GRAINS_3D (default 5).
  2. Pick NUM_PERTURB_SITES (default 3) random distinct sites.
  3. Add PERTURB_AMOUNT (default 1) grain to each (5+1=6 = threshold → toppling).
  4. Stabilise.
  5. Record the final grid, toppling statistics, and seed.
"""

import json
import os
import time
import numpy as np
from tqdm import tqdm

import config
from sandpile_3d import Sandpile3D


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(n: int, seed: int | None = None,
                     threshold: int = 6,
                     initial_grains: int = 5,
                     num_perturb: int = 3,
                     perturb_amount: int = 1) -> dict:
    """Run one 3D sandpile trial.

    Parameters
    ----------
    n : int
        Cube side length.
    seed : int | None
        Random seed.
    threshold : int
        Toppling threshold (default 6 for 3D).
    initial_grains : int
        Grains on every site before perturbation (default 5).
    num_perturb : int
        Number of random sites to perturb.
    perturb_amount : int
        Grains added to each perturbed site.

    Returns
    -------
    dict with keys:
        grid          – np.ndarray, final stable grid (n x n x n)
        topples       – int
        distinct      – int
        max_grains    – int
        min_grains    – int
        distribution  – dict {grains: count}
        seed          – int
        perturb_sites – list of (x,y,z)
    """
    rng = np.random.default_rng(seed)
    sp = Sandpile3D(n, threshold=threshold)
    sp.fill(initial_grains)

    # Pick random distinct perturbation sites
    all_indices = list(np.ndindex(n, n, n))
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

def run_batch(n: int, num_trials: int,
              output_dir: str | None = None,
              threshold: int = 6,
              initial_grains: int = 5,
              base_seed: int = 42,
              save_interval: int = 100) -> list[dict]:
    """Run *num_trials* 3D sandpile experiments.

    Parameters
    ----------
    n : int
        Cube side length.
    num_trials : int
        Number of trials.
    output_dir : str | None
        Directory to save grids and stats.
    threshold : int
    initial_grains : int
    base_seed : int
    save_interval : int

    Returns
    -------
    list of metadata dicts (without grid arrays).
    """
    if output_dir is None:
        output_dir = config.OUTPUT_3D_DIR

    grids_dir = os.path.join(output_dir, "grids")
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    all_results = []
    topple_list = []
    distinct_list = []
    max_grain_list = []
    distributions = []

    start_time = time.time()

    for trial_idx in tqdm(range(num_trials), desc=f"3D n={n}"):
        seed = base_seed + trial_idx
        result = run_single_trial(
            n, seed=seed,
            threshold=threshold,
            initial_grains=initial_grains,
            num_perturb=config.NUM_PERTURB_SITES,
            perturb_amount=config.PERTURB_AMOUNT,
        )

        # Save grid to disk
        grid_path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
        np.save(grid_path, result["grid"])

        meta = {k: v for k, v in result.items() if k != "grid"}
        all_results.append(meta)
        topple_list.append(result["topples"])
        distinct_list.append(result["distinct"])
        max_grain_list.append(result["max_grains"])

        for g, c in result["distribution"].items():
            while len(distributions) <= g:
                distributions.append(0)
            distributions[g] += c

        if (trial_idx + 1) % save_interval == 0 or trial_idx == num_trials - 1:
            _save_intermediate(
                trial_idx + 1, all_results, topple_list, distinct_list,
                max_grain_list, distributions, stats_dir, start_time
            )

    _save_final(all_results, topple_list, distinct_list, max_grain_list,
                distributions, stats_dir, start_time, num_trials)

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_intermediate(done: int, all_results, topple_list, distinct_list,
                       max_grain_list, distributions, stats_dir, start_time):
    elapsed = time.time() - start_time
    agg = {
        "trials_completed": done,
        "elapsed_sec": round(elapsed, 1),
        "mean_topples": float(np.mean(topple_list)),
        "median_topples": float(np.median(topple_list)),
        "max_topples": int(np.max(topple_list)),
        "min_topples": int(np.min(topple_list)),
        "mean_distinct_sites": float(np.mean(distinct_list)),
        "max_grains_observed": int(np.max(max_grain_list)),
        "distribution": distributions,
    }
    path = os.path.join(stats_dir, f"aggregate_{done:06d}.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)


def _save_final(all_results, topple_list, distinct_list, max_grain_list,
                distributions, stats_dir, start_time, num_trials):
    elapsed = time.time() - start_time

    meta_path = os.path.join(stats_dir, "all_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary = {
        "num_trials": num_trials,
        "n": config.N_3D if "config" in globals() else 0,
        "total_elapsed_sec": round(elapsed, 1),
        "trials_per_sec": round(num_trials / elapsed, 2) if elapsed > 0 else 0,
        "topple_stats": {
            "mean": float(np.mean(topple_list)),
            "median": float(np.median(topple_list)),
            "std": float(np.std(topple_list)),
            "min": int(np.min(topple_list)),
            "max": int(np.max(topple_list)),
        },
        "distinct_site_stats": {
            "mean": float(np.mean(distinct_list)),
            "median": float(np.median(distinct_list)),
            "std": float(np.std(distinct_list)),
        },
        "max_grains_overall": int(np.max(max_grain_list)),
        "distribution": distributions,
    }
    summary_path = os.path.join(stats_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {num_trials} trials in {elapsed:.1f}s "
          f"({num_trials / elapsed:.1f} trials/s)")
    print(f"Topples — mean: {summary['topple_stats']['mean']:.1f}, "
          f"max: {summary['topple_stats']['max']}")

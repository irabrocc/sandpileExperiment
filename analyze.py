"""
Statistical analysis for sandpile experiments.

Computes aggregate statistics across trials and detects patterns
(recurring structures, symmetries, unexpected behaviour).
"""

import json
import os
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_metadata(stats_dir: str) -> list[dict]:
    """Load all trial metadata from all_metadata.json."""
    path = os.path.join(stats_dir, "all_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_grid(grids_dir: str, trial_idx: int) -> np.ndarray:
    """Load a single saved grid."""
    path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Grid not found: {path}")
    return np.load(path)


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_statistics(metadata: list[dict]) -> dict:
    """Compute summary statistics from trial metadata.

    Returns a dict with:
        num_trials, topple_stats, distinct_stats, max_grains_stats,
        avalanche_size_distribution, grain_distribution_aggregate
    """
    topples = np.array([m["topples"] for m in metadata])
    distincts = np.array([m["distinct"] for m in metadata])
    max_grains = np.array([m["max_grains"] for m in metadata])

    # Aggregate grain distribution
    dist_agg = Counter()
    for m in metadata:
        for g, c in m["distribution"].items():
            dist_agg[int(g)] += c

    return {
        "num_trials": len(metadata),
        "topple_stats": {
            "mean": float(np.mean(topples)),
            "median": float(np.median(topples)),
            "std": float(np.std(topples)),
            "min": int(np.min(topples)),
            "max": int(np.max(topples)),
            "p25": float(np.percentile(topples, 25)),
            "p75": float(np.percentile(topples, 75)),
            "p95": float(np.percentile(topples, 95)),
            "p99": float(np.percentile(topples, 99)),
        },
        "distinct_site_stats": {
            "mean": float(np.mean(distincts)),
            "median": float(np.median(distincts)),
            "std": float(np.std(distincts)),
        },
        "max_grains_stats": {
            "mean": float(np.mean(max_grains)),
            "max": int(np.max(max_grains)),
        },
        "avalanche_sizes": topples.tolist(),
        "grain_distribution_aggregate": dict(sorted(dist_agg.items())),
    }


# ---------------------------------------------------------------------------
# Pattern detection (2D)
# ---------------------------------------------------------------------------

def detect_patterns(grid: np.ndarray, background: int = 3) -> dict:
    """Identify basic patterns in a stable sandpile grid.

    Parameters
    ----------
    grid : np.ndarray, shape (N, N)
    background : int
        The background grain count (3 for 2D experiments).

    Returns
    -------
    dict with:
        changed_fraction : float
            Fraction of sites whose grain count differs from background.
        grain_counts : Counter
            Distribution of grain values.
        bounding_box : tuple
            (x_min, x_max, y_min, y_max) of the non-background region.
    """
    N = grid.shape[0]
    mask = (grid != background)
    changed_fraction = mask.sum() / (N * N)

    grain_counts = Counter(int(v) for v in grid.flatten())

    # Bounding box
    ys, xs = np.where(mask)
    if len(xs) > 0:
        bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    else:
        bbox = (0, 0, 0, 0)

    return {
        "changed_fraction": float(changed_fraction),
        "grain_counts": dict(sorted(grain_counts.items())),
        "bounding_box": bbox,
    }


def compute_symmetry(grid: np.ndarray) -> dict:
    """Check basic symmetries of a sandpile configuration.

    Returns a dict of correlation coefficients for:
        - horizontal reflection (left-right)
        - vertical reflection (top-bottom)
        - 180-degree rotation
        - 90-degree rotation (only if N is square)
    """
    corr = {}

    # Horizontal reflection
    flipped_lr = np.fliplr(grid)
    corr["horizontal_reflection"] = float(np.corrcoef(
        grid.ravel(), flipped_lr.ravel())[0, 1])

    # Vertical reflection
    flipped_ud = np.flipud(grid)
    corr["vertical_reflection"] = float(np.corrcoef(
        grid.ravel(), flipped_ud.ravel())[0, 1])

    # 180-degree rotation (equivalent to flip both axes)
    rotated_180 = np.rot90(grid, 2)
    corr["rotation_180"] = float(np.corrcoef(
        grid.ravel(), rotated_180.ravel())[0, 1])

    # 90-degree rotation
    rotated_90 = np.rot90(grid, 1)
    corr["rotation_90"] = float(np.corrcoef(
        grid.ravel(), rotated_90.ravel())[0, 1])

    return corr


def analyze_patterns_across_trials(grids_dir: str, metadata: list[dict],
                                   max_samples: int = 200) -> dict:
    """Analyze patterns across multiple trials.

    Loads up to *max_samples* grids and computes aggregate pattern metrics.

    Returns a dict with aggregate statistics about patterns.
    """
    num_to_load = min(len(metadata), max_samples)
    changed_fracs = []
    sym_h = []
    sym_v = []
    sym_180 = []
    sym_90 = []

    for i in range(num_to_load):
        try:
            grid = load_grid(grids_dir, i)
        except FileNotFoundError:
            continue

        pat = detect_patterns(grid)
        changed_fracs.append(pat["changed_fraction"])

        sym = compute_symmetry(grid)
        sym_h.append(sym["horizontal_reflection"])
        sym_v.append(sym["vertical_reflection"])
        sym_180.append(sym["rotation_180"])
        sym_90.append(sym["rotation_90"])

    return {
        "samples_analyzed": len(changed_fracs),
        "changed_fraction": {
            "mean": float(np.mean(changed_fracs)),
            "std": float(np.std(changed_fracs)),
        },
        "symmetry_correlations": {
            "horizontal_reflection": float(np.mean(sym_h)),
            "vertical_reflection": float(np.mean(sym_v)),
            "rotation_180": float(np.mean(sym_180)),
            "rotation_90": float(np.mean(sym_90)),
        },
    }


# ---------------------------------------------------------------------------
# 3D analysis helpers
# ---------------------------------------------------------------------------

def analyze_3d_grid(grid_3d: np.ndarray, background: int = 5) -> dict:
    """Basic pattern detection for a 3D sandpile cube.

    Parameters
    ----------
    grid_3d : np.ndarray, shape (n, n, n)
    background : int
        Background grain count (5 for 3D experiments).

    Returns
    -------
    dict with slice-wise and volumetric statistics.
    """
    n = grid_3d.shape[0]
    mask = (grid_3d != background)
    total_sites = n ** 3
    changed_fraction = mask.sum() / total_sites

    grain_counts = Counter(int(v) for v in grid_3d.flatten())

    # Per-slice changed fraction
    slice_fracs = []
    for z in range(n):
        sl = grid_3d[:, :, z]
        slice_fracs.append(float((sl != background).mean()))

    return {
        "changed_fraction": float(changed_fraction),
        "grain_counts": dict(sorted(grain_counts.items())),
        "slice_changed_fractions": slice_fracs,
    }


def analyze_3d_patterns_across_trials(grids_dir: str, metadata: list[dict],
                                      background: int = 5,
                                      max_samples: int = 100) -> dict:
    """Analyze 3D patterns across multiple trials.

    Loads up to *max_samples* grids and computes aggregate metrics
    including per-slice changed fraction averages.

    Returns
    -------
    dict with:
        samples_analyzed : int
        changed_fraction : dict with mean, std
        avg_slice_changed_fractions : list[float]
            Average changed fraction per z-slice across all sampled trials.
    """
    num_to_load = min(len(metadata), max_samples)
    changed_fracs = []
    all_slice_fracs = []  # list of lists

    for i in range(num_to_load):
        try:
            grid = load_grid(grids_dir, i)
        except FileNotFoundError:
            continue

        result = analyze_3d_grid(grid, background=background)
        changed_fracs.append(result["changed_fraction"])
        all_slice_fracs.append(result["slice_changed_fractions"])

    if not changed_fracs:
        return {"samples_analyzed": 0}

    # Average per-slice changed fractions
    n_slices = len(all_slice_fracs[0])
    avg_slice_fracs = [
        float(np.mean([sf[z] for sf in all_slice_fracs]))
        for z in range(n_slices)
    ]

    return {
        "samples_analyzed": len(changed_fracs),
        "changed_fraction": {
            "mean": float(np.mean(changed_fracs)),
            "std": float(np.std(changed_fracs)),
        },
        "avg_slice_changed_fractions": avg_slice_fracs,
    }

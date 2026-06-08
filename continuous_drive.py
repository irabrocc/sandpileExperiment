"""
Continuous-drive sandpile experiments for studying self-organised
criticality (SOC) and power-law avalanche-size distributions.

Instead of resetting the grid after each perturbation, we:
  1. Start from an empty (or filled) grid.
  2. Repeatedly add 1 grain to a random site.
  3. Stabilise and record the avalanche size (total topples).
  4. Continue without resetting — the grid evolves continuously.

Over many iterations the system self-organises into a critical state
where avalanche sizes follow a power-law distribution.
"""

import json
import os
import time
import numpy as np
from tqdm import tqdm

import config
from sandpile_2d import Sandpile2D
from sandpile_3d import Sandpile3D


# ---------------------------------------------------------------------------
# 2D continuous drive
# ---------------------------------------------------------------------------

def continuous_drive_2d(
    N: int,
    num_additions: int,
    threshold: int = 4,
    initial_grains: int = 0,
    seed: int | None = None,
    skip_transient: int = 0,
) -> dict:
    """Continuously drive a 2D sandpile and record avalanche statistics.

    Parameters
    ----------
    N : int
        Grid side length.
    num_additions : int
        Number of grain-addition + stabilisation cycles.
    threshold : int
        Toppling threshold (4 standard for 2D).
    initial_grains : int
        Starting grains on every site (0 = empty grid).
    seed : int | None
        Random seed.
    skip_transient : int
        Number of initial additions to discard (transient phase before
        the system reaches the critical state).

    Returns
    -------
    dict with keys:
        avalanche_sizes    – list[int], total topples per addition
        distinct_sites     – list[int], distinct sites that toppled
        final_grid         – np.ndarray (N×N), the final configuration
        N, num_additions, skip_transient
    """
    rng = np.random.default_rng(seed)
    sp = Sandpile2D(N, threshold=threshold)
    sp.fill(initial_grains)

    sizes = []
    distincts = []

    total = skip_transient + num_additions
    for i in tqdm(range(total), desc=f"2D continuous N={N}"):
        x, y = rng.integers(0, N, size=2)
        sp.add_grain(x, y)
        stats = sp.stabilize()

        if i >= skip_transient:
            sizes.append(stats["total_topples"])
            distincts.append(stats["distinct_sites"])

    return {
        "avalanche_sizes": sizes,
        "distinct_sites": distincts,
        "final_grid": sp.copy_grid(),
        "N": N,
        "num_additions": num_additions,
        "skip_transient": skip_transient,
        "threshold": threshold,
        "initial_grains": initial_grains,
    }


# ---------------------------------------------------------------------------
# 3D continuous drive
# ---------------------------------------------------------------------------

def continuous_drive_3d(
    n: int,
    num_additions: int,
    threshold: int = 6,
    initial_grains: int = 0,
    seed: int | None = None,
    skip_transient: int = 0,
) -> dict:
    """Continuously drive a 3D sandpile and record avalanche statistics.

    Parameters
    ----------
    n : int
        Cube side length.
    num_additions : int
        Number of grain-addition + stabilisation cycles.
    threshold : int
        Toppling threshold (6 standard for 3D).
    initial_grains : int
        Starting grains on every site.
    seed : int | None
    skip_transient : int
        Transient additions to discard.

    Returns
    -------
    dict with avalanche_sizes, distinct_sites, final_grid, etc.
    """
    rng = np.random.default_rng(seed)
    sp = Sandpile3D(n, threshold=threshold)
    sp.fill(initial_grains)

    sizes = []
    distincts = []

    total = skip_transient + num_additions
    for i in tqdm(range(total), desc=f"3D continuous n={n}"):
        x, y, z = rng.integers(0, n, size=3)
        sp.add_grain(x, y, z)
        stats = sp.stabilize()

        if i >= skip_transient:
            sizes.append(stats["total_topples"])
            distincts.append(stats["distinct_sites"])

    return {
        "avalanche_sizes": sizes,
        "distinct_sites": distincts,
        "final_grid": sp.copy_grid(),
        "n": n,
        "num_additions": num_additions,
        "skip_transient": skip_transient,
        "threshold": threshold,
        "initial_grains": initial_grains,
    }


# ---------------------------------------------------------------------------
# Power-law analysis helpers
# ---------------------------------------------------------------------------

def estimate_tau_mle(sizes: list[int], s_min: int | None = None) -> dict:
    """Estimate the power-law exponent τ via MLE (maximum likelihood).

    For a discrete power law P(s) ∝ s^(-τ) with s ≥ s_min:
        τ̂ = 1 + n / Σ_i ln(s_i / (s_min - 0.5))

    Parameters
    ----------
    sizes : list[int]
        Avalanche sizes (> 0).
    s_min : int | None
        Lower cutoff. If None, uses min(sizes).

    Returns
    -------
    dict with tau, s_min, n_samples, standard_error
    """
    s = np.array([x for x in sizes if x > 0], dtype=np.float64)
    if len(s) == 0:
        return {"tau": None, "s_min": 0, "n_samples": 0, "stderr": None}

    if s_min is None:
        s_min = int(s.min())
    s = s[s >= s_min]
    n = len(s)

    if n < 2:
        return {"tau": None, "s_min": s_min, "n_samples": n, "stderr": None}

    # MLE for discrete power law (Clauset et al. 2009, eq 3.7)
    denom = np.sum(np.log(s / (s_min - 0.5)))
    if denom == 0:
        return {"tau": None, "s_min": s_min, "n_samples": n, "stderr": None}
    tau = 1.0 + n / denom

    # Standard error (eq 3.8)
    stderr = (tau - 1.0) / np.sqrt(n)

    return {"tau": float(tau), "s_min": s_min, "n_samples": n,
            "stderr": float(stderr)}


def complementary_cdf(sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the complementary CDF: P(S > s).

    Returns (s_values, ccdf) where ccdf[i] = fraction of avalanches
    with size > s_values[i].
    """
    s = np.sort([x for x in sizes if x > 0])
    if len(s) == 0:
        return np.array([]), np.array([])
    n = len(s)
    unique, indices = np.unique(s, return_index=True)
    ccdf = 1.0 - indices / n
    # Add the point before the first value
    return unique, ccdf


def log_binned_histogram(sizes: list[int], bins_per_decade: float = 2.0
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a logarithmically-binned histogram suitable for power-law
    visualisation on log-log axes.

    Parameters
    ----------
    sizes : list[int]
    bins_per_decade : float
        Number of bins per factor of 10.

    Returns
    -------
    bin_centers, density, bin_edges – arrays for plotting
    """
    s = np.array([x for x in sizes if x > 0], dtype=np.float64)
    if len(s) == 0:
        return np.array([]), np.array([]), np.array([])

    s_min, s_max = s.min(), s.max()
    if s_min == s_max:
        return np.array([s_min]), np.array([1.0]), np.array([s_min, s_max + 1])

    num_decades = np.log10(s_max / s_min)
    num_bins = max(2, int(np.ceil(num_decades * bins_per_decade)))

    # Log-spaced bin edges
    bin_edges = np.logspace(np.log10(s_min), np.log10(s_max), num_bins + 1)
    bin_edges = np.unique(np.round(bin_edges))  # integer edges
    bin_edges = bin_edges.astype(np.float64)

    counts, _ = np.histogram(s, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Density = count / (total * bin_width)
    density = counts / (len(s) * bin_widths)
    # Avoid zeros for log scale
    density = np.maximum(density, 1e-300)

    return bin_centers, density, bin_edges


# ---------------------------------------------------------------------------
# Batch runner wrapper
# ---------------------------------------------------------------------------

def run_continuous_2d_batch(
    N: int,
    num_additions: int,
    skip_transient: int = 0,
    initial_grains: int = 0,
    threshold: int = 4,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict:
    """Run a 2D continuous-drive experiment and save results."""
    if output_dir is None:
        output_dir = config.OUTPUT_2D_DIR

    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, "stats")
    grids_dir = os.path.join(output_dir, "grids")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    start = time.time()
    result = continuous_drive_2d(
        N=N,
        num_additions=num_additions,
        threshold=threshold,
        initial_grains=initial_grains,
        seed=seed,
        skip_transient=skip_transient,
    )
    elapsed = time.time() - start

    sizes = result["avalanche_sizes"]

    # Save final grid
    grid_path = os.path.join(grids_dir, "continuous_final.npy")
    np.save(grid_path, result["final_grid"])

    # Save avalanche sizes as .npy and .json summary
    sizes_path = os.path.join(stats_dir, "continuous_sizes.npy")
    np.save(sizes_path, np.array(sizes, dtype=np.int64))

    # Fit power law
    tau_info = estimate_tau_mle(sizes, s_min=1)

    summary = {
        "mode": "continuous_2d",
        "N": N,
        "num_additions": num_additions,
        "skip_transient": skip_transient,
        "initial_grains": initial_grains,
        "threshold": threshold,
        "elapsed_sec": round(elapsed, 1),
        "total_grains_added": num_additions,
        "total_topples": int(np.sum(sizes)),
        "mean_avalanche_size": float(np.mean(sizes)) if sizes else 0,
        "median_avalanche_size": float(np.median(sizes)) if sizes else 0,
        "max_avalanche_size": int(np.max(sizes)) if sizes else 0,
        "zero_avalanche_fraction": float(np.mean(np.array(sizes) == 0)),
        "power_law_fit": tau_info,
        "n_samples": len(sizes),
    }
    summary_path = os.path.join(stats_dir, "continuous_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n2D continuous drive: {num_additions} additions in {elapsed:.1f}s")
    print(f"  Avalanche sizes — mean: {summary['mean_avalanche_size']:.1f}, "
          f"max: {summary['max_avalanche_size']}")
    if tau_info["tau"] is not None:
        print(f"  Estimated τ = {tau_info['tau']:.3f} ± {tau_info['stderr']:.3f}")
    print(f"  Zero-avalanche fraction: {summary['zero_avalanche_fraction']:.3f}")

    return result


def run_continuous_3d_batch(
    n: int,
    num_additions: int,
    skip_transient: int = 0,
    initial_grains: int = 0,
    threshold: int = 6,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict:
    """Run a 3D continuous-drive experiment and save results."""
    if output_dir is None:
        output_dir = config.OUTPUT_3D_DIR

    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, "stats")
    grids_dir = os.path.join(output_dir, "grids")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    start = time.time()
    result = continuous_drive_3d(
        n=n,
        num_additions=num_additions,
        threshold=threshold,
        initial_grains=initial_grains,
        seed=seed,
        skip_transient=skip_transient,
    )
    elapsed = time.time() - start

    sizes = result["avalanche_sizes"]

    np.save(os.path.join(grids_dir, "continuous_final.npy"),
            result["final_grid"])
    np.save(os.path.join(stats_dir, "continuous_sizes.npy"),
            np.array(sizes, dtype=np.int64))

    tau_info = estimate_tau_mle(sizes, s_min=1)

    summary = {
        "mode": "continuous_3d",
        "n": n,
        "num_additions": num_additions,
        "skip_transient": skip_transient,
        "initial_grains": initial_grains,
        "threshold": threshold,
        "elapsed_sec": round(elapsed, 1),
        "total_topples": int(np.sum(sizes)),
        "mean_avalanche_size": float(np.mean(sizes)) if sizes else 0,
        "median_avalanche_size": float(np.median(sizes)) if sizes else 0,
        "max_avalanche_size": int(np.max(sizes)) if sizes else 0,
        "power_law_fit": tau_info,
        "n_samples": len(sizes),
    }
    with open(os.path.join(stats_dir, "continuous_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n3D continuous drive: {num_additions} additions in {elapsed:.1f}s")
    print(f"  Avalanche sizes — mean: {summary['mean_avalanche_size']:.1f}, "
          f"max: {summary['max_avalanche_size']}")
    if tau_info["tau"] is not None:
        print(f"  Estimated τ = {tau_info['tau']:.3f} ± {tau_info['stderr']:.3f}")

    return result

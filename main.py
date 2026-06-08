#!/usr/bin/env python3
"""
Sandpile Statistical Experiments — Command-Line Entry Point.

Two experiment paradigms:
  A) Perturbation mode: reset grid each trial, perturb 3 sites, stabilise.
  B) Continuous-drive (SOC) mode: never reset, add grains continuously,
     record avalanche sizes to study power-law distributions.

Usage:
    # Perturbation experiments
    python main.py --mode 2d [--N 200] [--trials 1000] [--seed 42]
    python main.py --mode 3d [--n 30]  [--trials 1000] [--seed 42]
    python main.py --all
    python main.py --analyze 2d
    python main.py --analyze 3d

    # Interactive 3D viewer
    python main.py --view3d
    python main.py --view3d --trial 42

    # Interactive 2D viewer
    python main.py --view2d
    python main.py --view2d --trial 100

    # Continuous-drive SOC experiments (power-law analysis)
    python main.py --continuous 2d --N 100 --additions 100000
    python main.py --continuous 3d --n 20  --additions 50000
    python main.py --analyze-continuous 2d
    python main.py --analyze-continuous 3d

    # Interactive 3D viewer
    python main.py --view3d
    python main.py --view3d --trial 42 --z 50
"""

import argparse
import json
import os
import sys

import config
from config import ensure_dirs


def run_2d(args):
    """Run 2D sandpile experiment."""
    from experiment_2d import run_batch
    ensure_dirs()
    N = args.N or config.N_2D
    trials = args.trials or config.NUM_TRIALS_2D
    seed = args.seed or config.BASE_SEED

    print(f"=== 2D Sandpile Experiment ===")
    print(f"Grid: {N}x{N}, Trials: {trials}, Seed: {seed}")
    print(f"Initial grains: {config.INITIAL_GRAINS_2D}, "
          f"Threshold: {config.TOPPLE_THRESHOLD_2D}")
    print(f"Output: {config.OUTPUT_2D_DIR}")
    print()

    run_batch(
        N=N,
        num_trials=trials,
        output_dir=config.OUTPUT_2D_DIR,
        threshold=config.TOPPLE_THRESHOLD_2D,
        initial_grains=config.INITIAL_GRAINS_2D,
        base_seed=seed,
        save_interval=config.SAVE_INTERVAL,
    )

    print(f"\n2D experiment complete. Results in {config.OUTPUT_2D_DIR}")


def run_3d(args):
    """Run 3D sandpile experiment."""
    from experiment_3d import run_batch
    ensure_dirs()
    n = args.n or config.N_3D
    trials = args.trials or config.NUM_TRIALS_3D
    seed = args.seed or config.BASE_SEED

    print(f"=== 3D Sandpile Experiment ===")
    print(f"Cube: {n}x{n}x{n}, Trials: {trials}, Seed: {seed}")
    print(f"Initial grains: {config.INITIAL_GRAINS_3D}, "
          f"Threshold: {config.TOPPLE_THRESHOLD_3D}")
    print(f"Output: {config.OUTPUT_3D_DIR}")
    print()

    run_batch(
        n=n,
        num_trials=trials,
        output_dir=config.OUTPUT_3D_DIR,
        threshold=config.TOPPLE_THRESHOLD_3D,
        initial_grains=config.INITIAL_GRAINS_3D,
        base_seed=seed,
        save_interval=config.SAVE_INTERVAL,
    )

    print(f"\n3D experiment complete. Results in {config.OUTPUT_3D_DIR}")


def analyze_2d(args):
    """Analyze 2D results and produce visualizations."""
    import numpy as np
    from analyze import (
        load_metadata, load_grid, compute_statistics,
        analyze_patterns_across_trials, detect_patterns, compute_symmetry,
    )
    from visualize import (
        plot_grid, plot_grid_comparison, plot_avalanche_size_histogram,
        plot_toppling_frequency_map,
    )

    metadata = load_metadata(config.STATS_2D_DIR)
    stats = compute_statistics(metadata)
    print(json.dumps(stats["topple_stats"], indent=2))

    # Plot avalanche size histogram
    plot_avalanche_size_histogram(
        stats["avalanche_sizes"],
        save_path=os.path.join(config.IMAGES_2D_DIR, "avalanche_histogram.png"),
    )

    # Plot toppling frequency map
    plot_toppling_frequency_map(
        config.GRIDS_2D_DIR, len(metadata),
        save_path=os.path.join(config.IMAGES_2D_DIR, "toppling_frequency.png"),
        max_to_load=500,
        background=config.INITIAL_GRAINS_2D,
    )

    # Sample grid visualizations
    sample_indices = [0, 1, 2, 5, 10, 20, 50, 100, 200]
    sample_indices = [i for i in sample_indices if i < len(metadata)]
    grids = []
    titles = []
    for idx in sample_indices:
        try:
            g = load_grid(config.GRIDS_2D_DIR, idx)
            grids.append(g)
            titles.append(f"Trial {idx} (topples={metadata[idx]['topples']})")
        except FileNotFoundError:
            pass

    if grids:
        plot_grid_comparison(
            grids, titles,
            save_path=os.path.join(config.IMAGES_2D_DIR, "sample_grids.png"),
        )

    # Pattern analysis
    pat = analyze_patterns_across_trials(config.GRIDS_2D_DIR, metadata)
    print("\nPattern analysis:")
    print(json.dumps(pat, indent=2))

    # Save stats as JSON
    stats_path = os.path.join(config.STATS_2D_DIR, "analysis.json")
    with open(stats_path, "w") as f:
        json.dump({**stats, "patterns": pat}, f, indent=2)

    print(f"\n2D analysis complete. Visualizations in {config.IMAGES_2D_DIR}")


def analyze_3d(args):
    """Analyze 3D results and produce visualizations."""
    import numpy as np
    from analyze import (
        load_metadata, load_grid, compute_statistics,
        analyze_3d_grid, analyze_3d_patterns_across_trials,
    )
    from visualize import (
        plot_avalanche_size_histogram, plot_3d_slices,
        plot_3d_slice_montage, plot_3d_toppling_frequency_map,
        plot_slice_changed_fraction_profile,
        plot_3d_toppling_frequency_slices,
    )

    metadata = load_metadata(config.STATS_3D_DIR)
    stats = compute_statistics(metadata)
    print(json.dumps(stats["topple_stats"], indent=2))

    # ---- Avalanche size histogram (same as 2D) ----
    plot_avalanche_size_histogram(
        stats["avalanche_sizes"],
        save_path=os.path.join(config.IMAGES_3D_DIR, "avalanche_histogram.png"),
    )

    # ---- Toppling frequency map (middle z-slice) ----
    plot_3d_toppling_frequency_map(
        config.GRIDS_3D_DIR, len(metadata),
        save_path=os.path.join(config.IMAGES_3D_DIR, "toppling_frequency.png"),
        max_to_load=200,
        background=config.INITIAL_GRAINS_3D,
    )

    # ---- Toppling frequency at multiple z-slices ----
    plot_3d_toppling_frequency_slices(
        config.GRIDS_3D_DIR, len(metadata),
        save_path=os.path.join(config.IMAGES_3D_DIR, "toppling_frequency_slices.png"),
        max_to_load=100,
        background=config.INITIAL_GRAINS_3D,
    )

    # ---- Sample 3D slice visualizations ----
    sample_indices = [0, 1, 2, 5, 10]
    sample_indices = [i for i in sample_indices if i < len(metadata)]

    for idx in sample_indices:
        try:
            g = load_grid(config.GRIDS_3D_DIR, idx)
        except FileNotFoundError:
            continue
        plot_3d_slices(
            g,
            title=f"3D Trial {idx} (topples={metadata[idx]['topples']})",
            save_path=os.path.join(config.IMAGES_3D_DIR, f"slices_trial_{idx:06d}.png"),
        )

    # ---- Slice montage comparison across trials ----
    grids_3d = []
    labels_3d = []
    for idx in sample_indices:
        try:
            grids_3d.append(load_grid(config.GRIDS_3D_DIR, idx))
            labels_3d.append(f"Trial {idx}")
        except FileNotFoundError:
            pass
    if len(grids_3d) >= 2:
        plot_3d_slice_montage(
            grids_3d, labels_3d,
            save_path=os.path.join(config.IMAGES_3D_DIR, "slice_montage.png"),
        )

    # ---- 3D pattern analysis across trials ----
    pat = analyze_3d_patterns_across_trials(
        config.GRIDS_3D_DIR, metadata,
        background=config.INITIAL_GRAINS_3D,
        max_samples=200,
    )
    print("\n3D Pattern analysis:")
    print(json.dumps({k: v for k, v in pat.items()
                      if k != "avg_slice_changed_fractions"}, indent=2))

    # ---- Slice changed fraction profile ----
    if pat.get("avg_slice_changed_fractions"):
        plot_slice_changed_fraction_profile(
            pat["avg_slice_changed_fractions"],
            save_path=os.path.join(config.IMAGES_3D_DIR,
                                   "slice_changed_fraction_profile.png"),
            title=f"Changed Fraction per z-slice "
                  f"(avg over {pat['samples_analyzed']} trials)",
        )

    # Save stats as JSON
    stats_path = os.path.join(config.STATS_3D_DIR, "analysis.json")
    with open(stats_path, "w") as f:
        json.dump({**stats, "patterns": pat}, f, indent=2)

    print(f"\n3D analysis complete. Visualizations in {config.IMAGES_3D_DIR}")


# ---------------------------------------------------------------------------
# Continuous-drive experiments
# ---------------------------------------------------------------------------

def run_continuous_2d(args):
    """Run 2D continuous-drive SOC experiment."""
    from continuous_drive import run_continuous_2d_batch
    ensure_dirs()
    N = args.N or config.N_2D
    additions = args.additions or 100_000
    skip = args.skip_transient or 10_000
    initial = args.initial_grains if args.initial_grains is not None else 0
    seed = args.seed or config.BASE_SEED

    print(f"=== 2D Continuous-Drive SOC Experiment ===")
    print(f"Grid: {N}x{N}, Additions: {additions:,}, Skip transient: {skip:,}")
    print(f"Initial grains: {initial}, Threshold: {config.TOPPLE_THRESHOLD_2D}")
    print()

    run_continuous_2d_batch(
        N=N,
        num_additions=additions,
        skip_transient=skip,
        initial_grains=initial,
        threshold=config.TOPPLE_THRESHOLD_2D,
        seed=seed,
    )

    print(f"\n2D continuous drive complete. Results in {config.OUTPUT_2D_DIR}")


def run_continuous_3d(args):
    """Run 3D continuous-drive SOC experiment."""
    from continuous_drive import run_continuous_3d_batch
    ensure_dirs()
    n = args.n or config.N_3D
    additions = args.additions or 50_000
    skip = args.skip_transient or 5_000
    initial = args.initial_grains if args.initial_grains is not None else 0
    seed = args.seed or config.BASE_SEED

    print(f"=== 3D Continuous-Drive SOC Experiment ===")
    print(f"Cube: {n}x{n}x{n}, Additions: {additions:,}, Skip transient: {skip:,}")
    print(f"Initial grains: {initial}, Threshold: {config.TOPPLE_THRESHOLD_3D}")
    print()

    run_continuous_3d_batch(
        n=n,
        num_additions=additions,
        skip_transient=skip,
        initial_grains=initial,
        threshold=config.TOPPLE_THRESHOLD_3D,
        seed=seed,
    )

    print(f"\n3D continuous drive complete. Results in {config.OUTPUT_3D_DIR}")


def analyze_continuous_2d(args):
    """Analyze 2D continuous-drive results — power-law plots."""
    import numpy as np
    from continuous_drive import estimate_tau_mle
    from visualize import plot_power_law_distribution, plot_grid

    sizes_path = os.path.join(config.STATS_2D_DIR, "continuous_sizes.npy")
    if not os.path.exists(sizes_path):
        print(f"Error: no continuous-drive data found at {sizes_path}")
        print("Run --continuous 2d first.")
        return

    sizes = np.load(sizes_path).tolist()
    tau_info = estimate_tau_mle(sizes, s_min=1)
    print(f"MLE power-law fit (s_min=1): τ = {tau_info['tau']:.3f} "
          f"± {tau_info['stderr']:.3f}  (n={tau_info['n_samples']})")

    # Also try with higher s_min to see if fit improves
    for s_min_try in [2, 3, 5, 10]:
        ti = estimate_tau_mle(sizes, s_min=s_min_try)
        if ti["tau"] is not None:
            print(f"  s_min={s_min_try:3d}: τ = {ti['tau']:.3f} ± {ti['stderr']:.3f}")

    plot_power_law_distribution(
        sizes,
        tau=tau_info["tau"],
        save_path=os.path.join(config.IMAGES_2D_DIR, "power_law_distribution.png"),
        title=f"2D Avalanche Size Distribution (N={config.N_2D})",
    )

    # Also plot the final grid from continuous driving
    grid_path = os.path.join(config.GRIDS_2D_DIR, "continuous_final.npy")
    if os.path.exists(grid_path):
        plot_grid(
            np.load(grid_path),
            title="Final Grid — 2D Continuous Drive",
            save_path=os.path.join(config.IMAGES_2D_DIR, "continuous_final_grid.png"),
        )

    print(f"\n2D continuous analysis complete. Plots in {config.IMAGES_2D_DIR}")


def analyze_continuous_3d(args):
    """Analyze 3D continuous-drive results — power-law plots."""
    import numpy as np
    from continuous_drive import estimate_tau_mle
    from visualize import plot_power_law_distribution, plot_3d_slices

    sizes_path = os.path.join(config.STATS_3D_DIR, "continuous_sizes.npy")
    if not os.path.exists(sizes_path):
        print(f"Error: no continuous-drive data found at {sizes_path}")
        print("Run --continuous 3d first.")
        return

    sizes = np.load(sizes_path).tolist()
    tau_info = estimate_tau_mle(sizes, s_min=1)
    print(f"MLE power-law fit (s_min=1): τ = {tau_info['tau']:.3f} "
          f"± {tau_info['stderr']:.3f}  (n={tau_info['n_samples']})")

    for s_min_try in [2, 3, 5, 10]:
        ti = estimate_tau_mle(sizes, s_min=s_min_try)
        if ti["tau"] is not None:
            print(f"  s_min={s_min_try:3d}: τ = {ti['tau']:.3f} ± {ti['stderr']:.3f}")

    plot_power_law_distribution(
        sizes,
        tau=tau_info["tau"],
        save_path=os.path.join(config.IMAGES_3D_DIR, "power_law_distribution.png"),
        title=f"3D Avalanche Size Distribution (n={config.N_3D})",
    )

    grid_path = os.path.join(config.GRIDS_3D_DIR, "continuous_final.npy")
    if os.path.exists(grid_path):
        plot_3d_slices(
            np.load(grid_path),
            title="Final Grid — 3D Continuous Drive",
            save_path=os.path.join(config.IMAGES_3D_DIR, "continuous_final_slices.png"),
        )

    print(f"\n3D continuous analysis complete. Plots in {config.IMAGES_3D_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def interactive_view_2d(args):
    """Launch the interactive 2D sandpile grid explorer."""
    from visualize import interactive_2d_viewer

    ensure_dirs()
    print("=== Interactive 2D Sandpile Viewer ===")
    print(f"Loading grids from: {config.GRIDS_2D_DIR}")
    print("Controls:")
    print("  Trial slider — switch between trials")
    print("  Keyboard: ← → to navigate trials")
    print("  Close the window to exit.")
    print()

    interactive_2d_viewer(
        grids_dir=config.GRIDS_2D_DIR,
        stats_dir=config.STATS_2D_DIR,
        initial_trial=args.trial or 0,
    )


def interactive_view_3d(args):
    """Launch the interactive 3D sandpile slice explorer."""
    from visualize import interactive_3d_viewer

    ensure_dirs()
    print("=== Interactive 3D Sandpile Viewer ===")
    print(f"Loading grids from: {config.GRIDS_3D_DIR}")
    print("Controls:")
    print("  Trial slider — switch between trials")
    print("  z-slice slider — scroll through depth")
    print("  Keyboard: ← → (z-slice), ↑ ↓ (trial)")
    print("  Close the window to exit.")
    print()

    interactive_3d_viewer(
        grids_dir=config.GRIDS_3D_DIR,
        stats_dir=config.STATS_3D_DIR,
        initial_trial=args.trial or 0,
        initial_z=args.z,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sandpile Statistical Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode 2d --N 200 --trials 1000
  python main.py --mode 3d --n 30 --trials 500
  python main.py --analyze 2d
  python main.py --continuous 2d --N 100 --additions 100000
  python main.py --continuous 3d --n 20 --additions 50000 --skip-transient 10000
  python main.py --analyze-continuous 2d
        """,
    )
    # ---- Experiment modes ----
    parser.add_argument("--mode", choices=["2d", "3d"],
                        help="Run perturbation experiment (2d or 3d)")
    parser.add_argument("--all", action="store_true",
                        help="Run both 2D and 3D perturbation experiments")
    parser.add_argument("--analyze", choices=["2d", "3d"],
                        help="Analyze perturbation experiment results")

    # ---- Continuous-drive modes ----
    parser.add_argument("--continuous", choices=["2d", "3d"],
                        help="Run continuous-drive SOC experiment")
    parser.add_argument("--analyze-continuous", choices=["2d", "3d"],
                        help="Analyze continuous-drive results (power-law plots)")

    # ---- Interactive viewer ----
    parser.add_argument("--view2d", action="store_true",
                        help="Launch interactive 2D sandpile grid explorer")
    parser.add_argument("--view3d", action="store_true",
                        help="Launch interactive 3D sandpile slice explorer")
    parser.add_argument("--trial", type=int, default=0,
                        help="Initial trial index for interactive viewer")
    parser.add_argument("--z", type=int, default=None,
                        help="Initial z-slice for interactive viewer")

    # ---- Common parameters ----
    parser.add_argument("--N", type=int, help="2D grid size")
    parser.add_argument("--n", type=int, help="3D cube size")
    parser.add_argument("--trials", type=int, help="Number of trials (perturbation mode)")
    parser.add_argument("--seed", type=int, help="Base random seed")

    # ---- Continuous-drive parameters ----
    parser.add_argument("--additions", type=int,
                        help="Number of grain additions (continuous mode, default 100k)")
    parser.add_argument("--skip-transient", type=int,
                        help="Transient additions to discard (default 10k)")
    parser.add_argument("--initial-grains", type=int,
                        help="Initial grains on every site (default 0 = empty)")

    args = parser.parse_args()

    # Route to the correct handler
    if args.view2d:
        interactive_view_2d(args)
    elif args.view3d:
        interactive_view_3d(args)
    elif args.analyze_continuous:
        if args.analyze_continuous == "2d":
            analyze_continuous_2d(args)
        else:
            analyze_continuous_3d(args)
    elif args.continuous:
        if args.continuous == "2d":
            run_continuous_2d(args)
        else:
            run_continuous_3d(args)
    elif args.analyze:
        if args.analyze == "2d":
            analyze_2d(args)
        else:
            analyze_3d(args)
    elif args.all:
        run_2d(args)
        run_3d(args)
    elif args.mode == "2d":
        run_2d(args)
    elif args.mode == "3d":
        run_3d(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

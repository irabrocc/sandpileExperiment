"""
Visualization utilities for sandpile experiments.

Provides interactive viewers for browsing 2D and 3D sandpile grid
snapshots, including a local 18×18 patch inspection feature.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Colour map
# ---------------------------------------------------------------------------

# Discrete colour map for sandpile grain counts
# 2D threshold = 4, 3D threshold = 6
# High-contrast hue progression: each grain count gets a distinct, easily
# told-apart colour.  Values 1 (yellow) and 2 (blue) are deliberately
# complementary so they are never confused.
_SAND_COLORS = [
    "#f7f7f7",  # 0 – off-white (empty)
    "#ffd966",  # 1 – warm yellow
    "#6fa8dc",  # 2 – sky blue
    "#93c47d",  # 3 – light green (2D background)
    "#e69138",  # 4 – warm orange
    "#8e7133",  # 5 – brown (3D background)
    "#cc0000",  # 6+ – red (critical / avalanche)
]
SAND_CMAP = mcolors.ListedColormap(_SAND_COLORS)

# BoundaryNorm so that every integer grain count maps to exactly one colour,
# regardless of the data range or vmax used by imshow.  Without this,
# ListedColormap stretches its colours across [vmin, vmax], so the same
# grain count can render as different colours when vmax differs across views.
_SAND_BOUNDS = [i - 0.5 for i in range(len(_SAND_COLORS) + 1)]
SAND_NORM = mcolors.BoundaryNorm(_SAND_BOUNDS, ncolors=len(_SAND_COLORS))





# ---------------------------------------------------------------------------
# Interactive viewers (2D and 3D)
# ---------------------------------------------------------------------------

def interactive_2d_viewer(grids_dir: str, stats_dir: str | None = None,
                          initial_trial: int = 0):
    """Interactive 2D sandpile grid explorer.

    Opens a matplotlib window with a trial slider to browse grid snapshots.
    Perturbation sites are marked with red '+' crosses and a legend.

    Parameters
    ----------
    grids_dir : str
        Directory containing trial_NNNNNN.npy files.
    stats_dir : str | None
        Directory with all_metadata.json for displaying trial info.
    initial_trial : int
        Starting trial index.
    """
    import json
    from matplotlib.widgets import Slider

    # --- Discover available trials ---
    trial_indices = []
    for fname in sorted(os.listdir(grids_dir)):
        if fname.startswith("trial_") and fname.endswith(".npy"):
            try:
                idx = int(fname.replace("trial_", "").replace(".npy", ""))
                trial_indices.append(idx)
            except ValueError:
                pass
    trial_indices.sort()

    if not trial_indices:
        print(f"No trial grid files found in {grids_dir}")
        return

    # --- Load metadata if available ---
    metadata_map = {}
    if stats_dir is not None:
        meta_path = os.path.join(stats_dir, "all_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata_list = json.load(f)
            metadata_map = {i: m for i, m in enumerate(metadata_list)}

    # --- Load first grid ---
    path0 = os.path.join(grids_dir, f"trial_{trial_indices[0]:06d}.npy")
    sample_grid = np.load(path0)

    # Clamp initial value
    trial_idx_pos = trial_indices.index(initial_trial) if initial_trial in trial_indices else 0
    current_trial_idx_pos = trial_idx_pos

    # --- Build the figure ---
    fig, ax_img = plt.subplots(figsize=(8, 9.5))
    plt.subplots_adjust(bottom=0.18, left=0.10, right=0.88)

    # Initial display
    trial_idx = trial_indices[current_trial_idx_pos]
    path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
    grid = np.load(path)

    im = ax_img.imshow(grid, cmap=SAND_CMAP, norm=SAND_NORM,
                       interpolation="nearest", origin="upper")

    # --- Perturbation site markers ---
    meta = metadata_map.get(trial_idx, {})
    perturb_sites = meta.get("perturb_sites", [])
    perturb_xs = [p[0] for p in perturb_sites]
    perturb_ys = [p[1] for p in perturb_sites]
    (perturb_markers,) = ax_img.plot(
        perturb_xs, perturb_ys, "P",
        color="red", markersize=14, markeredgewidth=1.5,
        markeredgecolor="darkred", label="Perturb sites"
    )

    topples = meta.get("topples", "?")
    distinct = meta.get("distinct", "?")
    title_text = (f"2D Sandpile — Trial {trial_idx}  "
                  f"(topples={topples:,}, distinct={distinct:,})")
    ax_title = ax_img.set_title(title_text, fontsize=13)
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")

    # --- Colorbar (placed explicitly to the right, outside the axes) ---
    cbar_ax = fig.add_axes([0.90, 0.30, 0.03, 0.50])
    cbar = fig.colorbar(im, cax=cbar_ax,
                        ticks=range(len(_SAND_COLORS)))
    cbar.set_label("Grains", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # --- Perturb-sites legend (below the colorbar) ---
    legend_ax = fig.add_axes([0.86, 0.20, 0.12, 0.06])
    legend_ax.axis("off")
    legend_ax.legend(
        [perturb_markers], ["Perturb sites"],
        loc="center", fontsize=8, framealpha=0.85, facecolor="white",
        ncol=1, markerscale=0.6, handletextpad=0.3,
    )

    # --- Local 18×18 patch preview (inset axes, top-left of grid) ---
    local_patches_dir = os.path.join(os.path.dirname(grids_dir), "local_patches")
    os.makedirs(local_patches_dir, exist_ok=True)

    ax_inset = fig.add_axes([0.12, 0.75, 0.18, 0.18])
    ax_inset.set_title("18×18 Patch", fontsize=8, pad=3)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    # Draw a border around the inset
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    inset_patch = np.zeros((18, 18), dtype=np.int32)
    im_inset = ax_inset.imshow(inset_patch, cmap=SAND_CMAP, norm=SAND_NORM,
                                interpolation="nearest", origin="upper")

    # --- Trial slider ---
    ax_slider_trial = plt.axes([0.15, 0.05, 0.73, 0.03])
    slider_trial = Slider(
        ax=ax_slider_trial,
        label="Trial",
        valmin=0,
        valmax=len(trial_indices) - 1,
        valinit=current_trial_idx_pos,
        valfmt="%d",
        valstep=1,
    )

    # --- Cached grid ---
    cache = {"idx": trial_idx, "grid": grid}

    def update(_=None):
        nonlocal current_trial_idx_pos, cache

        new_trial_pos = int(round(slider_trial.val))
        if new_trial_pos == current_trial_idx_pos:
            return

        current_trial_idx_pos = new_trial_pos
        trial_idx = trial_indices[current_trial_idx_pos]

        if cache["idx"] == trial_idx:
            grid = cache["grid"]
        else:
            path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
            grid = np.load(path)
            cache = {"idx": trial_idx, "grid": grid}

        im.set_data(grid)

        # Update perturbation site markers
        meta = metadata_map.get(trial_idx, {})
        perturb_sites = meta.get("perturb_sites", [])
        if perturb_sites:
            perturb_markers.set_data(
                [p[0] for p in perturb_sites],
                [p[1] for p in perturb_sites],
            )
            perturb_markers.set_visible(True)
        else:
            perturb_markers.set_visible(False)

        topples = meta.get("topples", "?")
        distinct = meta.get("distinct", "?")
        ax_title.set_text(
            f"2D Sandpile — Trial {trial_idx}  "
            f"(topples={topples:,}, distinct={distinct:,})"
        )

        fig.canvas.draw_idle()

    slider_trial.on_changed(update)

    # --- Keyboard shortcuts ---
    def on_key(event):
        if event.key == "right" or event.key == "d":
            slider_trial.set_val(
                min(current_trial_idx_pos + 1, len(trial_indices) - 1))
        elif event.key == "left" or event.key == "a":
            slider_trial.set_val(max(current_trial_idx_pos - 1, 0))

    # --- Mouse handlers for local 18×18 patch ---
    def _extract_patch(grid, x, y):
        """Extract an 18×18 patch centred at (x, y) from *grid*.
        Clamps to valid indices; returns the (possibly smaller) patch
        and its top-left corner (y0, x0) in grid coordinates."""
        N = grid.shape[0]
        half = 9
        x0 = max(0, x - half)
        x1 = min(N, x + half)
        y0 = max(0, y - half)
        y1 = min(N, y + half)
        return grid[y0:y1, x0:x1], y0, x0

    def _save_patch(patch, trial_idx, x, y):
        """Render and save a local patch to disk with grain-count labels."""
        save_name = f"patch_trial_{trial_idx:06d}_x{x}_y{y}.png"
        save_path = os.path.join(local_patches_dir, save_name)

        fig_p, ax_p = plt.subplots(figsize=(3.5, 3.5))
        ax_p.imshow(patch, cmap=SAND_CMAP, norm=SAND_NORM,
                    interpolation="nearest", origin="upper")
        ax_p.set_title(f"Trial {trial_idx}  centre=({x},{y})  "
                       f"{patch.shape[0]}×{patch.shape[1]}",
                       fontsize=9)
        # Annotate each cell with its grain count
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                ax_p.text(j, i, str(patch[i, j]),
                          ha="center", va="center", fontsize=6,
                          color="black" if patch[i, j] <= 2 else "white")
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        fig_p.tight_layout(pad=0.5)
        fig_p.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig_p)
        print(f"Saved 18×18 local patch → {save_path}")

    def on_mouse_move(event):
        """Update the 18×18 preview inset as the mouse moves over the grid."""
        if event.inaxes != ax_img:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        grid = cache["grid"]
        N = grid.shape[0]

        half = 9
        # Build an 18×18 padded array centred at (x,y)
        padded = np.full((18, 18), -1, dtype=np.int32)
        x0 = max(0, x - half)
        x1 = min(N, x + half)
        y0 = max(0, y - half)
        y1 = min(N, y + half)
        sub = grid[y0:y1, x0:x1]

        py0 = half - (y - y0)
        py1 = py0 + sub.shape[0]
        px0 = half - (x - x0)
        px1 = px0 + sub.shape[1]
        padded[py0:py1, px0:px1] = sub

        im_inset.set_data(padded)
        fig.canvas.draw_idle()

    def on_click(event):
        """Save the 18×18 local patch on mouse click (left button)."""
        if event.inaxes != ax_img:
            return
        if event.button != 1:  # left-click only
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        grid = cache["grid"]
        patch, _, _ = _extract_patch(grid, x, y)
        trial_idx = trial_indices[current_trial_idx_pos]
        _save_patch(patch, trial_idx, x, y)

    def on_key_extended(event):
        """Extended keyboard handler: 's' saves the patch at the
        last-known mouse position (falls back to grid centre)."""
        if event.key == "s":
            if event.inaxes == ax_img and event.xdata is not None:
                x = int(round(event.xdata))
                y = int(round(event.ydata))
            else:
                N = cache["grid"].shape[0]
                x, y = N // 2, N // 2
            grid = cache["grid"]
            patch, _, _ = _extract_patch(grid, x, y)
            trial_idx = trial_indices[current_trial_idx_pos]
            _save_patch(patch, trial_idx, x, y)
        else:
            on_key(event)  # delegate to the original arrow-key handler

    fig.canvas.mpl_connect("key_press_event", on_key_extended)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


def interactive_3d_viewer(grids_dir: str, stats_dir: str | None = None,
                          initial_trial: int = 0, initial_z: int | None = None):
    """Interactive 3D sandpile slice explorer.

    Opens a matplotlib window with two sliders:
      - Trial index: select which trial's grid to display
      - Z-slice: scroll through z-depths of the 3D grid

    The display updates in real time as sliders are moved.

    Parameters
    ----------
    grids_dir : str
        Directory containing trial_NNNNNN.npy files.
    stats_dir : str | None
        Directory with all_metadata.json for displaying trial info.
    initial_trial : int
        Starting trial index.
    initial_z : int | None
        Starting z-slice. If None, uses the middle slice.
    """
    import json
    from matplotlib.widgets import Slider

    # --- Discover available trials ---
    trial_indices = []
    for fname in sorted(os.listdir(grids_dir)):
        if fname.startswith("trial_") and fname.endswith(".npy"):
            try:
                idx = int(fname.replace("trial_", "").replace(".npy", ""))
                trial_indices.append(idx)
            except ValueError:
                pass
    trial_indices.sort()

    if not trial_indices:
        print(f"No trial grid files found in {grids_dir}")
        return

    # --- Load metadata if available ---
    metadata_map = {}
    if stats_dir is not None:
        meta_path = os.path.join(stats_dir, "all_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata_list = json.load(f)
            metadata_map = {i: m for i, m in enumerate(metadata_list)}

    # --- Load first grid to determine dimensions ---
    path0 = os.path.join(grids_dir, f"trial_{trial_indices[0]:06d}.npy")
    sample_grid = np.load(path0)
    n = sample_grid.shape[0]

    if initial_z is None:
        initial_z = n // 2

    # Clamp initial values
    trial_idx_pos = trial_indices.index(initial_trial) if initial_trial in trial_indices else 0
    current_trial_idx_pos = trial_idx_pos
    current_z = max(0, min(initial_z, n - 1))

    # --- Build the figure ---
    fig, ax_img = plt.subplots(figsize=(8, 9.5))
    plt.subplots_adjust(bottom=0.22, left=0.10, right=0.88)

    # Initial display
    trial_idx = trial_indices[current_trial_idx_pos]
    path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
    grid = np.load(path)

    im = ax_img.imshow(grid[:, :, current_z], cmap=SAND_CMAP, norm=SAND_NORM,
                       interpolation="nearest", origin="upper")

    meta = metadata_map.get(trial_idx, {})
    topples = meta.get("topples", "?")
    title_text = (f"3D Sandpile — Trial {trial_idx}  "
                  f"(z={current_z}/{n - 1}, topples={topples:,})")
    ax_title = ax_img.set_title(title_text, fontsize=13)
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")

    cbar_ax = fig.add_axes([0.90, 0.30, 0.03, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax,
                        ticks=range(len(_SAND_COLORS)))
    cbar.set_label("Grains", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # --- Slider axes ---
    ax_slider_trial = plt.axes([0.15, 0.10, 0.73, 0.03])
    ax_slider_z = plt.axes([0.15, 0.05, 0.73, 0.03])

    slider_trial = Slider(
        ax=ax_slider_trial,
        label="Trial",
        valmin=0,
        valmax=len(trial_indices) - 1,
        valinit=current_trial_idx_pos,
        valfmt="%d",
        valstep=1,
    )

    slider_z = Slider(
        ax=ax_slider_z,
        label="z-slice",
        valmin=0,
        valmax=n - 1,
        valinit=current_z,
        valfmt="%d",
        valstep=1,
    )

    # --- Cached grid to avoid reloading same trial ---
    cache = {"idx": trial_idx, "grid": grid}

    def update(_=None):
        nonlocal current_trial_idx_pos, current_z, cache

        new_trial_pos = int(round(slider_trial.val))
        new_z = int(round(slider_z.val))

        trial_changed = (new_trial_pos != current_trial_idx_pos)
        z_changed = (new_z != current_z)

        if not trial_changed and not z_changed:
            return

        current_trial_idx_pos = new_trial_pos
        current_z = new_z

        trial_idx = trial_indices[current_trial_idx_pos]

        # Load grid (use cache if same trial)
        if cache["idx"] == trial_idx:
            grid = cache["grid"]
        else:
            path = os.path.join(grids_dir, f"trial_{trial_idx:06d}.npy")
            grid = np.load(path)
            cache = {"idx": trial_idx, "grid": grid}

        im.set_data(grid[:, :, current_z])

        meta = metadata_map.get(trial_idx, {})
        topples = meta.get("topples", "?")
        ax_title.set_text(
            f"3D Sandpile — Trial {trial_idx}  "
            f"(z={current_z}/{n - 1}, topples={topples:,})"
        )

        fig.canvas.draw_idle()

    slider_trial.on_changed(update)
    slider_z.on_changed(update)

    # --- Keyboard shortcuts ---
    def on_key(event):
        if event.key == "right" or event.key == "d":
            slider_z.set_val(min(current_z + 1, n - 1))
        elif event.key == "left" or event.key == "a":
            slider_z.set_val(max(current_z - 1, 0))
        elif event.key == "up" or event.key == "w":
            slider_trial.set_val(
                min(current_trial_idx_pos + 1, len(trial_indices) - 1))
        elif event.key == "down" or event.key == "s":
            slider_trial.set_val(max(current_trial_idx_pos - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()




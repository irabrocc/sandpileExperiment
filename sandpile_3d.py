"""
3D Abelian Sandpile Model.

Each site (i,j,k) holds an integer number of grains.
A site topples when grains >= threshold (default 6 in 3D).
On toppling, it loses `threshold` grains and gives 1 grain to each
of its 6 orthogonal neighbours (±x, ±y, ±z). Grains that go off the
grid are lost (open boundary).

The stabilisation uses a queue of unstable sites for efficiency.
"""

from collections import deque
import numpy as np


class Sandpile3D:
    """3D Abelian sandpile on an n x n x n cube with open boundaries."""

    def __init__(self, n: int, threshold: int = 6):
        if n < 1:
            raise ValueError("n must be positive")
        if threshold < 2:
            raise ValueError("Threshold must be >= 2")
        self.n = n
        self.threshold = threshold
        self.grid = np.zeros((n, n, n), dtype=np.int32)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def fill(self, grains: int):
        """Set every site to *grains*."""
        self.grid.fill(grains)

    def add_grain(self, x: int, y: int, z: int, amount: int = 1):
        """Add *amount* grains at (x, y, z). Returns True if the site
        became unstable (>= threshold)."""
        self.grid[x, y, z] += amount
        return self.grid[x, y, z] >= self.threshold

    def add_grains_at(self, positions: list[tuple[int, int, int]],
                      amount: int = 1):
        """Add *amount* grains at each position in *positions*."""
        for x, y, z in positions:
            self.add_grain(x, y, z, amount)

    # ------------------------------------------------------------------
    # Stabilisation
    # ------------------------------------------------------------------

    def stabilize(self) -> dict:
        """Run toppling until every site is below threshold.

        Returns a dict with statistics:
            total_topples : int
            distinct_sites : int
            max_grains : int
            min_grains : int
        """
        n = self.n
        T = self.threshold
        grid = self.grid

        # Collect initially unstable sites
        unstable = np.argwhere(grid >= T)
        queue = deque()
        in_queue = set()

        for i, j, k in unstable:
            t = (int(i), int(j), int(k))
            queue.append(t)
            in_queue.add(t)

        total_topples = 0
        toppled_set = set()

        while queue:
            x, y, z = queue.popleft()
            in_queue.discard((x, y, z))

            if grid[x, y, z] < T:
                continue

            # ---- topple ----
            grid[x, y, z] -= T
            total_topples += 1
            toppled_set.add((x, y, z))

            # Distribute to 6 neighbours
            # +x
            if x + 1 < n:
                grid[x + 1, y, z] += 1
                if grid[x + 1, y, z] >= T and (x + 1, y, z) not in in_queue:
                    queue.append((x + 1, y, z))
                    in_queue.add((x + 1, y, z))
            # -x
            if x > 0:
                grid[x - 1, y, z] += 1
                if grid[x - 1, y, z] >= T and (x - 1, y, z) not in in_queue:
                    queue.append((x - 1, y, z))
                    in_queue.add((x - 1, y, z))
            # +y
            if y + 1 < n:
                grid[x, y + 1, z] += 1
                if grid[x, y + 1, z] >= T and (x, y + 1, z) not in in_queue:
                    queue.append((x, y + 1, z))
                    in_queue.add((x, y + 1, z))
            # -y
            if y > 0:
                grid[x, y - 1, z] += 1
                if grid[x, y - 1, z] >= T and (x, y - 1, z) not in in_queue:
                    queue.append((x, y - 1, z))
                    in_queue.add((x, y - 1, z))
            # +z
            if z + 1 < n:
                grid[x, y, z + 1] += 1
                if grid[x, y, z + 1] >= T and (x, y, z + 1) not in in_queue:
                    queue.append((x, y, z + 1))
                    in_queue.add((x, y, z + 1))
            # -z
            if z > 0:
                grid[x, y, z - 1] += 1
                if grid[x, y, z - 1] >= T and (x, y, z - 1) not in in_queue:
                    queue.append((x, y, z - 1))
                    in_queue.add((x, y, z - 1))

            # Re-check self: if still unstable, re-add to queue
            if grid[x, y, z] >= T and (x, y, z) not in in_queue:
                queue.append((x, y, z))
                in_queue.add((x, y, z))

        return {
            "total_topples": total_topples,
            "distinct_sites": len(toppled_set),
            "max_grains": int(grid.max()),
            "min_grains": int(grid.min()),
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_stable(self) -> bool:
        return bool((self.grid < self.threshold).all())

    def grain_distribution(self) -> dict[int, int]:
        """Return a dict {grains: count} for all sites."""
        unique, counts = np.unique(self.grid, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def copy_grid(self) -> np.ndarray:
        return self.grid.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        np.save(path, self.grid)

    def load(self, path: str):
        self.grid = np.load(path)
        self.n = self.grid.shape[0]

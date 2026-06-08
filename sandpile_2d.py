"""
2D Abelian Sandpile Model.

Each site (i,j) holds an integer number of grains.
A site topples when grains >= threshold (default 4).
On toppling, it loses `threshold` grains and gives 1 grain to each
of its 4 orthogonal neighbours. Grains that go off the grid are lost
(open boundary condition).

The stabilization uses a queue (deque) of unstable sites for
O(N^2 + topples) performance rather than repeated full-grid scans.
"""

from collections import deque
import numpy as np


class Sandpile2D:
    """2D Abelian sandpile on an N x N grid with open boundaries."""

    def __init__(self, N: int, threshold: int = 4):
        if N < 1:
            raise ValueError("N must be positive")
        if threshold < 2:
            raise ValueError("Threshold must be >= 2")
        self.N = N
        self.threshold = threshold
        self.grid = np.zeros((N, N), dtype=np.int32)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def fill(self, grains: int):
        """Set every site to *grains*."""
        self.grid.fill(grains)

    def add_grain(self, x: int, y: int, amount: int = 1):
        """Add *amount* grains at (x, y). Returns True if the site
        became unstable (>= threshold)."""
        self.grid[x, y] += amount
        return self.grid[x, y] >= self.threshold

    def add_grains_at(self, positions: list[tuple[int, int]], amount: int = 1):
        """Add *amount* grains at each position in *positions*."""
        for x, y in positions:
            self.add_grain(x, y, amount)

    # ------------------------------------------------------------------
    # Stabilisation
    # ------------------------------------------------------------------

    def stabilize(self) -> dict:
        """Run toppling until every site is below threshold.

        Returns a dict with statistics:
            total_topples : int       – total number of toppling events
            distinct_sites : int      – number of distinct sites that toppled
            max_grains : int          – maximum grains on any site after stabilization
            min_grains : int          – minimum grains (usually 0, 1, 2, or 3 in stable config)
        """
        N = self.N
        T = self.threshold
        grid = self.grid

        # Collect initially unstable sites
        unstable = np.argwhere(grid >= T)
        queue = deque()
        in_queue = set()

        for i, j in unstable:
            t = (int(i), int(j))
            queue.append(t)
            in_queue.add(t)

        total_topples = 0
        toppled_set = set()

        while queue:
            x, y = queue.popleft()
            in_queue.discard((x, y))

            if grid[x, y] < T:
                continue  # stabilised by a neighbour's toppling

            # ---- topple ----
            grid[x, y] -= T
            total_topples += 1
            toppled_set.add((x, y))

            # Distribute to 4 neighbours
            # up
            if x > 0:
                grid[x - 1, y] += 1
                if grid[x - 1, y] >= T and (x - 1, y) not in in_queue:
                    queue.append((x - 1, y))
                    in_queue.add((x - 1, y))
            # down
            if x < N - 1:
                grid[x + 1, y] += 1
                if grid[x + 1, y] >= T and (x + 1, y) not in in_queue:
                    queue.append((x + 1, y))
                    in_queue.add((x + 1, y))
            # left
            if y > 0:
                grid[x, y - 1] += 1
                if grid[x, y - 1] >= T and (x, y - 1) not in in_queue:
                    queue.append((x, y - 1))
                    in_queue.add((x, y - 1))
            # right
            if y < N - 1:
                grid[x, y + 1] += 1
                if grid[x, y + 1] >= T and (x, y + 1) not in in_queue:
                    queue.append((x, y + 1))
                    in_queue.add((x, y + 1))

            # Re-check self: if still unstable, re-add to queue
            if grid[x, y] >= T and (x, y) not in in_queue:
                queue.append((x, y))
                in_queue.add((x, y))

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
        self.N = self.grid.shape[0]

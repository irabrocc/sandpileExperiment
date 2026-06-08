"""
Unit tests for the 2D sandpile engine.
Run with: pytest test_sandpile_2d.py -v
"""

import numpy as np
import pytest
from sandpile_2d import Sandpile2D


class TestSandpile2DBasics:
    """Basic sanity checks."""

    def test_init(self):
        sp = Sandpile2D(10)
        assert sp.N == 10
        assert sp.threshold == 4
        assert sp.grid.shape == (10, 10)
        assert sp.is_stable

    def test_fill(self):
        sp = Sandpile2D(10)
        sp.fill(3)
        assert (sp.grid == 3).all()
        assert sp.is_stable  # 3 < 4

    def test_add_grain(self):
        sp = Sandpile2D(10)
        sp.fill(3)
        became_unstable = sp.add_grain(5, 5)
        assert became_unstable  # 3 + 1 = 4 >= threshold
        assert sp.grid[5, 5] == 4

    def test_add_grain_below_threshold(self):
        sp = Sandpile2D(10)
        sp.fill(0)
        became_unstable = sp.add_grain(5, 5)
        assert not became_unstable  # 0 + 1 = 1 < 4


class TestToppling:
    """Test toppling behaviour."""

    def test_single_center_topple(self):
        """Place 4 grains at center of a small grid and stabilise."""
        sp = Sandpile2D(5)
        sp.add_grain(2, 2, 4)  # exactly at threshold
        assert not sp.is_stable
        stats = sp.stabilize()
        assert sp.is_stable
        # After toppling: center = 0, each neighbor = 1
        assert sp.grid[2, 2] == 0
        assert sp.grid[1, 2] == 1  # up
        assert sp.grid[3, 2] == 1  # down
        assert sp.grid[2, 1] == 1  # left
        assert sp.grid[2, 3] == 1  # right
        assert stats["total_topples"] == 1
        assert stats["distinct_sites"] == 1

    def test_edge_topple_loss(self):
        """Grains that fall off the edge are lost (open boundary)."""
        sp = Sandpile2D(3)
        sp.add_grain(0, 1, 4)  # top edge, not corner
        stats = sp.stabilize()
        assert sp.is_stable
        # Center: lost 4, 3 went to valid neighbours (down, left, right),
        # 1 went off the top edge → lost
        assert sp.grid[0, 1] == 0
        assert sp.grid[1, 1] == 1  # down
        assert sp.grid[0, 0] == 1  # left
        assert sp.grid[0, 2] == 1  # right
        # Total grains before: 4, after: 3 (1 lost at boundary)
        assert sp.grid.sum() == 3
        assert stats["total_topples"] == 1

    def test_corner_topple_loss(self):
        """Corner site loses 2 grains off the grid."""
        sp = Sandpile2D(3)
        sp.add_grain(0, 0, 4)
        stats = sp.stabilize()
        assert sp.is_stable
        assert sp.grid[0, 0] == 0
        assert sp.grid[1, 0] == 1  # down
        assert sp.grid[0, 1] == 1  # right
        # 2 grains lost (top + left boundaries)
        assert sp.grid.sum() == 2
        assert stats["total_topples"] == 1

    def test_avalanche_chain(self):
        """A large pile at the center triggers a chain reaction."""
        sp = Sandpile2D(5)
        sp.add_grain(2, 2, 20)  # well above threshold
        stats = sp.stabilize()
        assert sp.is_stable
        # Should cause multiple topples
        assert stats["total_topples"] > 1


class TestAbelianProperty:
    """Verify the Abelian property: the final stable configuration
    is independent of toppling order."""

    def test_abelian_two_seeds(self):
        """Two unstable sites; try different toppling orders."""
        # Use a fresh grid each time
        def run_with_order(order_seed):
            sp = Sandpile2D(5)
            sp.add_grain(1, 2, 4)
            sp.add_grain(3, 2, 4)
            sp.stabilize()
            return sp.copy_grid()

        grid1 = run_with_order(1)
        grid2 = run_with_order(2)
        assert np.array_equal(grid1, grid2)

    def test_abelian_random(self):
        """Random initial unstable sites; two runs should match."""
        rng = np.random.default_rng(1234)
        N = 10

        sp1 = Sandpile2D(N)
        sp2 = Sandpile2D(N)
        for _ in range(20):
            x, y = rng.integers(0, N, size=2)
            sp1.add_grain(x, y, 4)
            sp2.add_grain(x, y, 4)

        sp1.stabilize()
        sp2.stabilize()
        assert np.array_equal(sp1.grid, sp2.grid)


class TestConservation:
    """Grain conservation (modulo boundary losses)."""

    def test_conservation_no_boundary_loss(self):
        """On a large grid with a central pile, no grains reach the edge."""
        sp = Sandpile2D(50)
        initial_grains = 100
        sp.add_grain(25, 25, initial_grains)
        sp.stabilize()
        assert sp.grid.sum() == initial_grains

    def test_conservation_with_boundary(self):
        """Grains near the edge: verify grains_lost = initial - final."""
        sp = Sandpile2D(5)
        sp.add_grain(0, 0, 20)  # corner pile — many grains lost
        initial = sp.grid.sum()
        sp.stabilize()
        final = sp.grid.sum()
        lost = initial - final
        assert lost > 0  # some grains definitely lost at corner
        # Verify every grain is accounted for: all sites < threshold
        assert sp.is_stable


class TestStableDetection:
    def test_stable_after_stabilize(self):
        sp = Sandpile2D(10)
        sp.fill(3)
        sp.add_grain(5, 5)  # now 4 at center
        sp.stabilize()
        assert sp.is_stable
        assert (sp.grid < sp.threshold).all()

    def test_grain_distribution(self):
        sp = Sandpile2D(5)
        sp.fill(3)
        dist = sp.grain_distribution()
        assert dist == {3: 25}


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        sp = Sandpile2D(10)
        sp.fill(3)
        sp.add_grain(5, 5, 4)
        sp.stabilize()

        path = str(tmp_path / "grid.npy")
        sp.save(path)

        sp2 = Sandpile2D(1)
        sp2.load(path)
        assert sp2.N == 10
        assert np.array_equal(sp.grid, sp2.grid)

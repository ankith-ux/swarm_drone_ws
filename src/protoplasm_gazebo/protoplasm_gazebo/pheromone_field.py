"""
pheromone_field.py — Protoplasm Pheromone Field (Port of field.js)

The shared "digital slime mold" substrate. Each cell stores pheromone
strength that decays over time unless reinforced by repeat drone traffic.
No LLM, no central planner — pure deterministic math.

Direct port of field.js with NumPy arrays replacing JavaScript typed arrays.
"""

import numpy as np
from typing import Tuple, Optional


class PheromoneField:
    """
    Grid-based pheromone field for stigmergic coordination.
    All parameters match the JavaScript implementation exactly.
    """

    def __init__(
        self,
        cols: int = 22,
        rows: int = 18,
        base_decay_rate: float = 0.035,
        reinforce_bonus: float = 0.05,
        reinforce_window: int = 15,
        max_strength: float = 0.85,
        min_strength: float = 0.0,
    ):
        self.cols = cols
        self.rows = rows
        self.base_decay_rate = base_decay_rate
        self.reinforce_bonus = reinforce_bonus
        self.reinforce_window = reinforce_window
        self.max_strength = max_strength
        self.min_strength = min_strength

        # NumPy arrays replacing JS Float32Array / Uint16Array / Int32Array
        n = cols * rows
        self.strength = np.zeros(n, dtype=np.float32)          # Current pheromone level
        self.visit_count = np.zeros(n, dtype=np.uint16)         # Total deposits
        self.last_visit = np.full(n, -9999, dtype=np.int32)     # Tick of last deposit
        self.unique_drone_count = np.zeros(n, dtype=np.uint8)   # Unique verifiers
        self.exploration_grid = np.zeros(n, dtype=np.float32)   # 0.0=fog, 1.0=scanned
        self.cell_uncertainty = np.full(n, 0.80, dtype=np.float32)  # Spatial uncertainty
        self.cell_confidence = np.zeros(n, dtype=np.float32)    # Spatial confidence

        # Set tracking unique drone IDs per cell (prevents single-drone false consensus)
        self._unique_drone_sets: list[set] = [set() for _ in range(n)]

        self._tick = 0

    # ── Indexing helpers ──────────────────────────────────────────────────────

    def _idx(self, col: int, row: int) -> int:
        return row * self.cols + col

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    # ── Public API ────────────────────────────────────────────────────────────

    def deposit(
        self,
        col: int,
        row: int,
        amount: float,
        drone_id: Optional[int] = None,
        confidence: float = 0.0,
        uncertainty: float = 0.0,
    ):
        """
        Deposit pheromone at a grid cell.
        Diminishing returns for repeated visits from the SAME drone.
        Over-validation cap when unique drone count >= 3.
        """
        if not self._in_bounds(col, row):
            return
        i = self._idx(col, row)

        drone_set = self._unique_drone_sets[i]
        is_new_unique = drone_id is not None and drone_id not in drone_set

        deposit_multiplier = 1.0
        if drone_id is not None:
            if is_new_unique:
                drone_set.add(drone_id)
                self.unique_drone_count[i] = len(drone_set)

                # Over-Validation Cap: 3+ unique drones → 0.15x to prevent saturation
                if len(drone_set) > 3:
                    deposit_multiplier = 0.15
                else:
                    # High info gain for NEW independent corroboration
                    deposit_multiplier = 1.35
            else:
                # Diminishing returns for same drone revisiting
                deposit_multiplier = 0.35

        actual_deposit = amount * deposit_multiplier
        self.strength[i] = min(self.max_strength, self.strength[i] + actual_deposit)
        self.visit_count[i] += 1
        self.last_visit[i] = self._tick

        # Update spatial confidence (running weighted average)
        if confidence > 0:
            alpha = 0.3
            self.cell_confidence[i] = (1 - alpha) * self.cell_confidence[i] + alpha * confidence

        # Reduce spatial uncertainty based on visit density
        if uncertainty < self.cell_uncertainty[i]:
            self.cell_uncertainty[i] = 0.7 * self.cell_uncertainty[i] + 0.3 * uncertainty

        # Mark cell as explored
        explore_gain = 0.25 if is_new_unique else 0.08
        self.exploration_grid[i] = min(1.0, self.exploration_grid[i] + explore_gain)

    def tick(self):
        """Advance one tick — apply global pheromone decay."""
        self._tick += 1

        # Exponential decay: strength *= (1 - decay_rate)
        self.strength *= (1.0 - self.base_decay_rate)

        # Clamp to minimum
        self.strength = np.maximum(self.strength, self.min_strength)

        # Slowly increase uncertainty in unvisited cells (entropy drift)
        stale_mask = (self._tick - self.last_visit) > self.reinforce_window
        self.cell_uncertainty[stale_mask] = np.minimum(
            1.0, self.cell_uncertainty[stale_mask] + 0.002
        )

    def get_strength(self, col: int, row: int) -> float:
        """Get pheromone strength at (col, row)."""
        if not self._in_bounds(col, row):
            return 0.0
        return float(self.strength[self._idx(col, row)])

    def get_unique_drone_count(self, col: int, row: int) -> int:
        """Get count of unique drones that have visited (col, row)."""
        if not self._in_bounds(col, row):
            return 0
        return int(self.unique_drone_count[self._idx(col, row)])

    def get_gradient(self, col: int, row: int) -> Tuple[float, float]:
        """
        Compute the local pheromone gradient at (col, row).
        Returns (dx, dy) pointing toward increasing pheromone.
        """
        dx = 0.0
        dy = 0.0
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dc == 0 and dr == 0:
                    continue
                nc, nr = col + dc, row + dr
                if self._in_bounds(nc, nr):
                    s = self.strength[self._idx(nc, nr)]
                    dx += dc * s
                    dy += dr * s
        return (dx, dy)

    def get_uncertainty_gradient(self, col: int, row: int) -> Tuple[float, float]:
        """
        Compute gradient pointing toward HIGHER uncertainty (unexplored areas).
        Used for exploration drive.
        """
        dx = 0.0
        dy = 0.0
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dc == 0 and dr == 0:
                    continue
                nc, nr = col + dc, row + dr
                if self._in_bounds(nc, nr):
                    u = self.cell_uncertainty[self._idx(nc, nr)]
                    dx += dc * u
                    dy += dr * u
        return (dx, dy)

    def get_exploration_gradient(self, col: int, row: int) -> Tuple[float, float]:
        """
        Compute gradient pointing AWAY from permanently explored areas.
        Unlike uncertainty (which drifts back over time), exploration_grid never decreases.
        Points towards areas with lower exploration_grid values.
        """
        dx = 0.0
        dy = 0.0
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dc == 0 and dr == 0:
                    continue
                nc, nr = col + dc, row + dr
                if self._in_bounds(nc, nr):
                    # Invert the exploration value: we want to go towards 0.0, away from 1.0
                    unexplored = 1.0 - self.exploration_grid[self._idx(nc, nr)]
                    dx += dc * unexplored
                    dy += dr * unexplored
        return (dx, dy)

    def clear_local_attraction(self, col: int, row: int, radius: float):
        """
        Clear pheromone in a circular area around (col, row).
        Used after false-positive rejection or survivor extraction.
        """
        r_sq = radius * radius
        for r in range(self.rows):
            for c in range(self.cols):
                dist_sq = (c - col) ** 2 + (r - row) ** 2
                if dist_sq <= r_sq:
                    i = self._idx(c, r)
                    self.strength[i] *= 0.1
                    self.cell_confidence[i] *= 0.2

    def get_explored_percentage(self) -> float:
        """Return the percentage of cells that have been explored."""
        explored = np.sum(self.exploration_grid > 0.15)
        return float(explored / len(self.exploration_grid)) * 100.0

    def reset(self):
        """Reset the entire field to initial state."""
        n = self.cols * self.rows
        self.strength[:] = 0
        self.visit_count[:] = 0
        self.last_visit[:] = -9999
        self.unique_drone_count[:] = 0
        self.exploration_grid[:] = 0
        self.cell_uncertainty[:] = 0.80
        self.cell_confidence[:] = 0
        self._unique_drone_sets = [set() for _ in range(n)]
        self._tick = 0

    def to_grid_2d(self) -> np.ndarray:
        """Return pheromone strength as a 2D (rows, cols) array for visualization."""
        return self.strength.reshape(self.rows, self.cols)

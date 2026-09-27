"""
scenario_loader.py — Disaster Floorplan + Survivor Placement

Defines synthetic disaster scenarios as 2D grids of cell types.
Maps Gazebo world coordinates (meters) to grid cells.
Drones do NOT see this directly — they only see noisy sensor readings.

Supports:
  - 'disaster-zone': Original 50m×40m scenario (5 fixed survivors)
  - 'grand-challenge': 1000m×1000m scenario with 10 randomly-placed POIs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import random
import time


# ── Cell Types ────────────────────────────────────────────────────────────────
CELL_CLEAR = 'CLEAR'
CELL_RUBBLE = 'RUBBLE'
CELL_SURVIVOR = 'SURVIVOR'
CELL_HOT_DEBRIS = 'HOT_DEBRIS'
CELL_WIND_NOISE = 'WIND_NOISE'
CELL_HAZARD = 'HAZARD'


@dataclass
class SurvivorTarget:
    """Metadata for a survivor / POI placed in the scenario."""
    id: str
    name: str
    col: int
    row: int
    sector: str
    status: str = 'SEARCHING'    # SEARCHING → CONVERGING → RESCUE_DISPATCH → EXTRACTED
    confidence: float = 0.0
    solidify_ticks: int = 0
    is_primary_target: bool = False
    spawn_time: float = 0.0      # When this POI appears (seconds into mission)


@dataclass
class Structure:
    """A structural building footprint for rendering."""
    c0: int
    r0: int
    c1: int
    r1: int
    label: str


class DisasterScenario:
    """
    2D grid of cell types representing the ground truth of a disaster zone.
    Provides coordinate conversion between Gazebo world (meters) and grid cells.
    """

    def __init__(
        self,
        cols: int = 50,
        rows: int = 50,
        world_size_x: float = 1000.0,
        world_size_y: float = 1000.0,
        name: str = 'grand-challenge',
    ):
        self.cols = cols
        self.rows = rows
        self.world_size_x = world_size_x
        self.world_size_y = world_size_y
        self.name = name

        # Cell size in meters
        self.cell_size_x = world_size_x / cols
        self.cell_size_y = world_size_y / rows

        # Grid: flat array of cell type strings, indexed row * cols + col
        self.grid: List[str] = [CELL_CLEAR] * (cols * rows)

        # Survivor / POI targets
        self.survivors: List[SurvivorTarget] = []

        # Building footprints
        self.structures: List[Structure] = []

        # POIs that haven't spawned yet (grand-challenge mode)
        self.pending_pois: List[SurvivorTarget] = []

        self._build_scenario(name)

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert Gazebo world coordinates (meters) to grid cell (col, row).
        Gazebo world origin is at center; grid origin is top-left.
        """
        # Shift so (0,0) is at the corner instead of center
        gx = x + self.world_size_x / 2.0
        gy = y + self.world_size_y / 2.0
        col = int(gx / self.cell_size_x)
        row = int(gy / self.cell_size_y)
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return (col, row)

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """
        Convert grid cell (col, row) to Gazebo world coordinates (meters).
        Returns the center of the cell.
        """
        x = (col + 0.5) * self.cell_size_x - self.world_size_x / 2.0
        y = (row + 0.5) * self.cell_size_y - self.world_size_y / 2.0
        return (x, y)

    def survivor_world_positions(self) -> List[Tuple[str, float, float, float]]:
        """
        Return survivor positions in Gazebo world coordinates.
        Returns list of (name, x, y, z) tuples.
        """
        positions = []
        for s in self.survivors:
            x, y = self.grid_to_world(s.col, s.row)
            positions.append((s.name, x, y, 0.0))
        return positions

    # ── Grid access ───────────────────────────────────────────────────────────

    def get_cell_type(self, col: int, row: int) -> Optional[str]:
        """Get the cell type at grid position. Returns None for out-of-bounds."""
        if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
            return None
        return self.grid[row * self.cols + col]

    def get_cell_type_at_world(self, x: float, y: float) -> Optional[str]:
        """Get the cell type at a Gazebo world position."""
        col, row = self.world_to_grid(x, y)
        return self.get_cell_type(col, row)

    def get_survivor_positions(self) -> List[Tuple[int, int]]:
        """Return all survivor cell positions."""
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r * self.cols + c] == CELL_SURVIVOR:
                    positions.append((c, r))
        return positions

    # ── Dynamic POI spawning (grand-challenge) ────────────────────────────────

    def check_and_spawn_pois(self, elapsed_time: float) -> List[SurvivorTarget]:
        """
        Check if any pending POIs should spawn based on elapsed mission time.
        Returns list of newly spawned POIs.
        """
        newly_spawned = []
        remaining = []
        for poi in self.pending_pois:
            if elapsed_time >= poi.spawn_time:
                # Activate this POI on the grid
                self._fill_rect(CELL_SURVIVOR,
                                poi.col - 1, poi.row - 1,
                                poi.col + 1, poi.row + 1)
                self.survivors.append(poi)
                newly_spawned.append(poi)
            else:
                remaining.append(poi)
        self.pending_pois = remaining
        return newly_spawned

    # ── Scenario builders ─────────────────────────────────────────────────────

    def _fill_rect(self, cell_type: str, c0: int, r0: int, c1: int, r1: int):
        """Fill a rectangular region with a cell type."""
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= c < self.cols and 0 <= r < self.rows:
                    self.grid[r * self.cols + c] = cell_type

    def _build_scenario(self, name: str):
        """Build the scenario grid."""
        self.survivors = []
        self.structures = []
        self.pending_pois = []

        if name == 'grand-challenge':
            self._build_grand_challenge()
        elif name in ('disaster-zone', 'ambiguous'):
            self._build_disaster_zone()
        elif name == 'simple':
            self._build_simple()
        elif name == 'multi-survivor':
            self._build_multi_survivor()

    def _build_grand_challenge(self):
        """
        Grand Challenge 1 scenario: 1000m × 1000m area.
        10 POIs spawned at random positions and random times.
        Hazard zones, rubble, and decoys scattered throughout.
        """
        # Base floor: mostly clear with rubble borders
        for r in range(self.rows):
            for c in range(self.cols):
                # Border cells are rubble
                if c <= 1 or c >= self.cols - 2 or r <= 1 or r >= self.rows - 2:
                    self.grid[r * self.cols + c] = CELL_RUBBLE
                else:
                    self.grid[r * self.cols + c] = CELL_CLEAR

        # Scatter some hazard zones
        rng = random.Random(42)  # Deterministic seed for reproducibility
        for _ in range(5):
            hc = rng.randint(5, self.cols - 6)
            hr = rng.randint(5, self.rows - 6)
            self._fill_rect(CELL_HAZARD, hc - 1, hr - 1, hc + 1, hr + 1)

        # Scatter some hot debris (false positives)
        for _ in range(8):
            dc = rng.randint(5, self.cols - 6)
            dr = rng.randint(5, self.rows - 6)
            self._fill_rect(CELL_HOT_DEBRIS, dc - 1, dr - 1, dc + 1, dr + 1)

        # Scatter some wind noise zones (audio false positives)
        for _ in range(6):
            wc = rng.randint(5, self.cols - 6)
            wr = rng.randint(5, self.rows - 6)
            self._fill_rect(CELL_WIND_NOISE, wc - 1, wr - 1, wc + 1, wr + 1)

        # 10 POIs placed at random positions across the area
        # Drones discover these autonomously via sensor fusion
        poi_rng = random.Random(123)  # Deterministic seed for reproducibility

        for i in range(10):
            pc = poi_rng.randint(4, self.cols - 5)
            pr = poi_rng.randint(4, self.rows - 5)
            # Place survivor cell on the grid immediately (1 cell only)
            self.grid[pr * self.cols + pc] = CELL_SURVIVOR
            self.survivors.append(SurvivorTarget(
                id=f'POI_{i}',
                name=f'POI {i}',
                col=pc,
                row=pr,
                sector=f'SECTOR ({pc},{pr})',
            ))

    def _build_disaster_zone(self):
        """Original 50m×40m disaster zone scenario (backward compatible)."""
        # Base floor: clear corridors with rubble fields
        for r in range(self.rows):
            for c in range(self.cols):
                is_corridor = 2 <= c <= 19 and 2 <= r <= 15
                self.grid[r * self.cols + c] = CELL_CLEAR if is_corridor else CELL_RUBBLE

        # 5 DISTINCT SURVIVORS across the disaster zone
        # 1. Survivor Alpha — Northeast Ruined Tower (Sector 16, 4)
        self._fill_rect(CELL_SURVIVOR, 15, 3, 17, 5)
        self.survivors.append(SurvivorTarget(
            id='ALPHA', name='Survivor Alpha', col=16, row=4, sector='NE TOWER SECTOR'))

        # 2. Survivor Bravo — Southwest Collapsed Complex (Sector 5, 13)
        self._fill_rect(CELL_SURVIVOR, 4, 12, 6, 14)
        self.survivors.append(SurvivorTarget(
            id='BRAVO', name='Survivor Bravo', col=5, row=13, sector='SW RUINS SECTOR'))

        # 3. Survivor Charlie — Northwest Hazard Perimeter (Sector 5, 4)
        self._fill_rect(CELL_SURVIVOR, 4, 3, 6, 5)
        self.survivors.append(SurvivorTarget(
            id='CHARLIE', name='Survivor Charlie', col=5, row=4, sector='NW STRUCTURAL GAP'))

        # 4. Survivor Delta — Southeast Vault Ruin (Sector 16, 13)
        self._fill_rect(CELL_SURVIVOR, 15, 12, 17, 14)
        self.survivors.append(SurvivorTarget(
            id='DELTA', name='Survivor Delta', col=16, row=13, sector='SE VAULT SECTOR'))

        # 5. Survivor Echo — Center Corridor Shelter (Sector 11, 9)
        self._fill_rect(CELL_SURVIVOR, 10, 8, 12, 10)
        self.survivors.append(SurvivorTarget(
            id='ECHO', name='Survivor Echo', col=11, row=9, sector='CENTER CORRIDOR'))

        # Hazards & Thermal Debris
        self._fill_rect(CELL_HOT_DEBRIS, 14, 11, 18, 15)
        self._fill_rect(CELL_WIND_NOISE, 10, 7, 12, 9)
        self._fill_rect(CELL_HAZARD, 9, 2, 11, 4)

        # Building footprints
        self.structures.append(Structure(2, 2, 7, 6, 'SECTOR W-1'))
        self.structures.append(Structure(14, 2, 19, 6, 'SECTOR E-1'))
        self.structures.append(Structure(2, 11, 7, 16, 'SECTOR W-2'))
        self.structures.append(Structure(14, 11, 19, 16, 'SECTOR E-2'))

    def _build_simple(self):
        """Simple single-survivor scenario."""
        for r in range(self.rows):
            for c in range(self.cols):
                is_corridor = 2 <= c <= 19 and 2 <= r <= 15
                self.grid[r * self.cols + c] = CELL_CLEAR if is_corridor else CELL_RUBBLE

        self._fill_rect(CELL_SURVIVOR, 15, 3, 18, 6)
        self.survivors.append(SurvivorTarget(
            id='ALPHA', name='Survivor Alpha', col=16, row=4, sector='SECTOR E-1'))
        self._fill_rect(CELL_HOT_DEBRIS, 3, 2, 7, 6)
        self._fill_rect(CELL_WIND_NOISE, 3, 11, 7, 15)
        self._fill_rect(CELL_HAZARD, 10, 7, 12, 9)

    def _build_multi_survivor(self):
        """Two-survivor scenario."""
        for r in range(self.rows):
            for c in range(self.cols):
                is_corridor = 2 <= c <= 19 and 2 <= r <= 15
                self.grid[r * self.cols + c] = CELL_CLEAR if is_corridor else CELL_RUBBLE

        self._fill_rect(CELL_SURVIVOR, 14, 2, 18, 5)
        self.survivors.append(SurvivorTarget(
            id='ALPHA', name='Survivor Alpha', col=16, row=3, sector='NORTH SECTOR'))
        self._fill_rect(CELL_SURVIVOR, 3, 11, 7, 14)
        self.survivors.append(SurvivorTarget(
            id='BRAVO', name='Survivor Bravo', col=5, row=12, sector='SOUTH SECTOR'))
        self._fill_rect(CELL_HOT_DEBRIS, 14, 11, 18, 15)
        self._fill_rect(CELL_WIND_NOISE, 3, 2, 7, 6)

    @staticmethod
    def get_scenario_names() -> List[str]:
        return ['grand-challenge', 'disaster-zone', 'simple', 'multi-survivor']

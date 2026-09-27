"""
sensor_fusion.py — Multi-Modal Sensor Fusion (Port of sensors.js)

Generates sensor readings for each cell type in a disaster scenario.
All values are [0, 1] normalized.

Cell types and their expected sensor profiles:
  SURVIVOR    → thermal HIGH, audio MEDIUM-HIGH, camera HIGH, gas MEDIUM
  HOT_DEBRIS  → thermal HIGH, audio LOW, camera LOW, gas LOW  (thermal false positive)
  WIND_NOISE  → thermal LOW, audio HIGH, camera LOW, gas LOW  (audio false positive)
  HAZARD      → thermal LOW, audio LOW, camera LOW, gas HIGH  (hazard zone)
  RUBBLE      → all LOW with minor noise
  CLEAR       → all very LOW

Vantage-point modulation: approach angle & distance affect sensor readings.
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple


# Noise parameter — calibrated to RoboCup Rescue-style sensor variability
NOISE_SIGMA = 0.08


def _gaussian(mean: float, sigma: float) -> float:
    """Sample from a clipped Gaussian distribution [0, 1]."""
    sample = np.random.normal(mean, sigma)
    return float(np.clip(sample, 0.0, 1.0))


def _wobble(tick: int, freq: float = 1.0, amp: float = 0.05) -> float:
    """Slow sinusoidal wobble to make readings feel 'live'."""
    return amp * math.sin((tick / 100.0) * freq * 2.0 * math.pi)


# ── Cell-type sensor profiles ────────────────────────────────────────────────
# Each profile returns {thermal, audio, camera, gas} ∈ [0, 1]

def _profile_survivor(tick: int) -> Dict[str, float]:
    return {
        'thermal': _gaussian(0.82 + _wobble(tick, 0.7, 0.04), NOISE_SIGMA),
        'audio':   _gaussian(0.75 + _wobble(tick, 1.3, 0.08), NOISE_SIGMA * 1.2),
        'camera':  _gaussian(0.88 + _wobble(tick, 0.5, 0.05), NOISE_SIGMA * 0.8),
        'gas':     _gaussian(0.45 + _wobble(tick, 0.4, 0.03), NOISE_SIGMA),
    }


def _profile_hot_debris(tick: int) -> Dict[str, float]:
    return {
        'thermal': _gaussian(0.82 + _wobble(tick, 0.3, 0.03), NOISE_SIGMA * 0.7),
        'audio':   _gaussian(0.07, NOISE_SIGMA * 0.6),
        'camera':  _gaussian(0.02, NOISE_SIGMA * 0.4),
        'gas':     _gaussian(0.12, NOISE_SIGMA * 0.8),
    }


def _profile_wind_noise(tick: int) -> Dict[str, float]:
    return {
        'thermal': _gaussian(0.10, NOISE_SIGMA * 0.5),
        'audio':   _gaussian(0.78 + _wobble(tick, 3.1, 0.15), NOISE_SIGMA * 1.5),
        'camera':  _gaussian(0.03, NOISE_SIGMA * 0.4),
        'gas':     _gaussian(0.08, NOISE_SIGMA * 0.4),
    }


def _profile_hazard(tick: int) -> Dict[str, float]:
    return {
        'thermal': _gaussian(0.18 + _wobble(tick, 0.5, 0.02), NOISE_SIGMA),
        'audio':   _gaussian(0.06, NOISE_SIGMA * 0.5),
        'camera':  _gaussian(0.02, NOISE_SIGMA * 0.4),
        'gas':     _gaussian(0.88 + _wobble(tick, 0.2, 0.02), NOISE_SIGMA * 0.6),
    }


def _profile_rubble(tick: int) -> Dict[str, float]:
    return {
        'thermal': _gaussian(0.18, NOISE_SIGMA * 1.1),
        'audio':   _gaussian(0.12, NOISE_SIGMA * 1.3),
        'camera':  _gaussian(0.05, NOISE_SIGMA * 1.1),
        'gas':     _gaussian(0.10, NOISE_SIGMA),
    }


def _profile_clear(tick: int) -> Dict[str, float]:
    return {
        'thermal': _gaussian(0.06, NOISE_SIGMA * 0.5),
        'audio':   _gaussian(0.05, NOISE_SIGMA * 0.5),
        'camera':  _gaussian(0.02, NOISE_SIGMA * 0.4),
        'gas':     _gaussian(0.04, NOISE_SIGMA * 0.4),
    }


# Profile lookup table
PROFILES = {
    'SURVIVOR':   _profile_survivor,
    'HOT_DEBRIS': _profile_hot_debris,
    'WIND_NOISE': _profile_wind_noise,
    'HAZARD':     _profile_hazard,
    'RUBBLE':     _profile_rubble,
    'CLEAR':      _profile_clear,
}


class SensorFusion:
    """
    Generates multi-modal sensor readings based on a drone's position
    relative to the disaster scenario grid.
    """

    def __init__(self, scenario):
        """
        Args:
            scenario: DisasterScenario instance providing ground truth cell types.
        """
        self.scenario = scenario

    def get_sensor_at(self, col: int, row: int, tick: int) -> Optional[Dict[str, float]]:
        """Get raw sensor reading at grid position (col, row) at given tick."""
        cell_type = self.scenario.get_cell_type(col, row)
        if cell_type is None:
            return None
        profile_fn = PROFILES.get(cell_type, _profile_clear)
        return profile_fn(tick)

    def get_local_readings(
        self,
        col: int,
        row: int,
        tick: int,
        drone_heading: float = 0.0,
        drone_x: float = 0.0,
        drone_y: float = 0.0,
        drone_id: int = 0,
    ) -> Dict[str, float]:
        """
        Get averaged readings in a 3×3 neighborhood with vantage-point modulation.

        Args:
            col, row: Grid cell position
            tick: Current simulation tick
            drone_heading: Drone heading in radians
            drone_x, drone_y: Drone's sub-cell position
            drone_id: Drone ID for distance wobble seeding

        Returns:
            Dict with thermal, audio, camera, gas, vantage_factor keys.
        """
        thermal = audio = camera = gas = 0.0
        count = 0

        # Vantage-point modulation (approach angle + distance)
        # Convert cell to world coordinates to compare with drone_x/drone_y
        cell_world_x = (col + 0.5) * self.scenario.cell_size_x - (self.scenario.world_size_x / 2.0)
        cell_world_y = (row + 0.5) * self.scenario.cell_size_y - (self.scenario.world_size_y / 2.0)
        
        dx = cell_world_x - drone_x
        dy = cell_world_y - drone_y
        cell_angle = math.atan2(dy, dx)
        angle_diff = abs(drone_heading - cell_angle)
        angle_factor = 0.88 + 0.24 * math.cos(angle_diff)
        distance_factor = 0.92 + 0.16 * math.sin(tick * 0.15 + drone_id)
        vantage_factor = angle_factor * distance_factor

        # Take the MAX reading in the 3x3 field of view.
        # This prevents a single-cell POI from being diluted by empty surrounding cells.
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                reading = self.get_sensor_at(col + dc, row + dr, tick)
                if reading is None:
                    continue
                thermal = max(thermal, reading['thermal'])
                audio = max(audio, reading['audio'])
                camera = max(camera, reading['camera'])
                gas = max(gas, reading['gas'])
                count += 1

        if count == 0:
            return {'thermal': 0, 'audio': 0, 'camera': 0, 'gas': 0, 'vantage_factor': 1.0}

        return {
            'thermal': float(np.clip(thermal * vantage_factor, 0.0, 1.0)),
            'audio':   0.0,  # Disabled
            'camera':  float(np.clip(camera * vantage_factor, 0.0, 1.0)),
            'gas':     0.0,  # Disabled
            'vantage_factor': vantage_factor,
        }

    def find_source_cell(self, col: int, row: int) -> Tuple[int, int]:
        """
        Scan the 3x3 neighborhood to find the actual SURVIVOR cell.
        Returns the exact (col, row) of the survivor, or the drone's own cell if none found.
        This prevents POI duplication when detecting from adjacent cells.
        """
        from .scenario_loader import CELL_SURVIVOR
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                nc, nr = col + dc, row + dr
                cell_type = self.scenario.get_cell_type(nc, nr)
                if cell_type == CELL_SURVIVOR:
                    return (nc, nr)
        return (col, row)


# ── Confidence / Uncertainty / Viscosity computations (from uncertainty.js) ──

SPREAD_MAX = 0.30
SOLIDIFY_MIN = 0.75


def compute_confidence(readings: Dict[str, float]) -> float:
    """
    Multi-modal sensor fusion confidence.
    Requires BOTH Thermal AND Camera to be high (the unique SURVIVOR signature).
    Only these two sensors are used.
    """
    t = readings.get('thermal', 0)
    c = readings.get('camera', 0)

    # STRICT CROSS-MODAL GATE: Both thermal AND camera must independently exceed 0.5
    if t < 0.50 or c < 0.50:
        return float(np.clip(max(t, c) * 0.30, 0.0, 0.40))

    # Above the gate: genuine cross-modal agreement
    base = 0.50 * t + 0.50 * c
    agreement_bonus = 0.15

    return float(np.clip(base + agreement_bonus, 0.0, 1.0))


def compute_uncertainty(readings: Dict[str, float], prev_uncertainty: float) -> float:
    """
    Bayesian-inspired uncertainty reduction.
    Uncertainty drops when readings are consistent and informative.
    """
    values = [readings.get('thermal', 0), readings.get('camera', 0)]
    mean_signal = np.mean(values)
    std_signal = np.std(values)

    # High mean + low variance = informative reading = reduce uncertainty
    info_gain = mean_signal * (1.0 - std_signal)
    new_uncertainty = prev_uncertainty * (1.0 - 0.40 * info_gain)

    return float(np.clip(new_uncertainty, 0.02, 1.0))


def compute_viscosity(confidence: float, uncertainty: float) -> float:
    """
    Viscosity = how 'stuck' a drone is. High viscosity = slowing down near target.
    """
    return float(np.clip(confidence * (1.0 - uncertainty * 0.2), 0.0, 1.0))


def classify_regime(viscosity: float) -> str:
    """Classify the behavioral regime based on viscosity."""
    if viscosity >= SOLIDIFY_MIN:
        return 'SOLIDIFY'
    elif viscosity <= SPREAD_MAX:
        return 'SPREAD'
    else:
        return 'CONVERGE'

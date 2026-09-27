"""
drone_agent.py — Individual Drone Agent (Port of drone.js)

Each drone is a fully independent ROS 2 node. It keeps its own local state and
only receives peer information through ROS topics. Sensor/world-model isolation
depends on SensorFusion/DisasterScenario; this file does not expose peer or
GCS state directly to the agent.

Coordination is stigmergic plus explicit low-bandwidth mission messages: drones
share pheromone updates, observations, POI claims, verification, and GCS relay
information through ROS topics.

Movement rules vary by viscosity regime:
  SPREAD    → high randomness, weak gradient pull (exploration)
  CONVERGE  → moderate gradient pull, reduced randomness (re-verify)
  SOLIDIFY  → strong gradient pull, minimal randomness (commit)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import numpy as np
from typing import Dict, Optional, Tuple

from .px4_bridge import PX4Bridge
from .sensor_fusion import (
    SensorFusion,
    compute_confidence,
    compute_uncertainty,
    compute_viscosity,
    classify_regime,
)
from .pheromone_field import PheromoneField
from .scenario_loader import DisasterScenario

from std_msgs.msg import String
import json
import base64


# ── Regime movement parameters (from drone.js) ───────────────────────────────
# Speeds are converted: JS grid-units/tick × cell_size_m × tick_rate ≈ m/s

REGIME_PARAMS = {
    'SPREAD': {
        'gradient_pull': 0.05,
        'random_weight': 0.80,
        'speed': 4.0,           # m/s — cruising speed (below max for smoother flight)
        'deposit_amount': 0.03,
        'inertia': 0.95,        # Very high inertia = smooth, gradual turns
    },
    'CONVERGE': {
        'gradient_pull': 0.80,
        'random_weight': 0.20,
        'speed': 3.0,           # m/s
        'deposit_amount': 0.12,
        'inertia': 0.93,
    },
    'SOLIDIFY': {
        'gradient_pull': 0.95,
        'random_weight': 0.05,
        'speed': 1.0,           # m/s
        'deposit_amount': 0.20,
        'inertia': 0.90,
    },
    'RESCUED': {
        'gradient_pull': 0.0,
        'random_weight': 0.0,
        'speed': 5.0,           # m/s — return speed
        'deposit_amount': 0.0,
        'inertia': 0.90,
    },
}

# NATO phonetic callsigns
CALLSIGNS = [
    'ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO',
    'FOXTROT', 'GOLF', 'HOTEL', 'INDIA', 'JULIET',
]


class DroneAgent(Node):
    """
    Autonomous drone agent running as an independent ROS 2 node.
    Implements the full sense → decide → move → deposit cycle from drone.js.
    """

    def __init__(self, drone_id: int, namespace: str = ''):
        super().__init__(f'drone_agent_{drone_id}')
        self.drone_id = drone_id
        self.callsign = CALLSIGNS[drone_id % len(CALLSIGNS)]

        # ── Parameters ──
        self.declare_parameter('altitude_m', 30.0)
        self.declare_parameter('world_size_x', 1000.0)
        self.declare_parameter('world_size_y', 1000.0)
        self.declare_parameter('grid_cols', 50)
        self.declare_parameter('grid_rows', 50)
        self.declare_parameter('scenario', 'grand-challenge')
        self.declare_parameter('tick_rate_hz', 20.0)
        self.declare_parameter('comm_range_m', 100.0)
        self.declare_parameter('battery_capacity_s', 1200.0)
        self.declare_parameter('gcs_x', -575.0)
        self.declare_parameter('gcs_y', 0.0)
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', 0.0)

        # Shared long-term coverage memory. This is a swarm-wide control-plane
        # state, not a decaying pheromone.
        self.declare_parameter('coverage_mark_radius_cells', 0)
        self.declare_parameter('coverage_sync_interval_ticks', 200)
        self.declare_parameter('coverage_target_refresh_ticks', 40)

        # Verification timing. A request gets a short ACK window; once ACKed,
        # the sentinel waits for the verifier heartbeat/mission to finish.
        self.declare_parameter('verify_accept_timeout_ticks', 300)
        self.declare_parameter('verifier_heartbeat_timeout_ticks', 120)

        # Global relay election. The control plane elects exactly one relay at
        # a time, even when search groups are geographically separated.
        self.declare_parameter('relay_election_window_ticks', 20)
        self.declare_parameter('relay_heartbeat_interval_ticks', 20)
        self.declare_parameter('relay_lock_timeout_ticks', 120)

        self.altitude = self.get_parameter('altitude_m').value
        world_x = self.get_parameter('world_size_x').value
        world_y = self.get_parameter('world_size_y').value
        cols = self.get_parameter('grid_cols').value
        rows = self.get_parameter('grid_rows').value
        scenario_name = self.get_parameter('scenario').value
        self.tick_rate_hz = self.get_parameter('tick_rate_hz').value
        self.comm_range_m = self.get_parameter('comm_range_m').value
        self.coverage_mark_radius_cells = max(0, int(self.get_parameter('coverage_mark_radius_cells').value))
        self.coverage_sync_interval_ticks = max(20, int(self.get_parameter('coverage_sync_interval_ticks').value))
        self.coverage_target_refresh_ticks = max(5, int(self.get_parameter('coverage_target_refresh_ticks').value))
        self.verify_accept_timeout_ticks = max(20, int(self.get_parameter('verify_accept_timeout_ticks').value))
        self.verifier_heartbeat_timeout_ticks = max(20, int(self.get_parameter('verifier_heartbeat_timeout_ticks').value))
        self.relay_election_window_ticks = max(5, int(self.get_parameter('relay_election_window_ticks').value))
        self.relay_heartbeat_interval_ticks = max(5, int(self.get_parameter('relay_heartbeat_interval_ticks').value))
        self.relay_lock_timeout_ticks = max(20, int(self.get_parameter('relay_lock_timeout_ticks').value))

        # Battery & Return-to-base state
        self.battery_capacity_s = self.get_parameter('battery_capacity_s').value
        self.battery_remaining_s = self.battery_capacity_s
        self.home_position = (
            self.get_parameter('gcs_x').value,
            self.get_parameter('gcs_y').value
        )
        self.is_returning_home = False
        self.has_landed = False

        self.spawn_offset_x = self.get_parameter('spawn_x').value
        self.spawn_offset_y = self.get_parameter('spawn_y').value

        # ── Core modules ──
        self.scenario = DisasterScenario(cols, rows, world_x, world_y, scenario_name)
        self.sensors = SensorFusion(self.scenario)
        self.field = PheromoneField(cols, rows)
        self.px4 = PX4Bridge(self, drone_id, namespace)

        # ── Agent state ──
        self.regime = 'SPREAD'
        self.viscosity = 0.0
        self.confidence = 0.0
        self.uncertainty = 1.0
        self.last_readings = {'thermal': 0, 'audio': 0, 'camera': 0, 'gas': 0}

        # Velocity vector
        self.vx = 0.0
        self.vy = 0.0
        self._heading_bias = np.random.uniform(0, 2 * math.pi)
        self._heading_angle = self._heading_bias

        # Sentinel state (drone that FOUND the survivor)
        self.is_sentinel = False
        self.sentinel_target = None
        self.sentinel_cell = None     # Locked (col, row) to prevent drift
        self.sentinel_world = None    # Locked (x, y) global position
        self.sentinel_ticks = 0
        self.sentinel_verified = False  # Has another drone confirmed?

        # Verifier state (drone flying TO a sentinel to verify)
        self.is_verifier = False
        self.verify_target_cell = None   # (col, row) of the POI to verify
        self.verify_target_world = None  # (x, y) global of the POI
        self.verify_requester_id = None  # drone_id of the sentinel
        self.verified_pois = set()       # POIs we've already verified/attempted
        
        # Abandoned POIs (timed out as sentinel). Used to recruit others later.
        self.unverified_pois = {}

        # Neighbor tracking for collision avoidance
        self.neighbor_positions: Dict[int, dict] = {}

        # Curiosity role: 20% explorers, 80% convergers
        self.curiosity_role = 'EXPLORER' if (drone_id % 5 == 0) else 'CONVERGER'

        # Tick counter
        self.tick = 0
        self._is_airborne = False

        # ── POI Intel & GCS Relay ──
        # Each drone maintains its own list of known POIs (found by self or heard from others)
        self.known_pois: Dict[str, dict] = {}  # key: "col_row", val: {col, row, x, y, finder, tick}
        # POIs that have been successfully relayed to GCS
        self.gcs_relayed_pois: set = set()
        # GCS relay mission state
        self.is_relay_mission = False
        self.relay_phase = 'FLYING_TO_GCS'  # FLYING_TO_GCS → DUMPING → RETURNING
        self.relay_dump_ticks = 0
        self.relay_resume_position: Optional[Tuple[float, float]] = None
        self.relay_claim_tick = 0
        self.relay_proposals: Dict[int, dict] = {}
        self.relay_proposal_tick = 0
        self.relay_proposal_sent = False
        self.active_relay_id: Optional[int] = None
        self.active_relay_tick = 0
        self.active_relay_lock_id = None

        # Global verification claims. A verifier acquires a short-lived global
        # lock before starting so only one verifier can work a POI at a time.
        self.verification_claims: Dict[str, dict] = {}
        self.pending_verify_request = None
        self.pending_verify_until_tick = 0
        self.verify_lock_settle_ticks = 5

        # ── Shared permanent coverage memory ──
        self.coverage_rows = rows
        self.coverage_cols = cols
        self.exploration_grid = np.zeros((rows, cols), dtype=np.bool_)
        self.coverage_seen_messages = set()
        self.coverage_target_cell = None
        self.coverage_target_tick = 0

        # Global peer state is separate from range-limited radio neighbors. It
        # is used only for swarm-wide arbitration and coverage synchronization.
        self.global_peer_states: Dict[int, dict] = {}
        self.control_seen = set()

        # Verification request/claim state. A sentinel waits for an ACK before
        # recruiting another drone, and retries if the request was lost.
        self.verify_request_in_flight = False
        self.verify_request_id = None
        self.verify_request_target_id = None
        self.verify_request_deadline_tick = 0
        self.verify_request_attempts = 0
        self.verify_started_tick = 0
        self.verify_rejected = False
        self.sentinel_recruit_interval_ticks = 200
        self.verifier_timeout_ticks = 600
        self.sentinel_timeout_ticks = 600
        self.verify_request_last_sent_tick = -10**9
        self.verifier_last_heartbeat_tick = 0
        self.pending_outgoing_verification: Dict[str, str] = {}

        # Lightweight negative-memory for a failed verification. It prevents a
        # single drone from immediately re-locking the same false positive,
        # while still allowing a genuinely new detection later.
        self.rejected_pois: Dict[str, int] = {}

        # Distributed message de-duplication. These are intentionally separate
        # from POI state so a message can be forwarded without changing meaning.
        self.seen_confirmations = set()
        self.seen_rejections = set()
        self.seen_clears = set()
        self.relay_claims: Dict[int, dict] = {}
        self.unverified_request_cooldowns: Dict[str, int] = {}

        # ── Communication (radio simulation via ROS topics) ──
        # Mission-critical messages (verification, POI confirmation, relay claims)
        # must not use BEST_EFFORT.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_radio = self.create_publisher(String, '/swarm/radio', qos)
        self.sub_radio = self.create_subscription(String, '/swarm/radio', self._on_radio, qos)

        # Global control plane: reliable, low-bandwidth coordination for
        # coverage synchronization and mission locks/elections. Physical radio
        # range is still enforced on /swarm/radio.
        self.pub_control = self.create_publisher(String, '/swarm/control', qos)
        self.sub_control = self.create_subscription(String, '/swarm/control', self._on_control, qos)

        # Track which survivor-found POIs we've already forwarded to prevent loops.
        self.relayed_pois = set()

        # ── Pheromone field sync (shared via topic) ──
        self.pub_deposit = self.create_publisher(String, '/swarm/deposits', qos)
        self.sub_deposit = self.create_subscription(String, '/swarm/deposits', self._on_deposit, qos)

        # ── Main loop timer ──
        self.timer = self.create_timer(1.0 / self.tick_rate_hz, self._tick_callback)

        self.get_logger().info(
            f'Drone {self.callsign} (ID={drone_id}) initialized | '
            f'Role: {self.curiosity_role} | Namespace: {namespace}'
        )

    # ── Main tick loop ────────────────────────────────────────────────────────

    def _tick_callback(self):
        """Main simulation loop: takeoff → sense → decide → move → deposit."""

        # Phase 0: Takeoff sequence
        if not self._is_airborne:
            reached = self.px4.takeoff_sequence(self.altitude)
            if reached:
                self._is_airborne = True
                self.get_logger().info(f'[{self.callsign}] Airborne at {self.altitude}m!')
            return

        # Must keep sending heartbeats to stay in offboard mode
        self.px4.send_heartbeat()

        if self.has_landed:
            return

        self.tick += 1
        local_pos = self.px4.get_position_enu()
        if local_pos is None:
            return

        # Convert PX4 local position (relative to spawn) to Gazebo global position
        x = local_pos[0] + self.spawn_offset_x
        y = local_pos[1] + self.spawn_offset_y
        z = local_pos[2]

        # Convert world position to grid cell
        col, row = self.scenario.world_to_grid(x, y)

        # Permanently remember explored cells and periodically synchronize the
        # full compact bitset so late-joining/reconnected drones catch up.
        self._mark_covered_cell(col, row, broadcast=True)
        if self.tick == 1 or self.tick % self.coverage_sync_interval_ticks == 0:
            self._broadcast_coverage_sync()

        # Keep swarm-wide status fresh for deterministic relay election.
        if self.tick % 20 == 0:
            self._broadcast_agent_status(x, y)
        self._expire_coordination_state()
        self._process_pending_verification()

        # ── Battery Logic ──
        self.battery_remaining_s = max(0.0, self.battery_remaining_s - (1.0 / self.tick_rate_hz))
        if self.battery_capacity_s > 0 and self.battery_remaining_s < self.battery_capacity_s * 0.20 and not self.is_returning_home:
            self._enter_return_to_base(x, y)

        if self.is_returning_home:
            dx = self.home_position[0] - x
            dy = self.home_position[1] - y
            dist = math.hypot(dx, dy)
            if dist < 5.0:
                self.get_logger().info(f'[{self.callsign}] 🛬 Arrived at base, landing...')
                self.px4.land()
                self.has_landed = True
                self.vx = 0.0
                self.vy = 0.0
                return

            # Fly home directly, but still keep inter-drone separation.
            speed = REGIME_PARAMS['RESCUED']['speed']
            desired_vx = (dx / max(dist, 0.1)) * speed
            desired_vy = (dy / max(dist, 0.1)) * speed
            self.vx, self.vy = self._safe_velocity(
                desired_vx, desired_vy, x, y, speed
            )
            self.px4.send_velocity(self.vy, self.vx, 0.0)  # PX4 takes NED (North=vy, East=vx)

            if self.tick % 100 == 0:
                self.get_logger().info(f'[{self.callsign}] 🔋 Returning home... {int(dist)}m to go.')
            return

        # ── GCS RELAY MISSION ──
        # If this drone has volunteered to relay intel back to GCS
        if self.is_relay_mission:
            self._execute_relay_mission(x, y, col, row, local_pos)
            # Still broadcast observations so dashboard tracks us
            if self.tick % 5 == 0:
                self._broadcast_observation(col, row)
            return

        # ── CHECK: Should we participate in a GCS relay election? ──
        # Only confirmed, not-yet-relayed information creates a proposal. A
        # global two-phase election guarantees at most one active relay.
        unrelayed_confirmed = [
            k for k, v in self.known_pois.items()
            if v.get('confirmed', False) and k not in self.gcs_relayed_pois
        ]
        self._maybe_start_relay_election(unrelayed_confirmed)

        # ── SKIP KNOWN POIs ──
        # If we are on a cell that already has a found survivor, don't re-detect it
        cell_key = f"{col}_{row}"
        current_poi = self.known_pois.get(cell_key)
        if current_poi and current_poi.get('confirmed', False):
            # A confirmed POI is already handled; do not waste search effort.
            self.regime = 'SPREAD'
            self.confidence = 0.0
            self.viscosity = 0.0
            self.uncertainty = 1.0
            # Fall through to MOVE below (skip SENSE/DECIDE/SENTINEL)
        else:
            # ── SENSE ──
            readings = self.sensors.get_local_readings(
                col, row, self.tick,
                drone_heading=self.px4.heading,
                drone_x=x, drone_y=y,
                drone_id=self.drone_id,
            )
            self.last_readings = readings

            # ── DECIDE ──
            self.confidence = compute_confidence(readings)
            self.uncertainty = compute_uncertainty(readings, self.uncertainty)
            self.viscosity = compute_viscosity(self.confidence, self.uncertainty)
            self.regime = classify_regime(self.viscosity)

        params = REGIME_PARAMS[self.regime]

        # ── SENTINEL MODE: hold position, wait for verification ──
        if self.is_sentinel:
            self.sentinel_ticks += 1
            local_sx, local_sy = self.sentinel_target
            s_col, s_row = self.sentinel_cell
            s_x, s_y = self.sentinel_world

            # Hold position
            self.px4.send_heartbeat(use_position=True)
            self.px4.send_position(
                x=local_sy, y=local_sx, z=-self.altitude
            )

            # Keep broadcasting SURVIVOR_FOUND to attract a verifier.
            if self.sentinel_ticks % 40 == 0:
                self._broadcast_survivor_found(s_col, s_row, s_x, s_y)

            # Recruit exactly one verifier at a time. Once a peer ACKs, the
            # sentinel waits on that verifier's heartbeat instead of starting a
            # second verifier after an arbitrary 10-second timer.
            if self.verify_request_in_flight:
                deadline_expired = self.sentinel_ticks >= self.verify_request_deadline_tick
                heartbeat_expired = (
                    self.verify_request_target_id is not None
                    and self.verifier_last_heartbeat_tick > 0
                    and self.tick - self.verifier_last_heartbeat_tick > self.verifier_heartbeat_timeout_ticks
                )
                if deadline_expired or heartbeat_expired:
                    self.get_logger().warn(
                        f'[{self.callsign}] ⚠️ Verifier {self.verify_request_target_id} '
                        f'lost/no heartbeat; releasing verification lock.'
                    )
                    poi_key = f"{s_col}_{s_row}"
                    self._broadcast_verify_lock_release(poi_key, reason='timeout')
                    self.verify_request_in_flight = False
                    self.verify_request_target_id = None
                    self.verify_request_id = None
                    self.verifier_last_heartbeat_tick = 0

            # If no verifier is currently assigned, recruit the single closest
            # fresh SPREAD drone within actual radio range. For abandoned POIs,
            # the same logic is reached from OBSERVATION handling below.
            if (not self.verify_request_in_flight
                    and self.sentinel_ticks % self.sentinel_recruit_interval_ticks == 0):
                closest_dist = float('inf')
                best_candidate = None
                my_pos = self.px4.get_position_enu()
                gx = my_pos[0] + self.spawn_offset_x if my_pos else s_x
                gy = my_pos[1] + self.spawn_offset_y if my_pos else s_y
                for nid, ndata in self.neighbor_positions.items():
                    age = self.tick - ndata.get('tick', 0)
                    if nid == self.drone_id or age > 100 or ndata.get('regime') != 'SPREAD':
                        continue
                    if ndata.get('is_returning_home', False):
                        continue
                    dist = math.hypot(gx - ndata.get('x', 0.0), gy - ndata.get('y', 0.0))
                    if dist > self.comm_range_m:
                        continue
                    # Do not recruit someone globally locked to another POI.
                    if self._peer_has_active_verification(nid):
                        continue
                    if dist < closest_dist:
                        closest_dist = dist
                        best_candidate = nid

                if best_candidate is not None:
                    request_id = f"{self.drone_id}:{s_col}:{s_row}:{self.sentinel_ticks}"
                    self._broadcast_verify_request(
                        s_col, s_row, s_x, s_y,
                        target_drone=best_candidate,
                        request_id=request_id,
                    )
                    self.verify_request_in_flight = True
                    self.verify_request_id = request_id
                    self.verify_request_target_id = best_candidate
                    # Initial ACK window. Once accepted, this is replaced by
                    # the full verifier timeout.
                    self.verify_request_deadline_tick = (
                        self.sentinel_ticks + self.verify_accept_timeout_ticks
                    )
                    self.verify_request_attempts += 1
                    self.verifier_last_heartbeat_tick = 0
                    self.get_logger().info(
                        f'[{self.callsign}] 🎯 Recruiting drone {best_candidate} '
                        f'for verification ({int(closest_dist)}m away)'
                    )

            if self.tick % 5 == 0:
                self._broadcast_observation(s_col, s_row)

            # Pheromone beacon
            self.field.deposit(s_col, s_row, 0.30,
                               drone_id=self.drone_id,
                               confidence=self.confidence,
                               uncertainty=self.uncertainty)
            self._publish_deposit(s_col, s_row, 0.30)

            # Logging
            if self.tick % 100 == 0:
                status = '✅ VERIFIED' if self.sentinel_verified else '⏳ AWAITING VERIFICATION'
                self.get_logger().info(
                    f'[{self.callsign}] 🔒 SENTINEL {status} at cell=({s_col},{s_row}) '
                    f'ticks={self.sentinel_ticks}'
                )

            # ── RELEASE CONDITIONS ──
            # 1. Another drone verified us → CONFIRMED, resume
            if self.sentinel_verified:
                poi_key = f"{s_col}_{s_row}"
                self.get_logger().info(
                    f'[{self.callsign}] ✅ POI CONFIRMED by peer! cell=({s_col},{s_row}) — Resuming SPREAD.'
                )
                # Mark as confirmed in known_pois
                if poi_key in self.known_pois:
                    self.known_pois[poi_key]['confirmed'] = True
                self._release_sentinel(s_col, s_row)
                return

            # 2. Another drone rejected the candidate. Release and keep a short
            # negative-memory so this same agent does not immediately lock it again.
            if self.verify_rejected:
                poi_key = f"{s_col}_{s_row}"
                self.rejected_pois[poi_key] = self.tick + 300
                self.known_pois.pop(poi_key, None)
                self.unverified_pois.pop(poi_key, None)
                self._release_sentinel(s_col, s_row)
                return

            # 3. Timeout after 300 ticks (15s) — no successful verification,
            # release but keep the candidate available for later recruitment.
            if self.sentinel_ticks > self.sentinel_timeout_ticks:
                self.get_logger().warn(
                    f'[{self.callsign}] ⚠️ SENTINEL TIMEOUT — No verifier arrived. '
                    f'Marking cell=({s_col},{s_row}) as unverified to recruit later.'
                )
                poi_key = f"{s_col}_{s_row}"
                self.unverified_pois[poi_key] = {
                    'col': s_col, 'row': s_row, 'x': s_x, 'y': s_y,
                    'finder_id': self.drone_id,
                }
                if poi_key in self.known_pois:
                    self.known_pois[poi_key]['confirmed'] = False
                    self.known_pois[poi_key]['status'] = 'candidate'
                self._release_sentinel(s_col, s_row)

            return

        # ── VERIFIER MODE: fly to a sentinel's POI to verify ──
        if self.is_verifier:
            self.verify_started_tick = self.verify_started_tick or self.tick
            if self.tick % 20 == 0:
                self._broadcast_verify_heartbeat()
            if self.tick - self.verify_started_tick > self.verifier_timeout_ticks:
                self.get_logger().warn(
                    f'[{self.callsign}] ⚠️ VERIFIER TIMEOUT — abandoning stale verification.'
                )
                self._finish_verifier(reset_heading=True, reason='timeout')
                if self.tick % 5 == 0:
                    self._broadcast_observation(col, row)
                return

            vx_t, vy_t = self.verify_target_world
            dx = vx_t - x
            dy = vy_t - y
            dist = math.hypot(dx, dy)

            # Fly toward the POI, but preserve inter-drone separation.
            if dist > 25.0:
                speed = REGIME_PARAMS['CONVERGE']['speed']
                desired_vx = (dx / max(dist, 0.1)) * speed
                desired_vy = (dy / max(dist, 0.1)) * speed
                self.vx, self.vy = self._safe_velocity(
                    desired_vx, desired_vy, x, y, speed
                )
                self.px4.send_velocity(self.vy, self.vx, 0.0)
                if self.tick % 60 == 0:
                    self.get_logger().info(
                        f'[{self.callsign}] 🔍 VERIFYING: Flying to POI... {int(dist)}m away'
                    )
            else:
                # We are close enough — run our own sensors.
                readings = self.sensors.get_local_readings(
                    col, row, self.tick,
                    drone_heading=self.px4.heading,
                    drone_x=x, drone_y=y,
                    drone_id=self.drone_id,
                )
                self.last_readings = readings
                conf = compute_confidence(readings)
                poi_key = f"{self.verify_target_cell[0]}_{self.verify_target_cell[1]}"

                if conf > 0.60:
                    # The verifier independently sees the same POI. Broadcast the
                    # actual POI coordinates, not the verifier's current position.
                    self._broadcast_verify_confirm(
                        self.verify_target_cell[0], self.verify_target_cell[1],
                        self.verify_requester_id,
                        poi_x=vx_t, poi_y=vy_t,
                        request_id=self.verify_request_id,
                    )
                    self.verified_pois.add(poi_key)
                    self._mark_poi_confirmed(
                        poi_key, self.verify_target_cell[0], self.verify_target_cell[1],
                        vx_t, vy_t,
                        finder=self.known_pois.get(poi_key, {}).get('finder', '??'),
                        finder_id=self.known_pois.get(poi_key, {}).get('finder_id'),
                    )
                    self.get_logger().info(
                        f'[{self.callsign}] ✅ VERIFIED! POI at cell='
                        f'({self.verify_target_cell[0]},{self.verify_target_cell[1]}) '
                        f'conf={conf:.2f} — Confirmed!'
                    )
                else:
                    self._broadcast_verify_reject(
                        self.verify_target_cell[0], self.verify_target_cell[1],
                        self.verify_requester_id,
                        poi_x=vx_t, poi_y=vy_t,
                        request_id=self.verify_request_id,
                    )
                    self._mark_poi_rejected(poi_key)
                    self.get_logger().info(
                        f'[{self.callsign}] ❌ VERIFY FAILED — conf={conf:.2f} too low. '
                        f'False positive at ({self.verify_target_cell[0]},{self.verify_target_cell[1]})'
                    )

                self._finish_verifier(reset_heading=True)

            if self.tick % 5 == 0:
                self._broadcast_observation(col, row)
            return

        # ── CHECK: Enter sentinel mode on SOLIDIFY ──
        if self.regime == 'SOLIDIFY':
            source_col, source_row = self.sensors.find_source_cell(col, row)
            poi_key = f"{source_col}_{source_row}"

            # DON'T create a second sentinel for an already-known candidate or
            # confirmed POI. Rejected POIs are temporarily ignored by this drone.
            rejection_expiry = self.rejected_pois.get(poi_key)
            if rejection_expiry is not None:
                if self.tick < rejection_expiry:
                    self.regime = 'SPREAD'
                    self.confidence = 0.0
                    self.viscosity = 0.0
                    self.uncertainty = 1.0
                    return
                self.rejected_pois.pop(poi_key, None)

            existing_poi = self.known_pois.get(poi_key)
            if existing_poi is not None:
                self.get_logger().info(
                    f'[{self.callsign}] ✉️ Skipping SOLIDIFY — POI ({source_col},{source_row}) '
                    f'already known ({existing_poi.get("status", "candidate")}) '
                    f'from {existing_poi.get("finder", "??")}.'
                )
                self.regime = 'SPREAD'
                self.confidence = 0.0
                self.viscosity = 0.0
                self.uncertainty = 1.0
            else:
                self.is_sentinel = True
                self.sentinel_verified = False
                self.verify_rejected = False
                self.verify_request_in_flight = False
                self.verify_request_id = None
                self.verify_request_target_id = None
                self.verify_request_deadline_tick = 0
                self.verify_request_attempts = 0
                self.sentinel_target = (local_pos[0], local_pos[1])
                source_wx, source_wy = self.scenario.grid_to_world(source_col, source_row)
                self.sentinel_cell = (source_col, source_row)
                self.sentinel_world = (source_wx, source_wy)
                self.sentinel_ticks = 0

                self.known_pois[poi_key] = {
                    'col': source_col, 'row': source_row,
                    'x': source_wx, 'y': source_wy,
                    'finder': self.callsign, 'finder_id': self.drone_id,
                    'tick': self.tick, 'confirmed': False, 'status': 'candidate',
                }
                self.get_logger().info(
                    f'[{self.callsign}] ===== SURVIVOR FOUND at ({source_wx:.1f},{source_wy:.1f}) '
                    f'cell=({source_col},{source_row}) conf={self.confidence:.2f} '
                    f'— AWAITING VERIFICATION ====='
                )
                self._broadcast_survivor_found(source_col, source_row, source_wx, source_wy)
                return

        # ── MOVE (normal exploration) ──
        # Pheromone gradient pull
        grad = self.field.get_gradient(col, row)
        grad_mag = math.hypot(grad[0], grad[1])
        if grad_mag > 0.001:
            grad_nx = grad[0] / grad_mag
            grad_ny = grad[1] / grad_mag
        else:
            grad_nx = 0.0
            grad_ny = 0.0

        # Uncertainty gradient (short-term exploration drive)
        u_grad = self.field.get_uncertainty_gradient(col, row)
        u_mag = math.hypot(u_grad[0], u_grad[1])
        if u_mag > 0.001:
            u_nx = u_grad[0] / u_mag
            u_ny = u_grad[1] / u_mag
        else:
            u_nx = 0.0
            u_ny = 0.0

        # Permanent Exploration gradient (long-term swarm-wide coverage).
        e_nx, e_ny = self._get_coverage_gradient(col, row, x, y)

        # Random walk component — pick a new heading every ~2 seconds (40 ticks at 20Hz)
        # More frequent heading changes = better coverage
        if not hasattr(self, '_target_rand_angle') or self.tick % 40 == 0:
            # Larger deviation = more exploration spread
            self._target_rand_angle = self._heading_bias + np.random.normal(0, 0.6)

        rand_x = math.cos(self._target_rand_angle)
        rand_y = math.sin(self._target_rand_angle)

        # ── KNOWN POI REPULSION ──
        # Push drones AWAY from areas where survivors have already been found.
        # This forces spread into unexplored regions instead of circling cleared areas.
        repel_x, repel_y = 0.0, 0.0
        if self.regime == 'SPREAD' and self.known_pois:
            for poi_key, poi_data in self.known_pois.items():
                if not poi_data.get('confirmed', False):
                    continue
                poi_col = poi_data.get('col', 0)
                poi_row = poi_data.get('row', 0)
                dx = col - poi_col
                dy = row - poi_row
                dist = math.hypot(dx, dy)
                if 0.1 < dist < 8.0:  # Repel within ~160m (8 cells × 20m)
                    force = 1.0 / (dist + 0.5)
                    repel_x += (dx / dist) * force
                    repel_y += (dy / dist) * force
            # Normalize
            rep_mag = math.hypot(repel_x, repel_y)
            if rep_mag > 0.001:
                repel_x /= rep_mag
                repel_y /= rep_mag

        repel_weight = 0.40 if self.regime == 'SPREAD' else 0.0

        # Blend: gradient pull + random walk + short-term exploration + long-term coverage + POI repulsion
        gp = params['gradient_pull']
        rw = params['random_weight']
        explore_weight = 0.15 if self.curiosity_role == 'EXPLORER' else 0.05
        # Add the long-term coverage weight (only when spreading)
        coverage_weight = 0.55 if self.regime == 'SPREAD' else 0.0

        target_vx = (gp * grad_nx + rw * rand_x + explore_weight * u_nx + coverage_weight * e_nx + repel_weight * repel_x)
        target_vy = (gp * grad_ny + rw * rand_y + explore_weight * u_ny + coverage_weight * e_ny + repel_weight * repel_y)

        # Dynamically brake based on viscosity. High viscosity = slow down to investigate.
        # This prevents the drone from accidentally flying past a POI before it can verify it.
        viscosity_braking = max(0.1, 1.0 - (self.viscosity ** 1.5))
        current_speed = params['speed'] * viscosity_braking

        # Normalize and scale to regime speed
        mag = math.hypot(target_vx, target_vy)
        if mag > 0.001:
            target_vx = (target_vx / mag) * current_speed
            target_vy = (target_vy / mag) * current_speed

        # Apply inertia (smooth velocity transitions)
        inertia = params['inertia']
        self.vx = inertia * self.vx + (1 - inertia) * target_vx
        self.vy = inertia * self.vy + (1 - inertia) * target_vy

        # ── COLLISION AVOIDANCE ──
        self.vx, self.vy = self._safe_velocity(
            self.vx, self.vy, x, y, params['speed'] * 1.5
        )

        # ── BOUNDARY ENFORCEMENT (soft turn-back) ──
        # Do not enforce boundaries if returning to the GCS outside the arena
        if self.regime != 'RESCUED':
            half_x = self.scenario.world_size_x / 2.0
            half_y = self.scenario.world_size_y / 2.0
            margin = 100.0       # Start turning back 100m from edge (much earlier)
            hard_limit = 0.0     # Hard override at the exact boundary line

            # If past hard limit, override velocity to return aggressively
            if abs(x) > half_x + hard_limit or abs(y) > half_y + hard_limit:
                return_speed = params['speed'] * 1.5 # Boost speed to get back in
                # Only push them towards the center on the axis they violated
                if abs(x) > half_x + hard_limit:
                    self.vx = (-x / max(abs(x), 0.1)) * return_speed
                if abs(y) > half_y + hard_limit:
                    self.vy = (-y / max(abs(y), 0.1)) * return_speed
            else:
                # Gentle repulsion near edges — proportional to how close to edge
                repulse = 4.0 # Double the repulsion strength
                if x > half_x - margin:
                    self.vx -= repulse * ((x - (half_x - margin)) / margin)
                elif x < -half_x + margin:
                    self.vx += repulse * ((-half_x + margin - x) / margin)
                if y > half_y - margin:
                    self.vy -= repulse * ((y - (half_y - margin)) / margin)
                elif y < -half_y + margin:
                    self.vy += repulse * ((-half_y + margin - y) / margin)

        # Hard clamp velocity
        max_speed = params['speed'] * 1.5
        speed = math.hypot(self.vx, self.vy)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed

        # Update heading bias for next tick — respond faster to new direction
        if math.hypot(self.vx, self.vy) > 0.1:
            self._heading_angle = math.atan2(self.vy, self.vx)
            self._heading_bias = 0.7 * self._heading_bias + 0.3 * self._heading_angle

        # Altitude hold: gently correct back to target altitude
        alt_error = self.altitude - z
        vz_cmd = np.clip(alt_error * 0.5, -1.0, 1.0)

        # Convert ENU velocity to NED for PX4 (swap x↔y, negate z)
        self.px4.send_velocity(
            vx=self.vy,      # NED North = ENU East
            vy=self.vx,      # NED East  = ENU y
            vz=-vz_cmd,      # NED Down  = negate
        )

        # ── DEPOSIT ──
        deposit_amount = params['deposit_amount']
        if deposit_amount > 0:
            self.field.deposit(
                col, row, deposit_amount,
                drone_id=self.drone_id,
                confidence=self.confidence,
                uncertainty=self.uncertainty,
            )
            # Broadcast deposit to other drones
            self._publish_deposit(col, row, deposit_amount)

        # ── BROADCAST observation via radio ──
        if self.tick % 5 == 0:  # Throttle radio to every 5 ticks
            self._broadcast_observation(col, row)

        # Periodic logging
        if self.tick % 100 == 0:
            r = self.last_readings
            self.get_logger().info(
                f'[{self.callsign}] pos=({x:.1f},{y:.1f}) cell=({col},{row}) '
                f'regime={self.regime} '
                f'sensors=[T:{r.get("thermal",0):.2f} A:{r.get("audio",0):.2f} '
                f'C:{r.get("camera",0):.2f} G:{r.get("gas",0):.2f}] '
                f'conf={self.confidence:.2f} '
                f'visc={self.viscosity:.2f} unc={self.uncertainty:.2f}'
            )

    # ── Communication ─────────────────────────────────────────────────────────

    def _execute_relay_mission(self, x, y, col, row, local_pos):
        """Fly to GCS, transmit only confirmed POIs, then return to the prior search point."""
        gcs_x, gcs_y = self.home_position
        dx = gcs_x - x
        dy = gcs_y - y
        dist_to_gcs = math.hypot(dx, dy)

        if self.relay_phase == 'FLYING_TO_GCS':
            if self.tick % self.relay_heartbeat_interval_ticks == 0:
                self._broadcast_relay_heartbeat()
            speed = REGIME_PARAMS['RESCUED']['speed']
            desired_vx = (dx / max(dist_to_gcs, 0.1)) * speed
            desired_vy = (dy / max(dist_to_gcs, 0.1)) * speed
            self.vx, self.vy = self._safe_velocity(
                desired_vx, desired_vy, x, y, speed
            )
            self.px4.send_velocity(self.vy, self.vx, 0.0)

            if self.tick % 60 == 0:
                self.get_logger().info(
                    f'[{self.callsign}] 📡 RELAY: Flying to GCS... {int(dist_to_gcs)}m to go'
                )

            if dist_to_gcs < self.comm_range_m:
                self.relay_phase = 'DUMPING'
                self.relay_dump_ticks = 0
                self.get_logger().info(
                    f'[{self.callsign}] 📡 RELAY: In GCS range! Dumping confirmed intel...'
                )

        elif self.relay_phase == 'DUMPING':
            self.px4.send_heartbeat(use_position=True)
            self.px4.send_position(
                x=local_pos[1], y=local_pos[0], z=-self.altitude
            )

            self.relay_dump_ticks += 1

            if self.relay_dump_ticks % 10 == 0:
                # CRITICAL: only confirmed POIs are eligible. Never mark an
                # unconfirmed candidate as relayed.
                unrelayed = [
                    key for key, poi in self.known_pois.items()
                    if poi.get('confirmed', False) and key not in self.gcs_relayed_pois
                ]
                for poi_key in unrelayed:
                    poi = self.known_pois[poi_key]
                    self._broadcast_gcs_relay(poi)
                    self.gcs_relayed_pois.add(poi_key)
                    self._broadcast_relay_gcs_relayed(poi_key)
                    self.get_logger().info(
                        f'[{self.callsign}] 📡 GCS_RELAY: Transmitted VERIFIED POI '
                        f'({poi["col"]},{poi["row"]}) to operators!'
                    )

            if self.relay_dump_ticks > 60:
                self.relay_phase = 'RETURNING'
                self.get_logger().info(
                    f'[{self.callsign}] ✅ RELAY DUMP COMPLETE: Returning to previous search area.'
                )

        elif self.relay_phase == 'RETURNING':
            # Actually return to the location where the relay mission started.
            resume = self.relay_resume_position
            if resume is None:
                self._finish_relay_mission()
                return

            dx = resume[0] - x
            dy = resume[1] - y
            dist = math.hypot(dx, dy)
            if dist < 10.0:
                self._finish_relay_mission()
                self.get_logger().info(
                    f'[{self.callsign}] ✅ RELAY RETURN COMPLETE: Resuming search.'
                )
                return

            speed = REGIME_PARAMS['SPREAD']['speed']
            desired_vx = (dx / max(dist, 0.1)) * speed
            desired_vy = (dy / max(dist, 0.1)) * speed
            self.vx, self.vy = self._safe_velocity(
                desired_vx, desired_vy, x, y, speed
            )
            self.px4.send_velocity(self.vy, self.vx, 0.0)

            if self.tick % 60 == 0:
                self.get_logger().info(
                    f'[{self.callsign}] 📡 RELAY: Returning to search area... {int(dist)}m to go'
                )


    # ── Shared coverage / global coordination ────────────────────────────────

    def _publish_control(self, payload: dict):
        """Publish a reliable swarm-wide control-plane message."""
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_control.publish(msg)

    def _mark_covered_cell(self, col: int, row: int, broadcast: bool = True):
        """Permanently mark a grid cell as covered and synchronize it globally."""
        changed = False
        radius = self.coverage_mark_radius_cells
        for rr in range(row - radius, row + radius + 1):
            for cc in range(col - radius, col + radius + 1):
                if 0 <= rr < self.coverage_rows and 0 <= cc < self.coverage_cols:
                    if not self.exploration_grid[rr, cc]:
                        self.exploration_grid[rr, cc] = True
                        changed = True
                        if broadcast:
                            self._broadcast_coverage_update(cc, rr)
        if changed:
            self.coverage_target_cell = None

    def _broadcast_coverage_update(self, col: int, row: int):
        coverage_id = f"{self.drone_id}:{col}:{row}"
        if coverage_id in self.coverage_seen_messages:
            return
        self.coverage_seen_messages.add(coverage_id)
        self._publish_control({
            'type': 'COVERAGE_UPDATE',
            'drone_id': self.drone_id,
            'col': int(col),
            'row': int(row),
            'coverage_id': coverage_id,
            'tick': self.tick,
        })

    def _encode_coverage(self) -> str:
        packed = np.packbits(self.exploration_grid.astype(np.uint8).reshape(-1))
        return base64.b64encode(packed.tobytes()).decode('ascii')

    def _merge_coverage(self, encoded: str):
        try:
            raw = base64.b64decode(encoded.encode('ascii'), validate=True)
            total = self.coverage_rows * self.coverage_cols
            incoming = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:total]
            incoming = incoming.reshape((self.coverage_rows, self.coverage_cols)).astype(bool)
            before = int(self.exploration_grid.sum())
            self.exploration_grid |= incoming
            after = int(self.exploration_grid.sum())
            if after > before:
                self.coverage_target_cell = None
        except (ValueError, TypeError):
            return

    def _broadcast_coverage_sync(self):
        self._publish_control({
            'type': 'COVERAGE_SYNC',
            'drone_id': self.drone_id,
            'rows': self.coverage_rows,
            'cols': self.coverage_cols,
            'data': self._encode_coverage(),
            'tick': self.tick,
        })

    def _get_coverage_gradient(self, col: int, row: int, x: float, y: float) -> Tuple[float, float]:
        """Return a unit vector toward a nearby unexplored cell."""
        if self.coverage_target_cell is not None:
            tc, tr = self.coverage_target_cell
            if 0 <= tr < self.coverage_rows and 0 <= tc < self.coverage_cols and not self.exploration_grid[tr, tc]:
                if self.tick - self.coverage_target_tick < self.coverage_target_refresh_ticks:
                    tx, ty = self.scenario.grid_to_world(tc, tr)
                    dx, dy = tx - x, ty - y
                    mag = math.hypot(dx, dy)
                    if mag > 0.001:
                        return dx / mag, dy / mag
            self.coverage_target_cell = None

        uncovered = np.argwhere(~self.exploration_grid)
        if uncovered.size == 0:
            return 0.0, 0.0

        # Prefer a nearby frontier, but add a tiny deterministic tie-breaker so
        # several drones don't all chase the same exact cell.
        best = None
        best_score = float('inf')
        phase = self.drone_id * 0.173
        for tr, tc in uncovered:
            tc_i, tr_i = int(tc), int(tr)
            tx, ty = self.scenario.grid_to_world(tc_i, tr_i)
            d = math.hypot(tx - x, ty - y)
            if d < 1.0:
                continue
            tie = abs(math.sin((tc_i + 1) * 12.9898 + (tr_i + 1) * 78.233 + phase)) * 0.25
            score = d + tie
            if score < best_score:
                best_score = score
                best = (tc_i, tr_i)

        if best is None:
            return 0.0, 0.0
        self.coverage_target_cell = best
        self.coverage_target_tick = self.tick
        tx, ty = self.scenario.grid_to_world(best[0], best[1])
        dx, dy = tx - x, ty - y
        mag = math.hypot(dx, dy)
        return (dx / mag, dy / mag) if mag > 0.001 else (0.0, 0.0)

    def _broadcast_agent_status(self, x: float, y: float):
        state = 'RTB' if self.is_returning_home else (
            'SENTINEL' if self.is_sentinel else (
                'VERIFIER' if self.is_verifier else (
                    'RELAY' if self.is_relay_mission else self.regime)))
        self._publish_control({
            'type': 'AGENT_STATUS',
            'drone_id': self.drone_id,
            'x': float(x),
            'y': float(y),
            'gcs_dist': float(math.hypot(self.home_position[0] - x, self.home_position[1] - y)),
            'state': state,
            'battery': float((self.battery_remaining_s / self.battery_capacity_s) * 100.0)
                       if self.battery_capacity_s > 0 else 0.0,
            'tick': self.tick,
        })

    def _expire_coordination_state(self):
        # Global relay lock timeout.
        if (self.active_relay_id is not None
                and self.active_relay_id != self.drone_id
                and self.tick - self.active_relay_tick > self.relay_lock_timeout_ticks):
            self.active_relay_id = None
            self.active_relay_lock_id = None

        for nid, claim in list(self.verification_claims.items()):
            last = claim.get('last_heartbeat', claim.get('tick', self.tick))
            if self.tick - last > self.verifier_heartbeat_timeout_ticks:
                self.verification_claims.pop(nid, None)

        for nid, proposal in list(self.relay_proposals.items()):
            if self.tick - proposal.get('tick', self.tick) > self.relay_election_window_ticks * 2:
                self.relay_proposals.pop(nid, None)

        for nid, data in list(self.global_peer_states.items()):
            if self.tick - data.get('tick', self.tick) > self.relay_lock_timeout_ticks:
                self.global_peer_states.pop(nid, None)

    def _peer_has_active_verification(self, drone_id: int) -> bool:
        for claim in self.verification_claims.values():
            if claim.get('verifier_id') == drone_id:
                return True
        state = self.global_peer_states.get(drone_id, {})
        return state.get('state') == 'VERIFIER'

    def _maybe_start_relay_election(self, unrelayed_confirmed):
        if not unrelayed_confirmed:
            self.relay_proposal_sent = False
            self.relay_proposal_tick = 0
            return
        if self.is_sentinel or self.is_verifier or self.is_relay_mission or self.is_returning_home:
            return
        if self.active_relay_id is not None and self.active_relay_id != self.drone_id:
            return

        if not self.relay_proposal_sent:
            my_pos = self.px4.get_position_enu()
            if not my_pos:
                return
            x = my_pos[0] + self.spawn_offset_x
            y = my_pos[1] + self.spawn_offset_y
            gcs_dist = math.hypot(self.home_position[0] - x, self.home_position[1] - y)
            self.relay_proposal_sent = True
            self.relay_proposal_tick = self.tick
            self.relay_proposals[self.drone_id] = {'gcs_dist': gcs_dist, 'tick': self.tick}
            self._publish_control({
                'type': 'RELAY_PROPOSAL',
                'drone_id': self.drone_id,
                'gcs_dist': gcs_dist,
                'tick': self.tick,
            })
            return

        if self.tick - self.relay_proposal_tick < self.relay_election_window_ticks:
            return

        # Pick exactly one winner among proposals received during the election window.
        fresh = [
            (nid, prop.get('gcs_dist', float('inf')))
            for nid, prop in self.relay_proposals.items()
            if self.tick - prop.get('tick', self.tick) <= self.relay_election_window_ticks
        ]
        if not fresh:
            self.relay_proposal_sent = False
            return
        winner_id, _ = min(fresh, key=lambda item: (item[1], item[0]))
        if winner_id != self.drone_id:
            self.relay_proposal_sent = False
            return

        lock_id = f"relay:{self.drone_id}:{self.tick}"
        self.active_relay_id = self.drone_id
        self.active_relay_tick = self.tick
        self.active_relay_lock_id = lock_id
        self.is_relay_mission = True
        self.relay_phase = 'FLYING_TO_GCS'
        self.relay_dump_ticks = 0
        self.relay_proposal_sent = False
        self.relay_resume_position = None
        self.regime = 'RESCUED'
        self._publish_control({
            'type': 'RELAY_LOCK',
            'drone_id': self.drone_id,
            'lock_id': lock_id,
            'tick': self.tick,
            'gcs_dist': min(v for n, v in fresh if n == winner_id),
        })
        self.get_logger().info(
            f'[{self.callsign}] 📡 RELAY ELECTION WON — sole relay drone for new GCS intel.'
        )
        my_pos = self.px4.get_position_enu()
        if my_pos:
            self.relay_resume_position = (
                my_pos[0] + self.spawn_offset_x,
                my_pos[1] + self.spawn_offset_y,
            )

    def _broadcast_relay_heartbeat(self):
        if self.active_relay_id != self.drone_id:
            return
        self.active_relay_tick = self.tick
        self._publish_control({
            'type': 'RELAY_HEARTBEAT',
            'drone_id': self.drone_id,
            'lock_id': self.active_relay_lock_id,
            'tick': self.tick,
        })

    def _broadcast_relay_release(self):
        if self.active_relay_id != self.drone_id:
            return
        self._publish_control({
            'type': 'RELAY_RELEASE',
            'drone_id': self.drone_id,
            'lock_id': self.active_relay_lock_id,
            'tick': self.tick,
        })
        self.active_relay_id = None
        self.active_relay_lock_id = None
        self.active_relay_tick = 0

    def _broadcast_relay_gcs_relayed(self, poi_key: str):
        self._publish_control({
            'type': 'GCS_RELAYED',
            'drone_id': self.drone_id,
            'poi_key': poi_key,
            'tick': self.tick,
        })

    def _broadcast_verify_lock_acquired(self, poi_key: str, request_id, requester_id: int):
        claim = {
            'verifier_id': self.drone_id,
            'requester_id': requester_id,
            'request_id': request_id,
            'tick': self.tick,
            'last_heartbeat': self.tick,
            'provisional': True,
            'order': (self.tick, self.drone_id),
        }
        self.verification_claims[poi_key] = claim
        self._publish_control({
            'type': 'VERIFY_LOCK_ACQUIRED',
            'drone_id': self.drone_id,
            'poi_key': poi_key,
            'request_id': request_id,
            'requester_id': requester_id,
            'tick': self.tick,
        })

    def _broadcast_verify_lock_release(self, poi_key: str, reason='finished'):
        claim = self.verification_claims.get(poi_key)
        if claim and claim.get('verifier_id') != self.drone_id and reason != 'timeout':
            return
        self.verification_claims.pop(poi_key, None)
        col = row = None
        try:
            col, row = map(int, poi_key.split('_'))
        except (ValueError, AttributeError):
            pass
        self._publish_control({
            'type': 'VERIFY_LOCK_RELEASE',
            'drone_id': self.drone_id,
            'poi_key': poi_key,
            'reason': reason,
            'requester_id': claim.get('requester_id') if claim else None,
            'request_id': claim.get('request_id') if claim else None,
            'col': col,
            'row': row,
            'tick': self.tick,
        })

    def _broadcast_verify_heartbeat(self):
        if not self.is_verifier or not self.verify_target_cell:
            return
        poi_key = f"{self.verify_target_cell[0]}_{self.verify_target_cell[1]}"
        claim = self.verification_claims.get(poi_key)
        if claim and claim.get('verifier_id') == self.drone_id:
            claim['last_heartbeat'] = self.tick
        self._publish_control({
            'type': 'VERIFY_HEARTBEAT',
            'drone_id': self.drone_id,
            'poi_key': poi_key,
            'requester_id': self.verify_requester_id,
            'request_id': self.verify_request_id,
            'tick': self.tick,
        })

    def _process_pending_verification(self):
        if not self.pending_verify_request:
            return
        if self.tick < self.pending_verify_until_tick:
            return
        req = self.pending_verify_request
        self.pending_verify_request = None
        poi_key = f"{req['col']}_{req['row']}"
        claim = self.verification_claims.get(poi_key)
        if not claim or claim.get('verifier_id') != self.drone_id:
            return
        if self.is_sentinel or self.is_verifier or self.is_relay_mission or self.is_returning_home:
            self._broadcast_verify_lock_release(poi_key, reason='busy')
            return

        self.verified_pois.add(poi_key)
        self.known_pois[poi_key] = {
            'col': req['col'], 'row': req['row'],
            'x': req['poi_x'], 'y': req['poi_y'],
            'finder': req.get('callsign', '??'),
            'finder_id': req.get('finder_id', req.get('drone_id')),
            'tick': self.tick, 'confirmed': False, 'status': 'verifying'
        }
        self.is_verifier = True
        self.verify_target_cell = (req['col'], req['row'])
        self.verify_target_world = (req['poi_x'], req['poi_y'])
        self.verify_requester_id = req.get('drone_id', -1)
        self.verify_request_id = req.get('request_id')
        self.verify_started_tick = self.tick
        self.verifier_last_heartbeat_tick = self.tick
        claim = self.verification_claims.get(poi_key)
        if claim and claim.get('verifier_id') == self.drone_id:
            claim['provisional'] = False
            claim['last_heartbeat'] = self.tick
        self.regime = 'CONVERGE'
        self._broadcast_verify_accept(
            self.verify_requester_id, self.verify_request_id,
            req['col'], req['row']
        )
        self.get_logger().info(
            f"[{self.callsign}] 🫡 RECRUITED by {req.get('callsign', '??')} "
            f"to verify POI at ({req['col']},{req['row']})"
        )

    def _on_control(self, msg: String):
        """Handle global control-plane messages without physical radio range filtering."""
        try:
            data = json.loads(msg.data)
            msg_type = data.get('type')
            sender = data.get('drone_id')
            if sender == self.drone_id:
                return

            if msg_type == 'COVERAGE_UPDATE':
                cid = data.get('coverage_id')
                if cid in self.coverage_seen_messages:
                    return
                self.coverage_seen_messages.add(cid)
                c, r = int(data['col']), int(data['row'])
                if 0 <= r < self.coverage_rows and 0 <= c < self.coverage_cols:
                    self.exploration_grid[r, c] = True
                    self.coverage_target_cell = None

            elif msg_type == 'COVERAGE_SYNC':
                if int(data.get('rows', -1)) != self.coverage_rows or int(data.get('cols', -1)) != self.coverage_cols:
                    return
                self._merge_coverage(data.get('data', ''))

            elif msg_type == 'AGENT_STATUS':
                if sender is not None:
                    self.global_peer_states[sender] = {
                        'x': float(data.get('x', 0.0)),
                        'y': float(data.get('y', 0.0)),
                        'gcs_dist': float(data.get('gcs_dist', float('inf'))),
                        'state': data.get('state', 'UNKNOWN'),
                        'battery': float(data.get('battery', 100.0)),
                        'tick': self.tick,
                    }
                    if data.get('state') == 'RELAY':
                        self.active_relay_id = sender
                        self.active_relay_tick = self.tick

            elif msg_type == 'RELAY_PROPOSAL':
                if sender is not None:
                    self.relay_proposals[sender] = {
                        'gcs_dist': float(data.get('gcs_dist', float('inf'))),
                        'tick': self.tick,
                    }

            elif msg_type == 'RELAY_LOCK':
                self.active_relay_id = sender
                self.active_relay_lock_id = data.get('lock_id')
                self.active_relay_tick = self.tick
                if sender != self.drone_id and self.is_relay_mission:
                    self._finish_relay_mission()
                self.relay_proposal_sent = False

            elif msg_type == 'RELAY_HEARTBEAT':
                self.active_relay_id = sender
                self.active_relay_lock_id = data.get('lock_id')
                self.active_relay_tick = self.tick

            elif msg_type == 'RELAY_RELEASE':
                if self.active_relay_id == sender:
                    self.active_relay_id = None
                    self.active_relay_lock_id = None
                    self.active_relay_tick = 0

            elif msg_type == 'GCS_RELAYED':
                poi_key = data.get('poi_key')
                if poi_key:
                    self.gcs_relayed_pois.add(poi_key)

            elif msg_type == 'POI_CONFIRMED':
                pc, pr = data.get('col'), data.get('row')
                if pc is None or pr is None:
                    return
                pkey = f"{int(pc)}_{int(pr)}"
                existing = self.known_pois.get(pkey, {})
                self._mark_poi_confirmed(
                    pkey, int(pc), int(pr),
                    float(data.get('poi_x', 0.0)), float(data.get('poi_y', 0.0)),
                    finder=existing.get('finder', '??'),
                    finder_id=existing.get('finder_id'),
                )
                self.pending_outgoing_verification.pop(pkey, None)
                self.unverified_pois.pop(pkey, None)
                self.unverified_request_cooldowns.pop(pkey, None)

            elif msg_type == 'POI_REJECTED':
                pc, pr = data.get('col'), data.get('row')
                if pc is None or pr is None:
                    return
                pkey = f"{int(pc)}_{int(pr)}"
                self._mark_poi_rejected(pkey)
                self.pending_outgoing_verification.pop(pkey, None)
                self.unverified_request_cooldowns.pop(pkey, None)

            elif msg_type == 'VERIFY_LOCK_ACQUIRED':
                poi_key = data.get('poi_key')
                if not poi_key or sender is None:
                    return
                incoming_order = (int(data.get('tick', self.tick)), int(sender))
                incoming = {
                    'verifier_id': sender,
                    'requester_id': data.get('requester_id'),
                    'request_id': data.get('request_id'),
                    'tick': self.tick,
                    'last_heartbeat': self.tick,
                    'provisional': True,
                    'order': incoming_order,
                }
                existing = self.verification_claims.get(poi_key)
                if existing is None:
                    self.verification_claims[poi_key] = incoming
                    return

                existing_owner = existing.get('verifier_id')
                existing_order = existing.get('order', (self.tick, existing_owner if existing_owner is not None else 10**9))
                # During the provisional settlement window, deterministic order
                # decides the winner. Once committed, the current verifier keeps
                # the lock until it finishes or times out.
                if (existing_owner == self.drone_id
                        and existing.get('provisional', False)
                        and incoming_order < existing_order):
                    if self.is_verifier and self.verify_target_cell:
                        current_key = f"{self.verify_target_cell[0]}_{self.verify_target_cell[1]}"
                        if current_key == poi_key:
                            self._finish_verifier(reset_heading=True)
                    self.verification_claims[poi_key] = incoming
                elif (existing.get('provisional', False)
                      and incoming_order < existing_order):
                    self.verification_claims[poi_key] = incoming

            elif msg_type == 'VERIFY_LOCK_RELEASE':
                poi_key = data.get('poi_key')
                if not poi_key:
                    return
                claim = self.verification_claims.get(poi_key)
                if claim and (claim.get('verifier_id') == sender or data.get('reason') == 'timeout'):
                    self.verification_claims.pop(poi_key, None)

                # A requester whose verifier disappeared keeps the abandoned
                # POI available for the next physical drone encounter.
                if (data.get('requester_id') == self.drone_id
                        and data.get('reason') in ('timeout', 'busy', 'rtb')):
                    try:
                        c, r = map(int, poi_key.split('_'))
                        existing = self.known_pois.get(poi_key)
                        if existing and not existing.get('confirmed', False):
                            self.unverified_pois[poi_key] = {
                                'col': c, 'row': r,
                                'x': float(existing.get('x', 0.0)),
                                'y': float(existing.get('y', 0.0)),
                                'finder_id': existing.get('finder_id'),
                            }
                    except (ValueError, TypeError):
                        pass

            elif msg_type == 'VERIFY_HEARTBEAT':
                poi_key = data.get('poi_key')
                if poi_key:
                    claim = self.verification_claims.get(poi_key)
                    if claim:
                        claim['last_heartbeat'] = self.tick
                        claim['tick'] = self.tick
                    if self.is_sentinel and self.sentinel_cell == tuple(map(int, poi_key.split('_'))):
                        if data.get('requester_id') == self.drone_id:
                            self.verifier_last_heartbeat_tick = self.tick

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return

    def _broadcast_gcs_relay(self, poi: dict):
        """Broadcast GCS_RELAY message — only the dashboard/GCS receives this."""
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else 0.0
        gy = pos[1] + self.spawn_offset_y if pos else 0.0
        msg_data = {
            'type': 'GCS_RELAY',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'col': poi['col'],
            'row': poi['row'],
            'poi_x': poi.get('x', 0.0),
            'poi_y': poi.get('y', 0.0),
            'finder': poi.get('finder', self.callsign),
            'finder_id': poi.get('finder_id'),
            'confirmed': True,
            'sender_x': gx,
            'sender_y': gy,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)

    def _release_sentinel(self, s_col: int, s_row: int):
        """Common cleanup when exiting sentinel mode and propagate pheromone clear."""
        self.is_sentinel = False
        self.sentinel_target = None
        self.sentinel_cell = None
        self.sentinel_world = None
        self.sentinel_ticks = 0
        self.sentinel_verified = False
        self.verify_rejected = False
        self.verify_request_in_flight = False
        self.verify_request_target_id = None
        self.verify_request_id = None
        self.regime = 'SPREAD'
        self.confidence = 0.0
        self.viscosity = 0.0
        self.uncertainty = 1.0
        self.field.clear_local_attraction(s_col, s_row, radius=2.0)
        self._broadcast_pheromone_clear(s_col, s_row, radius=2.0)
        self._heading_bias = np.random.uniform(0, 2 * math.pi)
        self.vx = math.cos(self._heading_bias) * REGIME_PARAMS['SPREAD']['speed']
        self.vy = math.sin(self._heading_bias) * REGIME_PARAMS['SPREAD']['speed']

    def _finish_verifier(self, reset_heading: bool = True, reason: str = 'finished'):
        """Exit verifier mode cleanly and release its global POI lock."""
        if self.verify_target_cell is not None:
            self._broadcast_verify_lock_release(
                f"{self.verify_target_cell[0]}_{self.verify_target_cell[1]}", reason=reason
            )
        self.is_verifier = False
        self.verify_target_cell = None
        self.verify_target_world = None
        self.verify_requester_id = None
        self.verify_request_id = None
        self.verify_started_tick = 0
        self.regime = 'SPREAD'
        self.confidence = 0.0
        self.viscosity = 0.0
        self.uncertainty = 1.0
        if reset_heading:
            self._heading_bias = np.random.uniform(0, 2 * math.pi)
        self.vx = math.cos(self._heading_bias) * REGIME_PARAMS['SPREAD']['speed']
        self.vy = math.sin(self._heading_bias) * REGIME_PARAMS['SPREAD']['speed']

    def _finish_relay_mission(self):
        """Reset relay state after returning to the search area."""
        if self.active_relay_id == self.drone_id:
            self._broadcast_relay_release()
        self.is_relay_mission = False
        self.relay_phase = 'FLYING_TO_GCS'
        self.relay_dump_ticks = 0
        self.relay_resume_position = None
        self.relay_claim_tick = 0
        self.regime = 'SPREAD'
        self.confidence = 0.0
        self.viscosity = 0.0
        self.uncertainty = 1.0
        self._heading_bias = np.random.uniform(0, 2 * math.pi)
        self.vx = math.cos(self._heading_bias) * REGIME_PARAMS['SPREAD']['speed']
        self.vy = math.sin(self._heading_bias) * REGIME_PARAMS['SPREAD']['speed']

    def _enter_return_to_base(self, x: float, y: float):
        """Abort active sub-missions cleanly before RTB due to low battery."""
        if self.is_sentinel and self.sentinel_cell:
            self._broadcast_verify_lock_release(
                f"{self.sentinel_cell[0]}_{self.sentinel_cell[1]}", reason='rtb'
            )
        if self.is_verifier and self.verify_target_cell:
            self._broadcast_verify_lock_release(
                f"{self.verify_target_cell[0]}_{self.verify_target_cell[1]}", reason='rtb'
            )
        if self.active_relay_id == self.drone_id:
            self._broadcast_relay_release()
        if self.is_sentinel and self.sentinel_cell and self.sentinel_world:
            s_col, s_row = self.sentinel_cell
            s_x, s_y = self.sentinel_world
            key = f"{s_col}_{s_row}"
            self.unverified_pois[key] = {
                'col': s_col, 'row': s_row, 'x': s_x, 'y': s_y,
                'finder_id': self.drone_id,
            }
        self.is_sentinel = False
        self.sentinel_target = None
        self.sentinel_cell = None
        self.sentinel_world = None
        self.sentinel_verified = False
        self.verify_rejected = False
        self.verify_request_in_flight = False
        self.verify_request_id = None
        self.verify_request_target_id = None
        self.is_verifier = False
        self.verify_target_cell = None
        self.verify_target_world = None
        self.verify_requester_id = None
        self.relay_resume_position = None if not self.is_relay_mission else self.relay_resume_position
        self.is_relay_mission = False
        self.relay_phase = 'FLYING_TO_GCS'
        self.relay_dump_ticks = 0
        self.is_returning_home = True
        self.regime = 'RESCUED'
        self.get_logger().warn(
            f'[{self.callsign}] 🔋 Battery low ({self.battery_remaining_s:.1f}s left) — RTB; active mission state cleared.'
        )

    def _safe_velocity(self, desired_vx: float, desired_vy: float,
                       x: float, y: float, max_speed: float) -> Tuple[float, float]:
        """Apply inter-drone repulsion to any direct-motion mission and clamp speed."""
        vx = desired_vx
        vy = desired_vy
        avoid_radius = 25.0
        avoid_strength = 5.0

        for nid, ndata in self.neighbor_positions.items():
            if nid == self.drone_id or self.tick - ndata.get('tick', 0) > 100:
                continue
            nx = ndata.get('x', 0.0)
            ny = ndata.get('y', 0.0)
            dx = x - nx
            dy = y - ny
            dist = math.hypot(dx, dy)
            if 0.1 < dist < avoid_radius:
                force = avoid_strength * (1.0 - dist / avoid_radius)
                vx += (dx / dist) * force
                vy += (dy / dist) * force

        speed = math.hypot(vx, vy)
        if speed > max_speed > 0.0:
            vx = (vx / speed) * max_speed
            vy = (vy / speed) * max_speed
        return vx, vy

    def _mark_poi_confirmed(self, poi_key: str, col: int, row: int,
                            x: float, y: float, finder: str = '??',
                            finder_id: Optional[int] = None):
        """Create/update a POI as confirmed while preserving the finder identity."""
        poi = self.known_pois.get(poi_key, {})
        self.known_pois[poi_key] = {
            'col': col, 'row': row, 'x': x, 'y': y,
            'finder': finder if finder != '??' else poi.get('finder', '??'),
            'finder_id': finder_id if finder_id is not None else poi.get('finder_id'),
            'tick': poi.get('tick', self.tick),
            'confirmed': True, 'status': 'confirmed',
        }
        self.unverified_pois.pop(poi_key, None)
        self.rejected_pois.pop(poi_key, None)

    def _mark_poi_rejected(self, poi_key: str):
        """Remove a rejected POI from active positive intel and keep negative memory."""
        self.known_pois.pop(poi_key, None)
        self.unverified_pois.pop(poi_key, None)
        self.rejected_pois[poi_key] = self.tick + 300

    def _broadcast_relay_claim(self, gcs_dist: float):
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else 0.0
        gy = pos[1] + self.spawn_offset_y if pos else 0.0
        msg_data = {
            'type': 'RELAY_CLAIM',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'gcs_dist': gcs_dist,
            'sender_x': gx,
            'sender_y': gy,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)

    def _broadcast_pheromone_clear(self, col: int, row: int, radius: float = 2.0):
        clear_id = f"{self.drone_id}:{col}:{row}:{self.tick}"
        self.seen_clears.add(clear_id)
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else 0.0
        gy = pos[1] + self.spawn_offset_y if pos else 0.0
        msg_data = {
            'type': 'PHEROMONE_CLEAR',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'clear_id': clear_id,
            'col': col,
            'row': row,
            'radius': radius,
            'sender_x': gx,
            'sender_y': gy,
            'hops': 0,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)

    def _broadcast_verify_confirm(self, col: int, row: int, requester_id: int,
                                  poi_x: float, poi_y: float, request_id=None):
        """Broadcast and allow hop-by-hop forwarding of a verified POI."""
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else poi_x
        gy = pos[1] + self.spawn_offset_y if pos else poi_y
        confirmation_id = f"{col}:{row}:{self.drone_id}:{request_id or self.tick}"
        self.seen_confirmations.add(confirmation_id)
        msg_data = {
            'type': 'VERIFY_CONFIRM',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'col': col,
            'row': row,
            'requester_id': requester_id,
            'request_id': request_id,
            'poi_x': poi_x,
            'poi_y': poi_y,
            'sender_x': gx,
            'sender_y': gy,
            'confirmation_id': confirmation_id,
            'hops': 0,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)
        self._publish_control({
            'type': 'POI_CONFIRMED',
            'drone_id': self.drone_id,
            'col': int(col), 'row': int(row),
            'poi_x': float(poi_x), 'poi_y': float(poi_y),
            'requester_id': requester_id,
            'request_id': request_id,
            'tick': self.tick,
            'confirmation_id': confirmation_id,
        })

    def _broadcast_verify_reject(self, col: int, row: int, requester_id: int,
                                 poi_x: float, poi_y: float, request_id=None):
        """Broadcast and forward a failed verification so stale candidates clear out."""
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else poi_x
        gy = pos[1] + self.spawn_offset_y if pos else poi_y
        rejection_id = f"{col}:{row}:{self.drone_id}:{request_id or self.tick}"
        self.seen_rejections.add(rejection_id)
        msg_data = {
            'type': 'VERIFY_REJECT',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'col': col,
            'row': row,
            'requester_id': requester_id,
            'request_id': request_id,
            'poi_x': poi_x,
            'poi_y': poi_y,
            'sender_x': gx,
            'sender_y': gy,
            'rejection_id': rejection_id,
            'hops': 0,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)
        self._publish_control({
            'type': 'POI_REJECTED',
            'drone_id': self.drone_id,
            'col': int(col), 'row': int(row),
            'poi_x': float(poi_x), 'poi_y': float(poi_y),
            'requester_id': requester_id,
            'request_id': request_id,
            'tick': self.tick,
            'rejection_id': rejection_id,
        })

    def _broadcast_verify_request(self, col: int, row: int, x: float, y: float,
                                  target_drone: int, request_id=None):
        """Ask one in-range peer to verify a POI."""
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else x
        gy = pos[1] + self.spawn_offset_y if pos else y
        msg_data = {
            'type': 'VERIFY_REQUEST',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'target_drone': target_drone,
            'col': col,
            'row': row,
            'px': x,
            'py': y,
            'poi_x': x,
            'poi_y': y,
            'sender_x': gx,
            'sender_y': gy,
            'request_id': request_id,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)

    def _broadcast_verify_accept(self, requester_id: int, request_id, col: int, row: int):
        """ACK a verification request so the sentinel knows the mission was received."""
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else 0.0
        gy = pos[1] + self.spawn_offset_y if pos else 0.0
        msg_data = {
            'type': 'VERIFY_ACCEPT',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'requester_id': requester_id,
            'request_id': request_id,
            'col': col,
            'row': row,
            'sender_x': gx,
            'sender_y': gy,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)

    def _broadcast_survivor_found(self, col: int, row: int, x: float, y: float):
        """Broadcast SURVIVOR_FOUND signal to all drones."""
        my_pos = self.px4.get_position_enu()
        gx = my_pos[0] + self.spawn_offset_x if my_pos else x
        gy = my_pos[1] + self.spawn_offset_y if my_pos else y

        # Register this POI in our own known_pois
        poi_key = f"{col}_{row}"
        existing = self.known_pois.get(poi_key, {})
        if existing.get('confirmed', False):
            # Keep a confirmed record authoritative. A later candidate cannot downgrade it.
            return
        self.known_pois[poi_key] = {
            'col': col, 'row': row, 'x': x, 'y': y,
            'finder': self.callsign, 'finder_id': self.drone_id,
            'tick': self.tick, 'confirmed': False, 'status': 'candidate'
        }
        
        msg_data = {
            'type': 'SURVIVOR_FOUND',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'col': col,
            'row': row,
            'px': x,
            'py': y,
            'sender_x': gx,
            'sender_y': gy,
            'confidence': self.confidence,
            'finder_id': self.drone_id,
            'tick': self.tick,
            'hops': 0
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)
        self.get_logger().info(
            f'[{self.callsign}] 📡 SURVIVOR_FOUND broadcast — '
            f'cell=({col},{row}) pos=({x:.1f},{y:.1f}) conf={self.confidence:.2f}'
        )

    def _broadcast_observation(self, col: int, row: int):
        """Broadcast current observation + position to nearby drones via radio topic."""
        pos = self.px4.get_position_enu()

        # Report the TRUE operational state, not the FSM regime
        if self.is_sentinel:
            report_regime = 'SENTINEL'
        elif self.is_verifier:
            report_regime = 'VERIFIER'
        elif self.is_relay_mission:
            report_regime = 'RELAY'
        elif self.is_returning_home:
            report_regime = 'RTB'
        else:
            report_regime = self.regime

        msg_data = {
            'type': 'OBSERVATION',
            'drone_id': self.drone_id,
            'callsign': self.callsign,
            'col': col,
            'row': row,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'regime': report_regime,
            'tick': self.tick,
            'speed': math.hypot(self.vx, self.vy),
            'battery': ((self.battery_remaining_s / self.battery_capacity_s) * 100.0
                        if self.battery_capacity_s > 0 else 0.0),
            'sensors': self.last_readings,
            'sender_x': pos[0] + self.spawn_offset_x if pos else 0.0,
            'sender_y': pos[1] + self.spawn_offset_y if pos else 0.0,
            'px': pos[0] if pos else 0.0,
            'py': pos[1] if pos else 0.0,
        }

        # Add sentinel/verifier metadata for dashboard
        if self.is_sentinel and self.sentinel_cell:
            msg_data['sentinel_cell'] = list(self.sentinel_cell)
            msg_data['sentinel_verified'] = self.sentinel_verified
        if self.is_verifier and self.verify_target_cell:
            msg_data['verify_target'] = list(self.verify_target_cell)
            msg_data['verify_for'] = self.verify_requester_id

        # Add candidate proposal if confidence is high enough
        if self.confidence >= 0.35 and not self.is_sentinel and not self.is_verifier:
            msg_data['type'] = 'CANDIDATE_PROPOSAL'

        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_radio.publish(msg)

    def _publish_deposit(self, col: int, row: int, amount: float):
        """Broadcast pheromone deposit with GLOBAL sender coordinates for range filtering."""
        pos = self.px4.get_position_enu()
        gx = pos[0] + self.spawn_offset_x if pos else 0.0
        gy = pos[1] + self.spawn_offset_y if pos else 0.0
        msg_data = {
            'drone_id': self.drone_id,
            'col': col,
            'row': row,
            'amount': amount,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'sender_x': gx,
            'sender_y': gy,
            'px': gx,
            'py': gy,
            'tick': self.tick,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_deposit.publish(msg)

    def _on_radio(self, msg: String):
        """Receive, validate, and selectively forward swarm mission messages."""
        try:
            data = json.loads(msg.data)
            sender_id = data.get('drone_id')
            if sender_id == self.drone_id:
                return

            my_pos = self.px4.get_position_enu()
            if not my_pos:
                return

            gx = my_pos[0] + self.spawn_offset_x
            gy = my_pos[1] + self.spawn_offset_y
            sender_x = data.get('sender_x', data.get('px'))
            sender_y = data.get('sender_y', data.get('py'))

            # All peer positions are GLOBAL x/y. Drop messages outside radio range.
            if sender_x is not None and sender_y is not None:
                dist = math.hypot(float(sender_x) - gx, float(sender_y) - gy)
                if dist > self.comm_range_m:
                    return

            msg_type = data.get('type')

            if msg_type == 'SURVIVOR_FOUND':
                if 'col' not in data or 'row' not in data:
                    return
                poi_key = f"{data['col']}_{data['row']}"
                existing = self.known_pois.get(poi_key)
                if not existing or not existing.get('confirmed', False):
                    self.known_pois[poi_key] = {
                        'col': data['col'], 'row': data['row'],
                        'x': float(data.get('px', data.get('poi_x', 0.0))),
                        'y': float(data.get('py', data.get('poi_y', 0.0))),
                        'finder': data.get('callsign', '??'),
                        'finder_id': data.get('finder_id', sender_id),
                        'tick': data.get('tick', self.tick),
                        'confirmed': False, 'status': 'candidate'
                    }
                    self.get_logger().info(
                        f"[{self.callsign}] 📥 Learned about candidate POI at "
                        f"({data['col']},{data['row']}) found by {data.get('callsign', '??')}"
                    )

                # Daisy-chain once per POI. The forwarding position must remain global.
                if poi_key not in self.relayed_pois and data.get('hops', 0) < 20:
                    self.relayed_pois.add(poi_key)
                    relay_data = dict(data)
                    relay_data['sender_x'] = gx
                    relay_data['sender_y'] = gy
                    relay_data['hops'] = data.get('hops', 0) + 1
                    out = String()
                    out.data = json.dumps(relay_data)
                    self.pub_radio.publish(out)

                if sender_id is not None:
                    self.neighbor_positions[sender_id] = {
                        'x': float(data.get('sender_x', gx)),
                        'y': float(data.get('sender_y', gy)),
                        'regime': 'SENTINEL',
                        'tick': self.tick,
                    }

            elif msg_type == 'VERIFY_ACCEPT':
                if data.get('requester_id') != self.drone_id:
                    return
                request_id = data.get('request_id')
                col, row = data.get('col'), data.get('row')
                poi_key = f"{col}_{row}"

                if (self.is_sentinel
                        and request_id == self.verify_request_id
                        and self.sentinel_cell == (col, row)):
                    self.verify_request_in_flight = True
                    self.verify_request_target_id = sender_id
                    self.verify_request_deadline_tick = (
                        self.sentinel_ticks + self.verifier_timeout_ticks
                    )
                    self.verifier_last_heartbeat_tick = self.tick
                    self.unverified_pois.pop(poi_key, None)
                    self.get_logger().info(
                        f"[{self.callsign}] ✅ Drone {data.get('callsign', '??')} "
                        f"accepted verification request."
                    )
                elif self.pending_outgoing_verification.get(poi_key) == request_id:
                    # Abandoned-POI requester: ACK means the physical contact
                    # succeeded, so stop asking other drones about this POI.
                    self.pending_outgoing_verification.pop(poi_key, None)
                    self.unverified_pois.pop(poi_key, None)
                    self.unverified_request_cooldowns.pop(poi_key, None)
                    self.get_logger().info(
                        f"[{self.callsign}] ✅ Drone {data.get('callsign', '??')} accepted "
                        f"verification request for abandoned POI ({col},{row})."
                    )

            elif msg_type == 'VERIFY_CONFIRM':
                vc_col = data.get('col')
                vc_row = data.get('row')
                if vc_col is None or vc_row is None:
                    return
                vc_key = f"{vc_col}_{vc_row}"
                confirmation_id = data.get(
                    'confirmation_id',
                    f"{vc_col}:{vc_row}:{sender_id}:{data.get('tick')}"
                )
                if confirmation_id in self.seen_confirmations:
                    return
                self.seen_confirmations.add(confirmation_id)

                poi_x = float(data.get('poi_x', data.get('px', 0.0)))
                poi_y = float(data.get('poi_y', data.get('py', 0.0)))
                existing = self.known_pois.get(vc_key, {})
                self._mark_poi_confirmed(
                    vc_key, vc_col, vc_row, poi_x, poi_y,
                    finder=existing.get('finder', data.get('callsign', '??')),
                    finder_id=existing.get('finder_id'),
                )
                self.unverified_pois.pop(vc_key, None)
                self.pending_outgoing_verification.pop(vc_key, None)
                self.unverified_request_cooldowns.pop(vc_key, None)

                if (self.is_sentinel
                        and data.get('requester_id') == self.drone_id
                        and self.sentinel_cell == (vc_col, vc_row)):
                    self.sentinel_verified = True
                    self.verify_request_in_flight = False
                    self.get_logger().info(
                        f"[{self.callsign}] 🎉 Received VERIFY_CONFIRM from "
                        f"{data.get('callsign', '??')} for cell=({vc_col},{vc_row})!"
                    )

                # Cancel duplicate verification of the same POI.
                if (self.is_verifier
                        and self.verify_target_cell == (vc_col, vc_row)
                        and sender_id != self.drone_id):
                    self.get_logger().info(
                        f"[{self.callsign}] 🛑 Cancelling verify mission — "
                        f"{data.get('callsign', '??')} already confirmed ({vc_col},{vc_row})"
                    )
                    self._finish_verifier(reset_heading=True)

                # Forward confirmation so disconnected local neighborhoods eventually
                # converge when there is a radio-connected path.
                if data.get('hops', 0) < 20:
                    relay_data = dict(data)
                    relay_data['sender_x'] = gx
                    relay_data['sender_y'] = gy
                    relay_data['hops'] = data.get('hops', 0) + 1
                    out = String()
                    out.data = json.dumps(relay_data)
                    self.pub_radio.publish(out)

            elif msg_type == 'VERIFY_REJECT':
                rc = data.get('col')
                rr = data.get('row')
                if rc is None or rr is None:
                    return
                rkey = f"{rc}_{rr}"
                rejection_id = data.get(
                    'rejection_id',
                    f"{rc}:{rr}:{sender_id}:{data.get('tick')}"
                )
                if rejection_id in self.seen_rejections:
                    return
                self.seen_rejections.add(rejection_id)

                self._mark_poi_rejected(rkey)
                self.pending_outgoing_verification.pop(rkey, None)
                self.unverified_request_cooldowns.pop(rkey, None)
                if (self.is_sentinel
                        and data.get('requester_id') == self.drone_id
                        and self.sentinel_cell == (rc, rr)):
                    self.verify_rejected = True
                    self.verify_request_in_flight = False
                    self.get_logger().info(
                        f"[{self.callsign}] ❌ Verification rejected POI "
                        f"({rc},{rr}) by {data.get('callsign', '??')}."
                    )

                if data.get('hops', 0) < 20:
                    relay_data = dict(data)
                    relay_data['sender_x'] = gx
                    relay_data['sender_y'] = gy
                    relay_data['hops'] = data.get('hops', 0) + 1
                    out = String()
                    out.data = json.dumps(relay_data)
                    self.pub_radio.publish(out)

            elif msg_type == 'VERIFY_REQUEST':
                if data.get('target_drone') != self.drone_id:
                    return
                if (self.is_sentinel or self.is_verifier or self.is_relay_mission
                        or self.is_returning_home or self.has_landed):
                    return
                if 'col' not in data or 'row' not in data:
                    return

                poi_key = f"{data['col']}_{data['row']}"
                if poi_key in self.verified_pois:
                    return
                existing_claim = self.verification_claims.get(poi_key)
                if existing_claim and existing_claim.get('verifier_id') not in (None, self.drone_id):
                    return

                # Provisional global lock. Wait a few ticks before accepting so
                # simultaneous requests for the same POI converge to one verifier.
                self._broadcast_verify_lock_acquired(
                    poi_key, data.get('request_id'), data.get('drone_id', -1)
                )
                self.pending_verify_request = dict(data)
                self.pending_verify_until_tick = self.tick + self.verify_lock_settle_ticks

            elif msg_type == 'PHEROMONE_CLEAR':
                clear_id = data.get('clear_id')
                if not clear_id or clear_id in self.seen_clears:
                    return
                self.seen_clears.add(clear_id)
                if 'col' not in data or 'row' not in data:
                    return
                self.field.clear_local_attraction(
                    int(data['col']), int(data['row']),
                    radius=float(data.get('radius', 2.0)),
                )
                if data.get('hops', 0) < 20:
                    relay_data = dict(data)
                    relay_data['sender_x'] = gx
                    relay_data['sender_y'] = gy
                    relay_data['hops'] = data.get('hops', 0) + 1
                    out = String()
                    out.data = json.dumps(relay_data)
                    self.pub_radio.publish(out)

            elif msg_type == 'GCS_RELAY':
                col = data.get('col')
                row = data.get('row')
                if col is None or row is None or not data.get('confirmed', False):
                    return
                poi_key = f"{col}_{row}"
                if poi_key not in self.gcs_relayed_pois:
                    self.gcs_relayed_pois.add(poi_key)
                    # Preserve the confirmed coordinates received from the relay.
                    self._mark_poi_confirmed(
                        poi_key, col, row,
                        float(data.get('poi_x', 0.0)),
                        float(data.get('poi_y', 0.0)),
                        finder=data.get('finder', data.get('callsign', '??')),
                        finder_id=data.get('finder_id'),
                    )
                    self.get_logger().info(
                        f"[{self.callsign}] 📡 Heard {data.get('callsign', '??')} "
                        f"relay VERIFIED POI ({col},{row}) to GCS."
                    )

            elif msg_type == 'RELAY_CLAIM':
                if sender_id is not None:
                    self.relay_claims[sender_id] = {
                        'gcs_dist': float(data.get('gcs_dist', float('inf'))),
                        'tick': self.tick,
                        'x': float(data.get('sender_x', 0.0)),
                        'y': float(data.get('sender_y', 0.0)),
                    }
                    self.neighbor_positions[sender_id] = {
                        'x': float(data.get('sender_x', 0.0)),
                        'y': float(data.get('sender_y', 0.0)),
                        'regime': 'RELAY',
                        'tick': self.tick,
                    }

            # Telemetry also updates the peer table.
            if msg_type in ('OBSERVATION', 'CANDIDATE_PROPOSAL') and sender_id is not None:
                nx = float(data.get('sender_x', data.get('px', 0.0)))
                ny = float(data.get('sender_y', data.get('py', 0.0)))
                regime = data.get('regime', 'UNKNOWN')
                self.neighbor_positions[sender_id] = {
                    'x': nx,
                    'y': ny,
                    'regime': regime,
                    'tick': self.tick,
                    'battery': data.get('battery', 100.0),
                    'is_returning_home': regime == 'RTB',
                }
                if regime == 'RELAY':
                    self.relay_claims[sender_id] = {
                        'gcs_dist': math.hypot(
                            self.home_position[0] - nx,
                            self.home_position[1] - ny,
                        ),
                        'tick': self.tick,
                        'x': nx, 'y': ny,
                    }

                # Retry abandoned POIs, but never spam one peer every observation.
                if self.unverified_pois and regime == 'SPREAD':
                    distance = math.hypot(gx - nx, gy - ny)
                    if distance <= self.comm_range_m:
                        poi_key, poi_data = next(iter(self.unverified_pois.items()))
                        cooldown_until = self.unverified_request_cooldowns.get(poi_key, 0)
                        if self.tick >= cooldown_until and poi_key not in self.verified_pois:
                            request_id = f"{self.drone_id}:{poi_data['col']}:{poi_data['row']}:{self.tick}"
                            self._broadcast_verify_request(
                                poi_data['col'], poi_data['row'],
                                poi_data['x'], poi_data['y'],
                                target_drone=sender_id, request_id=request_id
                            )
                            self.pending_outgoing_verification[poi_key] = request_id
                            self.unverified_request_cooldowns[poi_key] = self.tick + 200
                            self.get_logger().info(
                                f"[{self.callsign}] 🗣️ Asking {data.get('callsign', '??')} "
                                f"to verify abandoned POI at ({poi_data['col']},{poi_data['row']})"
                            )

            # Expire short-lived state.
            for key, expiry in list(self.rejected_pois.items()):
                if self.tick >= expiry:
                    self.rejected_pois.pop(key, None)
            for key, expiry in list(self.unverified_request_cooldowns.items()):
                if self.tick >= expiry and key not in self.unverified_pois:
                    self.unverified_request_cooldowns.pop(key, None)
            for nid, claim in list(self.relay_claims.items()):
                if self.tick - claim.get('tick', 0) > 200:
                    self.relay_claims.pop(nid, None)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return

    def _on_deposit(self, msg: String):
        """Receive pheromone deposits using the same GLOBAL coordinate convention as radio."""
        try:
            data = json.loads(msg.data)
            if data.get('drone_id') == self.drone_id:
                return

            if 'col' not in data or 'row' not in data or 'amount' not in data:
                return

            sender_x = data.get('sender_x', data.get('px'))
            sender_y = data.get('sender_y', data.get('py'))
            if sender_x is not None and sender_y is not None:
                my_pos = self.px4.get_position_enu()
                if my_pos:
                    gx = my_pos[0] + self.spawn_offset_x
                    gy = my_pos[1] + self.spawn_offset_y
                    dist = math.hypot(float(sender_x) - gx, float(sender_y) - gy)
                    if dist > self.comm_range_m:
                        return

            self.field.deposit(
                int(data['col']), int(data['row']), float(data['amount']),
                drone_id=int(data['drone_id']),
                confidence=float(data.get('confidence', 0.0)),
                uncertainty=float(data.get('uncertainty', 0.0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return

def main(args=None):
    rclpy.init(args=args)

    import sys
    from rclpy.utilities import remove_ros_args
    clean_args = remove_ros_args(args=sys.argv)
    try:
        drone_id = int(clean_args[1]) if len(clean_args) > 1 else 0
    except ValueError:
        drone_id = 0
    namespace = clean_args[2] if len(clean_args) > 2 else ''

    node = DroneAgent(drone_id=drone_id, namespace=namespace)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

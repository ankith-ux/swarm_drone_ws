"""
swarm_manager.py — Swarm Orchestrator (Port of swarm.js)

This is NOT a central planner. It is only a simulation runner:
  - Runs pheromone field tick/decay
  - Publishes the shared pheromone field state for visualization
  - Tracks mission score and extraction events
  - Logs swarm-level narrative events

Drones are self-directed. This node has no authority over drone behavior.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import json
import numpy as np

from .pheromone_field import PheromoneField
from .scenario_loader import DisasterScenario


class SwarmManager(Node):
    """
    Simulation orchestrator for the Protoplasm swarm.
    Manages the shared pheromone field and publishes visualization data.
    """

    def __init__(self):
        super().__init__('swarm_manager')

        # ── Parameters ──
        self.declare_parameter('num_drones', 5)
        self.declare_parameter('scenario', 'grand-challenge')
        self.declare_parameter('grid_cols', 50)
        self.declare_parameter('grid_rows', 50)
        self.declare_parameter('world_size_x', 1000.0)
        self.declare_parameter('world_size_y', 1000.0)
        self.declare_parameter('tick_rate_hz', 20.0)
        self.declare_parameter('pheromone_decay_rate', 0.020)

        cols = self.get_parameter('grid_cols').value
        rows = self.get_parameter('grid_rows').value
        world_x = self.get_parameter('world_size_x').value
        world_y = self.get_parameter('world_size_y').value
        scenario_name = self.get_parameter('scenario').value
        tick_rate = self.get_parameter('tick_rate_hz').value
        decay_rate = self.get_parameter('pheromone_decay_rate').value

        # ── Core modules ──
        self.scenario = DisasterScenario(cols, rows, world_x, world_y, scenario_name)
        self.field = PheromoneField(cols, rows, base_decay_rate=decay_rate)

        # ── Mission tracking ──
        self.tick = 0
        self.mission_score = 0
        self.survivors_extracted = 0
        self.rescued_cells = set()
        self.event_log = []

        # ── QoS ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribe to drone deposits to update shared field ──
        self.sub_deposit = self.create_subscription(
            String, '/swarm/deposits', self._on_deposit, qos)

        # ── Subscribe to drone radio for mission tracking ──
        self.sub_radio = self.create_subscription(
            String, '/swarm/radio', self._on_radio, qos)

        # ── Publish pheromone field for visualization ──
        self.pub_pheromone_viz = self.create_publisher(
            MarkerArray, '/swarm/pheromone_viz', 10)

        # ── Publish mission events ──
        self.pub_events = self.create_publisher(
            String, '/swarm/events', qos)

        # ── Publish zone markers for Gazebo/RViz ──
        self.pub_zone_viz = self.create_publisher(
            MarkerArray, '/swarm/zone_markers', 10)

        # ── Main tick timer ──
        self.timer = self.create_timer(1.0 / tick_rate, self._tick_callback)

        # ── Slow visualization timer (2 Hz) ──
        self.viz_timer = self.create_timer(0.5, self._publish_viz)

        # Publish zone markers once at startup
        self._publish_zone_markers()

        self.get_logger().info(
            f'SwarmManager started | Scenario: {scenario_name} | '
            f'Grid: {cols}x{rows} | Survivors: {len(self.scenario.survivors)}'
        )

    def _tick_callback(self):
        """Run one simulation tick: decay pheromones and track mission."""
        self.tick += 1
        self.field.tick()

        # Log exploration progress periodically
        if self.tick % 200 == 0:
            explored = self.field.get_explored_percentage()
            self.get_logger().info(
                f'[Tick {self.tick}] Map explored: {explored:.1f}% | '
                f'Score: {self.mission_score} | Extracted: {self.survivors_extracted}'
            )

    def _on_deposit(self, msg: String):
        """Receive deposits from drones and update the shared field."""
        try:
            data = json.loads(msg.data)
            self.field.deposit(
                data['col'], data['row'], data['amount'],
                drone_id=data['drone_id'],
                confidence=data.get('confidence', 0),
                uncertainty=data.get('uncertainty', 0),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    def _on_radio(self, msg: String):
        """Monitor radio traffic for mission-level events."""
        try:
            data = json.loads(msg.data)
            msg_type = data.get('type', '')

            # Track high-confidence candidate proposals
            if msg_type == 'CANDIDATE_PROPOSAL' and data.get('confidence', 0) >= 0.65:
                col, row = data['col'], data['row']
                cell_key = f'{col},{row}'

                # Check if this is a real survivor cell
                cell_type = self.scenario.get_cell_type(col, row)
                if cell_type == 'SURVIVOR' and cell_key not in self.rescued_cells:
                    unique_count = self.field.get_unique_drone_count(col, row)
                    if unique_count >= 2:
                        self._extract_survivor(col, row, data['callsign'])

        except (json.JSONDecodeError, KeyError):
            pass

    def _extract_survivor(self, col: int, row: int, callsign: str):
        """Mark a survivor as extracted and update mission score."""
        cell_key = f'{col},{row}'
        if cell_key in self.rescued_cells:
            return

        self.rescued_cells.add(cell_key)
        self.survivors_extracted += 1
        self.mission_score += 500

        # Find which survivor this is
        survivor_name = 'Unknown'
        for s in self.scenario.survivors:
            if abs(s.col - col) <= 2 and abs(s.row - row) <= 2:
                s.status = 'EXTRACTED'
                survivor_name = s.name
                break

        # Clear local pheromone attraction
        self.field.clear_local_attraction(col, row, 4.0)

        event = f'🏆 SCORE +500! {survivor_name} extracted at ({col},{row})! ' \
                f'Total: {self.survivors_extracted}/{len(self.scenario.survivors)}'
        self.get_logger().info(event)
        self._publish_event(event)

    def _publish_event(self, text: str):
        """Publish a mission event."""
        msg = String()
        msg.data = json.dumps({'tick': self.tick, 'text': text})
        self.pub_events.publish(msg)

    def _publish_viz(self):
        """Publish pheromone field as RViz MarkerArray."""
        markers = MarkerArray()
        grid = self.field.to_grid_2d()
        cell_x = self.scenario.cell_size_x
        cell_y = self.scenario.cell_size_y

        marker_id = 0
        for row in range(self.field.rows):
            for col in range(self.field.cols):
                strength = grid[row, col]
                if strength < 0.01:
                    continue  # Skip empty cells for performance

                world_x, world_y = self.scenario.grid_to_world(col, row)

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = world_x
                marker.pose.position.y = world_y
                marker.pose.position.z = 0.1  # Slightly above ground
                marker.scale.x = cell_x * 0.9
                marker.scale.y = cell_y * 0.9
                marker.scale.z = 0.1

                # Color: orange gradient based on strength
                marker.color.r = 1.0
                marker.color.g = 0.5 * (1.0 - strength)
                marker.color.b = 0.0
                marker.color.a = min(0.8, strength * 2.0)

                marker.lifetime.sec = 1  # Auto-expire

                markers.markers.append(marker)
                marker_id += 1

        self.pub_pheromone_viz.publish(markers)

    def _publish_zone_markers(self):
        """Publish scenario zone markers (survivors, hazards, etc.) once."""
        markers = MarkerArray()
        marker_id = 0

        # Zone colors matching scenario.js
        zone_colors = {
            'SURVIVOR':   (0.23, 0.86, 0.39, 0.5),   # Green
            'HOT_DEBRIS': (1.0,  0.43, 0.08, 0.5),    # Orange
            'WIND_NOISE': (0.23, 0.63, 1.0,  0.5),    # Blue
            'HAZARD':     (0.78, 0.16, 0.86, 0.5),    # Purple
        }

        for row in range(self.scenario.rows):
            for col in range(self.scenario.cols):
                cell_type = self.scenario.get_cell_type(col, row)
                if cell_type not in zone_colors:
                    continue

                world_x, world_y = self.scenario.grid_to_world(col, row)
                r, g, b, a = zone_colors[cell_type]

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = world_x
                marker.pose.position.y = world_y
                marker.pose.position.z = 0.05
                marker.scale.x = self.scenario.cell_size_x * 0.95
                marker.scale.y = self.scenario.cell_size_y * 0.95
                marker.scale.z = 0.05

                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = a

                # Persistent (no lifetime)
                markers.markers.append(marker)
                marker_id += 1

        self.pub_zone_viz.publish(markers)
        self.get_logger().info(f'Published {marker_id} zone markers')


def main(args=None):
    rclpy.init(args=args)
    node = SwarmManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

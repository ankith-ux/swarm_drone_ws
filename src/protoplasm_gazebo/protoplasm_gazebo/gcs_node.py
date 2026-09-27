"""
gcs_node.py — Ground Control Station (GCS) for Grand Challenge 1

The GCS sits 75m outside the operational area.
Its job is to:
  - Track mission time (45 minutes total)
  - Receive SURVIVOR_FOUND reports from the drone swarm
  - Log which POIs have been found and when
  - Track mission metrics (found count, detection times)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
import json
import time
import math

class GCSNode(Node):
    def __init__(self):
        super().__init__('gcs_node')
        
        # ── Parameters ──
        self.declare_parameter('mission_time_s', 2700.0) # 45 minutes
        self.mission_time_s = self.get_parameter('mission_time_s').value
        
        # GCS is at the operational center
        self.gcs_x = -575.0
        self.gcs_y = 0.0
        
        # Mission state
        self.start_time = time.time()
        self.reported_pois = {}  # poi_id -> {time, x, y, reporter}
        self.mission_ended = False
        
        # ── QoS ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # ── Subscribers ──
        # Listen to the swarm radio for SURVIVOR_FOUND broadcasts
        self.sub_radio = self.create_subscription(
            String, '/swarm/radio', self._on_radio, qos)
        
        # Also listen to direct GCS reports
        self.sub_reports = self.create_subscription(
            String, '/gcs/reports', self._on_report_received, qos)
            
        # ── Timer ──
        self.timer = self.create_timer(10.0, self._mission_tick)
        
        self.get_logger().info("==================================================")
        self.get_logger().info(f"🛰️  GCS INITIALIZED at ({self.gcs_x}, {self.gcs_y})")
        self.get_logger().info(f"⏱️  Mission duration: {int(self.mission_time_s / 60)} minutes")
        self.get_logger().info(f"📡 Listening on /swarm/radio and /gcs/reports")
        self.get_logger().info("==================================================")

    def _mission_tick(self):
        """Periodic mission status update."""
        elapsed = time.time() - self.start_time
        remaining = self.mission_time_s - elapsed
        
        if remaining <= 0 and not self.mission_ended:
            self.mission_ended = True
            self.get_logger().warn("🚨 MISSION TIME EXPIRED!")
            self._print_summary()
            return
            
        mins_left = int(remaining // 60)
        secs_left = int(remaining % 60)
        found = len(self.reported_pois)
        self.get_logger().info(
            f"⏳ Time: {mins_left}m {secs_left}s remaining | "
            f"POIs found: {found}/10"
        )

    def _on_radio(self, msg: String):
        """Listen to swarm radio for SURVIVOR_FOUND and GCS_RELAY broadcasts."""
        try:
            data = json.loads(msg.data)
            
            # Enforce 100m comm limit for GCS
            sender_x = data.get('sender_x')
            sender_y = data.get('sender_y')
            if sender_x is not None and sender_y is not None:
                dist = math.hypot(sender_x - self.gcs_x, sender_y - self.gcs_y)
                if dist > 100.0:
                    return  # GCS can't hear this drone directly!
            
            if data.get('type') in ('SURVIVOR_FOUND', 'GCS_RELAY'):
                self._register_poi(data)
        except json.JSONDecodeError:
            pass

    def _on_report_received(self, msg: String):
        """Handle incoming POI reports sent directly to the GCS."""
        try:
            data = json.loads(msg.data)
            if data.get('type') == 'POI_REPORT':
                self._register_poi(data)
        except json.JSONDecodeError:
            self.get_logger().error("Received malformed JSON on /gcs/reports")

    def _register_poi(self, data: dict):
        """Register a newly discovered POI."""
        # Build a location-based key (round to nearest 20m to deduplicate)
        x = data.get('px', data.get('x', 0.0))
        y = data.get('py', data.get('y', 0.0))
        col = data.get('col', 0)
        row = data.get('row', 0)
        reporter = data.get('callsign', data.get('reporter_callsign', 'UNKNOWN'))
        
        # Deduplicate by grid cell
        poi_key = f"{col}_{row}"
        if poi_key in self.reported_pois:
            return  # Already known
        
        elapsed = time.time() - self.start_time
        self.reported_pois[poi_key] = {
            'time': elapsed,
            'x': x,
            'y': y,
            'col': col,
            'row': row,
            'reporter': reporter,
        }
        
        found = len(self.reported_pois)
        self.get_logger().info(
            f"✅ POI #{found} FOUND at cell ({col},{row}) pos ({x:.1f},{y:.1f}) | "
            f"Reporter: {reporter} | Detection time: {elapsed:.1f}s"
        )
        
        if found >= 10:
            self.get_logger().info("🎉 ALL 10 POIs FOUND! Mission objective complete!")
            self._print_summary()

    def _print_summary(self):
        """Print a mission summary."""
        elapsed = time.time() - self.start_time
        self.get_logger().info("=" * 50)
        self.get_logger().info("📊 MISSION SUMMARY")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        self.get_logger().info(f"POIs found: {len(self.reported_pois)}/10")
        for key, info in self.reported_pois.items():
            self.get_logger().info(
                f"  📍 Cell ({info['col']},{info['row']}) — "
                f"found at t={info['time']:.1f}s by {info['reporter']}"
            )
        self.get_logger().info("=" * 50)

def main(args=None):
    rclpy.init(args=args)
    node = GCSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

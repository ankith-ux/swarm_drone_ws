"""
px4_bridge.py — PX4 ↔ ROS 2 Offboard Control Bridge

Handles the low-level communication with PX4 SITL:
  - Arming the drone
  - Switching to offboard flight mode
  - Sending velocity setpoints (TrajectorySetpoint)
  - Reading drone position (VehicleLocalPosition)

Each drone instance gets its own PX4Bridge with namespaced topics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

import numpy as np
from typing import Optional, Tuple


class PX4Bridge:
    """
    Interface between the swarm's abstract velocity commands and PX4's
    message protocol. One PX4Bridge per drone.
    """

    def __init__(self, node: Node, drone_id: int, namespace: str = ''):
        """
        Args:
            node: Parent ROS 2 node (for creating pubs/subs).
            drone_id: Unique drone ID (0-indexed).
            namespace: Topic namespace prefix (e.g., '/px4_1').
        """
        self.node = node
        self.drone_id = drone_id
        self.ns = namespace

        # ── QoS Profile matching PX4's DDS configuration ──
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers ──
        self.pub_offboard_mode = node.create_publisher(
            OffboardControlMode,
            f'{namespace}/fmu/in/offboard_control_mode', qos)

        self.pub_trajectory = node.create_publisher(
            TrajectorySetpoint,
            f'{namespace}/fmu/in/trajectory_setpoint', qos)

        self.pub_vehicle_command = node.create_publisher(
            VehicleCommand,
            f'{namespace}/fmu/in/vehicle_command', qos)

        # ── Subscribers ──
        self.sub_local_position = node.create_subscription(
            VehicleLocalPosition,
            f'{namespace}/fmu/out/vehicle_local_position_v1',
            self._on_local_position, qos)

        self.sub_vehicle_status = node.create_subscription(
            VehicleStatus,
            f'{namespace}/fmu/out/vehicle_status_v4',
            self._on_vehicle_status, qos)

        # ── State ──
        self.position: Optional[Tuple[float, float, float]] = None  # (x, y, z) NED
        self.velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) NED
        self.heading: float = 0.0  # radians
        self.is_armed: bool = False
        self.nav_state: int = 0
        self.offboard_counter: int = 0
        self._takeoff_altitude: float = -10.0  # NED convention: negative = up

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _on_local_position(self, msg: VehicleLocalPosition):
        """Update position from PX4."""
        self.position = (msg.x, msg.y, msg.z)
        self.velocity = (msg.vx, msg.vy, msg.vz)
        self.heading = msg.heading

    def _on_vehicle_status(self, msg: VehicleStatus):
        """Update vehicle status."""
        self.is_armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state

    # ── Command methods ───────────────────────────────────────────────────────

    def arm(self):
        """Send ARM command to PX4."""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,  # 1.0 = arm
        )
        self.node.get_logger().info(f'[Drone {self.drone_id}] ARM command sent')

    def disarm(self):
        """Send DISARM command to PX4."""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,  # 0.0 = disarm
        )

    def land(self):
        """Send LAND command to PX4."""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND
        )
        self.node.get_logger().info(f'[Drone {self.drone_id}] LAND command sent')

    def set_offboard_mode(self):
        """Switch PX4 to offboard flight mode."""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,   # custom mode
            param2=6.0,   # OFFBOARD mode
        )
        self.node.get_logger().info(f'[Drone {self.drone_id}] OFFBOARD mode requested')

    def send_heartbeat(self, use_position: bool = False):
        """
        Send offboard control mode heartbeat.
        Must be called at >= 2Hz for PX4 to stay in offboard mode.

        Args:
            use_position: If True, tell PX4 to expect position setpoints.
                          If False, tell PX4 to expect velocity setpoints.
        """
        msg = OffboardControlMode()
        msg.position = use_position
        msg.velocity = not use_position
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.pub_offboard_mode.publish(msg)

    def send_velocity(self, vx: float, vy: float, vz: float, yaw: float = float('nan')):
        """
        Send velocity setpoint to PX4 in NED frame.

        Args:
            vx: North velocity (m/s)
            vy: East velocity (m/s)
            vz: Down velocity (m/s, negative = climb)
            yaw: Desired yaw (radians), NaN = don't care
        """
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [vx, vy, vz]
        msg.yaw = yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.pub_trajectory.publish(msg)

    def send_position(self, x: float, y: float, z: float, yaw: float = float('nan')):
        """
        Send position setpoint to PX4 in NED frame.
        Used for takeoff and precise hovering.
        """
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = yaw
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.pub_trajectory.publish(msg)

    def takeoff_sequence(self, target_altitude: float = 10.0) -> bool:
        """
        Execute the takeoff sequence: stream setpoints → arm → offboard → climb.
        Returns True when the drone has reached target altitude.

        Must be called repeatedly in a timer loop at ~20Hz.
        PX4 requires setpoints streaming for ~2 seconds BEFORE accepting offboard.
        """
        self._takeoff_altitude = -target_altitude  # NED: negative = up

        # Always send position-mode heartbeat during takeoff
        self.send_heartbeat(use_position=True)
        self.offboard_counter += 1

        # Always send a position setpoint (even before arming)
        # PX4 needs these streaming before it will accept offboard mode
        if self.position is not None:
            self.send_position(
                self.position[0],  # Hold current X
                self.position[1],  # Hold current Y
                self._takeoff_altitude,  # Climb to altitude
            )
        else:
            self.send_position(0.0, 0.0, self._takeoff_altitude)

        # After 40 ticks (2 seconds at 20Hz), switch to offboard and arm
        if self.offboard_counter == 40:
            self.set_offboard_mode()
            self.arm()

        # Retry arm/offboard periodically in case first attempt was rejected
        if self.offboard_counter > 40 and self.offboard_counter % 40 == 0:
            if not self.is_armed:
                self.node.get_logger().warn(f'[Drone {self.drone_id}] Retrying arm...')
                self.set_offboard_mode()
                self.arm()

        # Check if we've reached altitude
        if self.position is not None:
            current_alt = -self.position[2]  # Convert NED to AGL
            if current_alt >= target_altitude * 0.85:
                return True

        return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
        """Publish a VehicleCommand message."""
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = self.drone_id + 1  # PX4 sysid = instance_id + 1
        msg.target_component = 1
        msg.source_system = self.drone_id + 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_command.publish(msg)

    def get_position_enu(self) -> Optional[Tuple[float, float, float]]:
        """
        Get position in ENU (East-North-Up) frame, which matches Gazebo.
        PX4 uses NED internally; this converts for the swarm logic.
        """
        if self.position is None:
            return None
        # NED → ENU: swap x↔y, negate z
        return (self.position[1], self.position[0], -self.position[2])

    def get_velocity_enu(self) -> Optional[Tuple[float, float, float]]:
        """Get velocity in ENU frame."""
        if self.velocity is None:
            return None
        return (self.velocity[1], self.velocity[0], -self.velocity[2])


def main():
    """Standalone test: takeoff a single drone."""
    rclpy.init()
    node = rclpy.create_node('px4_bridge_test')
    bridge = PX4Bridge(node, drone_id=0)

    def timer_cb():
        reached = bridge.takeoff_sequence(target_altitude=10.0)
        if reached:
            node.get_logger().info('Takeoff complete! Hovering.')
            # Send zero velocity to hover
            bridge.send_heartbeat()
            bridge.send_velocity(0.0, 0.0, 0.0)

    timer = node.create_timer(0.05, timer_cb)  # 20Hz
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

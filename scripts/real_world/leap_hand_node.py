"""Minimal ROS 2 bridge for the LEAP Hand.

Requires the official LEAP Hand Python API (https://github.com/leap-hand/LEAP_Hand_API):
    export PYTHONPATH=$PYTHONPATH:/path/to/LEAP_Hand_API/python

    <ns>/joint_states    out  sensor_msgs/JointState  (published at --rate)
    <ns>/joint_commands  in   sensor_msgs/JointState  position targets

Joint angles use the simulation convention (0 = flat hand); hardware angles are offset by pi. Joints are
ordered index -> middle -> ring -> thumb, which matches both the motor IDs and the policy action order.
The hardware setup follows the LEAP Hand API example: current-based position control, kP=600, kD=200,
current limit 550 mA, and the hand moves to the flat pose when the node starts.

Example:
    python scripts/real_world/leap_hand_node.py --port /dev/ttyUSB0
"""

import argparse

import numpy as np
import rclpy
from leap_hand_utils.dynamixel_client import DynamixelClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]  # fmt: skip
LOWER = np.array([-0.314, -1.047, -0.506, -0.366] * 3 + [-0.349, -0.349, -0.470, -1.340])
UPPER = np.array([2.230, 1.047, 1.885, 2.042] * 3 + [2.094, 2.094, 2.443, 1.880])
MOTORS = list(range(16))
HW_OFFSET = 3.14159  # hardware angle = sim angle + pi (same constant as the LEAP Hand API)


class LeapHandNode(Node):
    def __init__(self, args):
        super().__init__("leap_hand_node")
        self.client = DynamixelClient(MOTORS, args.port, 4_000_000)
        self.client.connect()

        # Same sequence as the LEAP Hand API: current-based position mode, torque on, gains, current limit.
        self.client.sync_write(MOTORS, np.ones(16) * 5, 11, 1)  # operating mode
        self.client.set_torque_enabled(MOTORS, True)
        self.client.sync_write(MOTORS, np.ones(16) * args.kp, 84, 2)  # P gain
        self.client.sync_write([0, 4, 8], np.ones(3) * (args.kp * 0.75), 84, 2)
        self.client.sync_write(MOTORS, np.zeros(16), 82, 2)  # I gain
        self.client.sync_write(MOTORS, np.ones(16) * args.kd, 80, 2)  # D gain
        self.client.sync_write([0, 4, 8], np.ones(3) * (args.kd * 0.75), 80, 2)
        self.client.sync_write(MOTORS, np.ones(16) * args.current_limit, 102, 2)  # current limit
        self.set_joint_target(np.zeros(16))  # flat hand

        self.state_pub = self.create_publisher(JointState, f"{args.ns}/joint_states", qos_profile_sensor_data)
        self.create_subscription(JointState, f"{args.ns}/joint_commands", self._on_command, qos_profile_sensor_data)
        self.create_timer(1.0 / args.rate, self._publish_state)
        self.get_logger().info(f"LEAP Hand connected on {args.port}")

    def set_joint_target(self, q: np.ndarray) -> None:
        self.client.write_desired_pos(MOTORS, np.clip(q, LOWER, UPPER) + HW_OFFSET)

    def _publish_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = (self.client.read_pos() - HW_OFFSET).tolist()
        self.state_pub.publish(msg)

    def _on_command(self, msg: JointState) -> None:
        if msg.name:
            index = {n: i for i, n in enumerate(msg.name)}
            q = np.array([msg.position[index[n]] for n in JOINT_NAMES])
        else:
            q = np.asarray(msg.position)
        self.set_joint_target(q)

    def shutdown(self) -> None:
        self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyUSB0", help="See `ls /dev/serial/by-id/`.")
    parser.add_argument("--ns", default="/leap_hand")
    parser.add_argument("--rate", type=float, default=60.0, help="Joint state publishing rate [Hz].")
    parser.add_argument("--kp", type=float, default=600.0)
    parser.add_argument("--kd", type=float, default=200.0)
    parser.add_argument("--current_limit", type=float, default=550.0, help="mA; use 350 for LEAP Hand Lite.")
    args = parser.parse_args()

    rclpy.init()
    node = LeapHandNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

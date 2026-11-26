"""Simulated UR5e + hand + object for dry-running deploy.py without hardware.

The TCP jumps to every commanded pose, the hand echoes its joint commands, and the object is published at the
reference initial pose plus an optional planar shift (to exercise trajectory augmentation). Uses the same
topics as ur5e_node.py and the hand drivers, so deploy.py runs unchanged.

Example:
    python scripts/real_world/fake_robot.py --hand wuji --checkpoint <run_dir>/model_XXXX.pt \
        --sim2real_offset 0.15 -0.43 0.083 --object_shift 0.02 -0.01 0.1
"""

import argparse

import numpy as np
import rclpy
from deploy import (
    HAND_DEFAULTS,
    add_setup_args,
    msg_to_pose,
    pose_to_msg,
    resolve_sim2real_offset,
)
from geometry_msgs.msg import PoseStamped
from policy import RegrindPolicy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class FakeRobot(Node):
    def __init__(self, args, policy: RegrindPolicy):
        super().__init__("fake_robot")
        hand_ns = args.hand_ns or HAND_DEFAULTS[args.hand]["hand_ns"]
        offset = resolve_sim2real_offset(args)
        self.names = [n.removeprefix("right_") for n in policy.joint_names]  # drivers may drop the prefix
        self.ee_pos, self.ee_quat = np.array([0.0, -0.4, 0.4]), np.array([1.0, 0.0, 0.0, 0.0])
        self.q = np.zeros(len(self.names))
        self.counts = {"goto": 0, "servo": 0, "hand": 0}

        pos, quat, joint = policy.object_target(0)
        dx, dy, dyaw = args.object_shift
        self.obj_pos = pos + offset + np.array([dx, dy, 0.0])
        self.obj_quat = (R.from_euler("z", dyaw) * R.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True)
        self.obj_joint = joint if policy.is_articulated else None

        q = qos_profile_sensor_data
        self.ee_pub = self.create_publisher(PoseStamped, f"{args.arm_ns}/ee_pose", q)
        self.js_pub = self.create_publisher(JointState, f"{hand_ns}/joint_states", q)
        self.obj_pub = self.create_publisher(PoseStamped, "/object_pose", q)
        self.obj_joint_pub = self.create_publisher(Float64, "/object_joint", q)
        self.create_subscription(PoseStamped, f"{args.arm_ns}/ee_command", lambda m: self._arm(m, "servo"), q)
        self.create_subscription(PoseStamped, f"{args.arm_ns}/goto_ee_pose", lambda m: self._arm(m, "goto"), 10)
        self.create_subscription(JointState, f"{hand_ns}/joint_commands", self._hand, q)
        self.create_timer(1 / 125, self._tick)
        self.get_logger().info(f"Fake robot up (arm {args.arm_ns}, hand {hand_ns})")

    def _arm(self, msg, kind):
        self.counts[kind] += 1
        self.ee_pos, self.ee_quat = msg_to_pose(msg)

    def _hand(self, msg):
        self.counts["hand"] += 1
        index = {n.removeprefix("right_"): i for i, n in enumerate(msg.name)}
        self.q = np.array([msg.position[index[n]] for n in self.names])

    def _tick(self):
        self.ee_pub.publish(pose_to_msg(self.ee_pos, self.ee_quat, "ur5e_base"))
        js = JointState()
        js.name, js.position = self.names, [float(x) for x in self.q]
        self.js_pub.publish(js)
        self.obj_pub.publish(pose_to_msg(self.obj_pos, self.obj_quat, "ur5e_base"))
        if self.obj_joint is not None:
            self.obj_joint_pub.publish(Float64(data=float(self.obj_joint)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hand", choices=list(HAND_DEFAULTS), required=True)
    add_setup_args(parser)
    parser.add_argument("--object_shift", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DYAW"))
    parser.add_argument("--retargeted_traj", default=None)
    parser.add_argument("--arm_ns", default="/ur5e")
    parser.add_argument("--hand_ns", default=None)
    args = parser.parse_args()

    rclpy.init()
    node = FakeRobot(args, RegrindPolicy(args.checkpoint, args.retargeted_traj))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[fake_robot] received: {node.counts}", flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

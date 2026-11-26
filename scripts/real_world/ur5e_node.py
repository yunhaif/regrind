"""Minimal ROS 2 bridge for a UR5e over RTDE (pip install ur-rtde).

    <ns>/ee_pose       out  geometry_msgs/PoseStamped   actual TCP pose (published at --rate)
    <ns>/ee_command    in   geometry_msgs/PoseStamped   streamed TCP target, tracked with servoL
    <ns>/goto_ee_pose  in   geometry_msgs/PoseStamped   distant TCP target, reached with a blocking moveL

Poses are in the robot base frame with quaternions (w, x, y, z). The TCP configured on the teach pendant
is the controlled frame; set it to the hand root frame used in simulation.

Motion commands (servoL / moveL, both blocking) run in their own callback group so the TCP pose keeps being
published while the arm moves.

Example:
    python scripts/real_world/ur5e_node.py --robot_ip 192.168.1.10
"""

import argparse
import time

import numpy as np
import rclpy
import rtde_control
import rtde_receive
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R


class UR5eNode(Node):
    def __init__(self, args):
        super().__init__("ur5e_node")
        self.args = args
        self.rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        self.frame = args.ns.strip("/") + "_base"
        self.last_target = None  # last accepted servo target position

        control = MutuallyExclusiveCallbackGroup()  # all rtde_control calls are serialized here
        self.pose_pub = self.create_publisher(PoseStamped, f"{args.ns}/ee_pose", qos_profile_sensor_data)
        self.create_subscription(
            PoseStamped, f"{args.ns}/ee_command", self._on_command, qos_profile_sensor_data, callback_group=control
        )
        self.create_subscription(PoseStamped, f"{args.ns}/goto_ee_pose", self._on_goto, 10, callback_group=control)
        self.create_timer(1.0 / args.rate, self._publish_pose, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"Connected to UR5e at {args.robot_ip}")

    def _tcp_pose(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]

    def _publish_pose(self) -> None:
        tcp = self._tcp_pose()
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, tcp[:3])
        o = msg.pose.orientation
        o.w, o.x, o.y, o.z = map(float, R.from_rotvec(tcp[3:]).as_quat(scalar_first=True))
        self.pose_pub.publish(msg)

    @staticmethod
    def _to_tcp(msg: PoseStamped) -> list[float]:
        p, o = msg.pose.position, msg.pose.orientation
        rotvec = R.from_quat([o.w, o.x, o.y, o.z], scalar_first=True).as_rotvec()
        return [p.x, p.y, p.z, *rotvec]

    def _on_command(self, msg: PoseStamped) -> None:
        target = self._to_tcp(msg)
        # Safety: reject discontinuous targets. Compare with the previous servo target (the arm legitimately lags
        # the target during fast motions), or with the TCP for the first target after a goto.
        ref = self.last_target if self.last_target is not None else self._tcp_pose()[:3]
        jump = np.linalg.norm(np.asarray(target[:3]) - ref)
        if jump > self.args.max_jump:
            self.get_logger().warning(f"Ignoring ee_command that jumps {jump:.3f} m", throttle_duration_sec=1.0)
            return
        self.last_target = np.asarray(target[:3])
        # servoL(pose, speed, acceleration, time, lookahead_time, gain); blocks for `time`
        ok = self.rtde_c.servoL(
            target, self.args.move_speed, self.args.move_acc, self.args.servo_dt, self.args.lookahead, self.args.gain
        )
        if not ok:
            self.get_logger().warning("servoL was rejected by the controller", throttle_duration_sec=1.0)

    def _on_goto(self, msg: PoseStamped) -> None:
        self.get_logger().info("Moving to goto_ee_pose")
        self.last_target = None
        self.rtde_c.servoStop()
        ok = self.rtde_c.moveL(self._to_tcp(msg), self.args.move_speed, self.args.move_acc)  # blocks until done
        time.sleep(0.1)
        self.get_logger().info("Reached goto_ee_pose" if ok else "moveL failed")

    def shutdown(self) -> None:
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot_ip", required=True)
    parser.add_argument("--ns", default="/ur5e")
    parser.add_argument("--rate", type=float, default=125.0, help="TCP pose publishing rate [Hz].")
    parser.add_argument("--servo_dt", type=float, default=1 / 30, help="servoL time; match the policy rate.")
    parser.add_argument("--lookahead", type=float, default=0.1, help="servoL lookahead time [s] (0.03-0.2).")
    parser.add_argument("--gain", type=float, default=300.0, help="servoL proportional gain (100-2000).")
    parser.add_argument("--move_speed", type=float, default=0.1, help="moveL speed [m/s].")
    parser.add_argument("--move_acc", type=float, default=0.1, help="moveL acceleration [m/s^2].")
    parser.add_argument(
        "--max_jump",
        type=float,
        default=0.1,
        help="Max distance [m] between consecutive servo targets (first target: from the TCP).",
    )
    args = parser.parse_args()

    rclpy.init()
    node = UR5eNode(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

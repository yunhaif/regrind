"""Run a trained REGRIND policy on a UR5e + dexterous hand (WUJI or LEAP) through ROS 2.

Topics (all poses are geometry_msgs/PoseStamped in the UR5e base frame, quaternion w-x-y-z):
    <arm_ns>/ee_pose            in   TCP pose                        (ur5e_node.py)
    <arm_ns>/ee_command         out  streamed TCP target -> servoL   (ur5e_node.py)
    <arm_ns>/goto_ee_pose       out  one-shot TCP target -> moveL    (ur5e_node.py)
    <hand_ns>/joint_states      in   sensor_msgs/JointState          (wujihand driver / leap_hand_node.py)
    <hand_ns>/joint_commands    out  sensor_msgs/JointState          (wujihand driver / leap_hand_node.py)
    --object_pose_topic         in   object pose (mocap or a pose estimator)
    --object_joint_topic        in   std_msgs/Float64, articulated objects only (e.g. scissors)

The UR5e TCP must be configured as the hand root frame used in simulation, and the robot base axes must be
aligned with the simulation world axes, so that  p_robot = p_sim + sim2real_offset  and orientations are equal.
The offset is stored per setup in <run_dir>/real/setup.yaml (written with --save_setup) and shared with vis.py.

Example (the first run stores the offset; later runs and vis.py read it):
    python scripts/real_world/deploy.py --hand wuji --checkpoint logs/rsl_rl/wujihand_screwdriver/<run>/model_20000.pt \
        --sim2real_offset 0.0 -0.5 0.083 --save_setup
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from policy import RegrindPolicy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

HAND_DEFAULTS = {
    "wuji": {"hand_ns": "/hand_0", "min_z": 0.018},
    "leap": {"hand_ns": "/leap_hand", "min_z": 0.020},
}


def add_setup_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for the sim-to-real offset, shared by deploy.py, vis.py and fake_robot.py."""
    parser.add_argument(
        "--sim2real_offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Position of the sim world origin in the UR5e base frame [m]; overrides the setup file.",
    )
    parser.add_argument(
        "--setup", default=None, help="Setup file with the offset (default: <run_dir>/real/setup.yaml)."
    )
    parser.add_argument("--save_setup", action="store_true", help="Write --sim2real_offset to the setup file.")


def resolve_sim2real_offset(args) -> np.ndarray:
    """--sim2real_offset if given (optionally saved with --save_setup), otherwise the value in the setup file."""
    path = args.setup or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "real", "setup.yaml")
    if args.sim2real_offset is not None:
        offset = np.asarray(args.sim2real_offset, dtype=float)
        if args.save_setup:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                yaml.safe_dump({"sim2real_offset": offset.tolist()}, f, default_flow_style=None)
            print(f"[setup] saved sim2real_offset {offset.tolist()} to {path}")
        return offset
    if args.save_setup:
        raise SystemExit("--save_setup needs --sim2real_offset X Y Z.")
    if not os.path.isfile(path):
        raise SystemExit(f"No sim2real offset: pass --sim2real_offset X Y Z (add --save_setup to store it in {path}).")
    with open(path) as f:
        offset = np.asarray(yaml.safe_load(f)["sim2real_offset"], dtype=float)
    print(f"[setup] sim2real_offset {offset.tolist()} from {path}")
    return offset


def pose_to_msg(pos: np.ndarray, quat_wxyz: np.ndarray, frame: str) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pos)
    o = msg.pose.orientation
    o.w, o.x, o.y, o.z = map(float, quat_wxyz)
    return msg


def msg_to_pose(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    p, o = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z])


def _strip(name: str) -> str:
    return name.removeprefix("right_")


class RobotInterface(Node):
    """Caches the latest sensor messages and publishes commands. Spun in a background thread."""

    def __init__(self, args, joint_names: list[str], commands: bool = True, node_name: str = "regrind_deploy"):
        super().__init__(node_name)
        self.joint_names = joint_names
        self.frame = args.arm_ns.strip("/") + "_base"
        self.ee_pose = self.joints = self.obj_pose = self.obj_joint = None

        qos = qos_profile_sensor_data  # best effort, matches the drivers
        self.create_subscription(
            PoseStamped, f"{args.arm_ns}/ee_pose", lambda m: setattr(self, "ee_pose", msg_to_pose(m)), qos
        )
        self.create_subscription(JointState, f"{args.hand_ns}/joint_states", self._on_joints, qos)
        self.create_subscription(
            PoseStamped, args.object_pose_topic, lambda m: setattr(self, "obj_pose", msg_to_pose(m)), qos
        )
        self.create_subscription(Float64, args.object_joint_topic, lambda m: setattr(self, "obj_joint", m.data), qos)
        if commands:
            self.ee_cmd_pub = self.create_publisher(PoseStamped, f"{args.arm_ns}/ee_command", qos)
            self.goto_pub = self.create_publisher(PoseStamped, f"{args.arm_ns}/goto_ee_pose", 10)
            self.hand_cmd_pub = self.create_publisher(JointState, f"{args.hand_ns}/joint_commands", qos)

    def _on_joints(self, msg: JointState) -> None:
        if msg.name:  # reorder by name (drivers may or may not use the "right_" prefix)
            index = {_strip(n): i for i, n in enumerate(msg.name)}
            self.joints = np.array([msg.position[index[_strip(n)]] for n in self.joint_names])
        else:
            self.joints = np.array(msg.position)

    def send_arm(self, pos, quat, goto: bool = False) -> None:
        msg = pose_to_msg(pos, quat, self.frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        (self.goto_pub if goto else self.ee_cmd_pub).publish(msg)

    def send_hand(self, joints) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [float(q) for q in joints]
        self.hand_cmd_pub.publish(msg)

    def wait_for(self, attrs: list[str], timeout: float = 30.0) -> None:
        t0 = time.time()
        while any(getattr(self, a) is None for a in attrs):
            if time.time() - t0 > timeout:
                missing = [a for a in attrs if getattr(self, a) is None]
                raise TimeoutError(f"No messages received for {missing}")
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to model_XXXX.pt inside the training run dir.")
    parser.add_argument("--hand", choices=list(HAND_DEFAULTS), required=True)
    add_setup_args(parser)
    parser.add_argument("--retargeted_traj", default=None, help="Override commands.motion.retargeted_traj_path.")
    parser.add_argument("--arm_ns", default="/ur5e")
    parser.add_argument("--hand_ns", default=None, help="Default: /hand_0 (wuji) or /leap_hand (leap).")
    parser.add_argument("--object_pose_topic", default="/object_pose")
    parser.add_argument("--object_joint_topic", default="/object_joint")
    parser.add_argument(
        "--no_traj_aug", action="store_true", help="Do not shift the reference to the measured initial object pose."
    )
    parser.add_argument("--clamp_traj_aug", action="store_true", help="Clamp that shift to the training range.")
    parser.add_argument("--min_z", type=float, default=None, help="Lowest allowed TCP height in the robot frame [m].")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default=None, help="Where to save the episode log (default: <run_dir>/real).")
    args = parser.parse_args()
    args.hand_ns = args.hand_ns or HAND_DEFAULTS[args.hand]["hand_ns"]
    min_z = args.min_z if args.min_z is not None else HAND_DEFAULTS[args.hand]["min_z"]
    offset = resolve_sim2real_offset(args)

    policy = RegrindPolicy(args.checkpoint, args.retargeted_traj, args.device)
    if args.no_traj_aug:
        policy.traj_aug = False
    print(
        f"[deploy] {len(policy.joint_names)} finger joints, {policy.episode_len} steps @ {1 / policy.dt:.0f} Hz, "
        f"articulated={policy.is_articulated}, traj_aug={policy.traj_aug}"
    )

    rclpy.init()
    robot = RobotInterface(args, policy.joint_names)
    executor = SingleThreadedExecutor()
    executor.add_node(robot)
    threading.Thread(target=executor.spin, daemon=True).start()

    log = []
    try:
        robot.wait_for(["ee_pose", "joints", "obj_pose"] + (["obj_joint"] if policy.is_articulated else []))

        # 1. place the object
        ref_pos, ref_quat, ref_joint = policy.object_target(0)
        print(
            f"[deploy] reference initial object pose (robot frame): pos={np.round(ref_pos + offset, 4)}, "
            f"quat_wxyz={np.round(ref_quat, 4)}"
            + (f", joint={ref_joint:.3f}" if policy.is_articulated else "")
        )
        input("Place the object and press Enter...")
        obj_pos, obj_quat = robot.obj_pose
        policy.set_initial_object_pose(obj_pos - offset, obj_quat, clamp=args.clamp_traj_aug)

        # 2. move hand and arm to the (possibly shifted) start of the reference
        base = policy.base_target(0)
        start_pos = base[:3] + offset
        start_quat = R.from_euler("XYZ", base[3:6]).as_quat(scalar_first=True)
        assert start_pos[2] >= min_z, f"Start pose z={start_pos[2]:.3f} below min_z; check the sim2real offset."
        robot.send_hand(base[6:])
        time.sleep(1.0)
        print(f"[deploy] moving arm to start pose {np.round(start_pos, 4)}")
        while robot.goto_pub.get_subscription_count() == 0:
            time.sleep(0.05)
        robot.send_arm(start_pos, start_quat, goto=True)
        t0 = time.time()
        while np.linalg.norm(robot.ee_pose[0] - start_pos) > 0.003:
            if time.time() - t0 > 30.0:
                raise TimeoutError("Arm did not reach the start pose.")
            time.sleep(0.05)
        time.sleep(0.5)
        policy.warmup()
        input("Press Enter to start the policy (Ctrl-C to stop)...")

        # 3. closed-loop control; arm and hand targets are published together every step
        next_tick = time.perf_counter()
        for t in range(policy.episode_len):
            ee_pos, ee_quat = robot.ee_pose
            obj_pos, obj_quat = robot.obj_pose
            tgt_pos, tgt_quat, tgt_joints, obs, action = policy.step(
                t, ee_pos - offset, ee_quat, robot.joints, obj_pos - offset, obj_quat, robot.obj_joint
            )
            tgt_pos = tgt_pos + offset
            if tgt_pos[2] < min_z:
                print(f"[deploy] t={t}: clamping target z {tgt_pos[2]:.4f} -> {min_z}")
                tgt_pos[2] = min_z
            robot.send_arm(tgt_pos, tgt_quat)
            robot.send_hand(tgt_joints)
            log.append(
                dict(
                    obs=obs,
                    action=action,
                    target_pos=tgt_pos,
                    target_quat=tgt_quat,
                    target_joints=tgt_joints,
                    ee_pos=ee_pos,
                    ee_quat=ee_quat,
                    joints=robot.joints.copy(),
                    obj_pos=obj_pos,
                    obj_quat=obj_quat,
                )
            )

            next_tick += policy.dt
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                print(f"[deploy] t={t}: control step overran by {-slack * 1e3:.1f} ms")
        print("[deploy] episode finished")
    except KeyboardInterrupt:
        print("\n[deploy] interrupted")
    finally:
        if log:
            out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "real")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, time.strftime("episode_%Y-%m-%d_%H-%M-%S.npz"))
            np.savez(path, **{k: np.stack([s[k] for s in log]) for k in log[0]})
            print(f"[deploy] saved {len(log)} steps to {path}")
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

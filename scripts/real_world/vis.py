"""Browser-based 3D view of the real-world setup (viser), for placing the object and checking frames.

Shows, in the UR5e base frame (open http://localhost:<port>):
  * the hand at the measured TCP pose with the measured finger joints,
  * the object (blue) at the measured pose, including the scissor joint for articulated objects,
  * the reference start pose of the object (green): place the real object onto it,
  * a translucent preview of the reference hand and object at step t (slider). t = 0 is the start pose that
    deploy.py moves the arm to; with trajectory augmentation it is shifted to the measured object pose,
plus the live offset of the object from the reference and whether training augmentation covered it.

It only listens to the same topics as deploy.py, so it can run alongside it. The sim-to-real offset is read
from the same setup file as deploy.py (<run_dir>/real/setup.yaml); --sim2real_offset overrides it.

Example:
    python scripts/real_world/vis.py --hand leap --checkpoint <run_dir>/model_XXXX.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import viser
from deploy import (
    HAND_DEFAULTS,
    RobotInterface,
    add_setup_args,
    resolve_sim2real_offset,
)
from policy import RegrindPolicy
from rclpy.executors import SingleThreadedExecutor
from regrind.assets import REGRIND_ASSETS_DIR
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

HAND_URDF = {
    "wuji": "mjcf/free_wuji_right_hand_full_simplified_v1.urdf",
    "leap": "leaphand_urdf/leap_hand_right.urdf",
}
# Object USD used in training (scene.object.spawn.usd_path) -> URDF with the same frame and scale.
OBJECT_URDF = {
    "scissors_flattened": "scissors/scissors.urdf",
    "scissors_2x_3d_printed_flattened": "scissors/scissors_2x.urdf",
    "screwdriver_flattened": "screwdriver/screwdriver.urdf",
    "screwdriver_1_5x_3d_printed_flattened": "screwdriver/screwdriver_1.5x.urdf",
}
BLUE, GREEN, PREVIEW = (0.2, 0.5, 1.0), (0.1, 0.85, 0.2, 0.5), (1.0, 0.75, 0.1, 0.35)


class Model:
    """A URDF placed at a pose. Joint values are given in `joint_names` order and mapped onto the URDF."""

    def __init__(self, server: viser.ViserServer, name: str, urdf: Path, joint_names=(), color=None):
        self.frame = server.scene.add_frame(name, axes_length=0.05, axes_radius=0.002, visible=False)
        self.urdf = ViserUrdf(server, urdf, root_node_name=name, mesh_color_override=color)
        urdf_names = self.urdf.get_actuated_joint_names()
        if set(urdf_names) <= set(joint_names):  # same names (e.g. WUJI)
            self.index = [list(joint_names).index(n) for n in urdf_names]
        else:  # same order (e.g. LEAP: URDF joints "0".."15" are the motor IDs)
            assert len(urdf_names) in (0, len(joint_names)), f"Cannot map joints {joint_names} onto {urdf_names}"
            self.index = list(range(len(urdf_names)))

    def set(self, pos, quat_wxyz, joints=None) -> None:
        self.frame.position, self.frame.wxyz = np.asarray(pos, float), np.asarray(quat_wxyz, float)
        self.frame.visible = True
        if joints is not None and self.index:
            self.urdf.update_cfg(np.asarray(joints, float)[self.index])

    def show(self, visible: bool) -> None:
        self.frame.visible = visible
        self.urdf.show_visual = visible


def object_urdf(policy: RegrindPolicy, override: str | None) -> Path:
    if override:
        return Path(override)
    stem = Path(policy.env_cfg["scene"]["object"]["spawn"]["usd_path"]).stem
    if stem not in OBJECT_URDF:
        raise KeyError(f"No URDF known for object USD '{stem}'; pass --object_urdf.")
    return REGRIND_ASSETS_DIR / OBJECT_URDF[stem]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hand", choices=list(HAND_DEFAULTS), required=True)
    add_setup_args(parser)
    parser.add_argument("--retargeted_traj", default=None)
    parser.add_argument("--object_urdf", default=None, help="Override the object model inferred from env.yaml.")
    parser.add_argument("--arm_ns", default="/ur5e")
    parser.add_argument("--hand_ns", default=None)
    parser.add_argument("--object_pose_topic", default="/object_pose")
    parser.add_argument("--object_joint_topic", default="/object_joint")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args.hand_ns = args.hand_ns or HAND_DEFAULTS[args.hand]["hand_ns"]
    offset = resolve_sim2real_offset(args)

    policy = RegrindPolicy(args.checkpoint, args.retargeted_traj)
    obj_joint_names = ["object_joint"] if policy.is_articulated else []

    # -- scene
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1)
    server.scene.add_frame("/robot_base", axes_length=0.15, axes_radius=0.004)
    hand_urdf, obj_urdf = REGRIND_ASSETS_DIR / HAND_URDF[args.hand], object_urdf(policy, args.object_urdf)
    hand = Model(server, "/hand", hand_urdf, policy.joint_names)
    obj = Model(server, "/object", obj_urdf, obj_joint_names, BLUE)
    ref_obj = Model(server, "/reference_object", obj_urdf, obj_joint_names, GREEN)
    prev_hand = Model(server, "/preview_hand", hand_urdf, policy.joint_names, PREVIEW)
    prev_obj = Model(server, "/preview_object", obj_urdf, obj_joint_names, PREVIEW)

    # The reference start pose of the object (sim frame + offset), shown in green.
    ref_pos, ref_quat, ref_joint = policy.object_target(0)
    ref_obj.set(ref_pos + offset, ref_quat, [ref_joint] if policy.is_articulated else None)
    server.initial_camera.look_at = ref_pos + offset
    server.initial_camera.position = ref_pos + offset + np.array([0.5, 0.5, 0.4])

    with server.gui.add_folder("Object placement"):
        gui_status = server.gui.add_markdown("Waiting for the object pose...")
        gui_dxy = server.gui.add_text("dx, dy [mm]", "", disabled=True)
        gui_dyaw = server.gui.add_text("dyaw [deg]", "", disabled=True)
        gui_dz = server.gui.add_text("dz [mm]", "", disabled=True)
        gui_djoint = server.gui.add_text("djoint [rad]", "", disabled=True, visible=policy.is_articulated)
    with server.gui.add_folder("Reference preview"):
        gui_show = server.gui.add_checkbox("Show preview", True)
        gui_play = server.gui.add_checkbox("Play", False)
        gui_t = server.gui.add_slider("Step t", 0, policy.episode_len - 1, 1, 0)
    with server.gui.add_folder("Robot"):
        gui_tcp = server.gui.add_text("TCP [m]", "", disabled=True)
        gui_inputs = server.gui.add_markdown("")

    lo, hi, yaw = policy.aug_bounds
    rng = (
        f"training range: dx [{lo[0] * 1e3:.0f}, {hi[0] * 1e3:.0f}] mm, dy [{lo[1] * 1e3:.0f}, {hi[1] * 1e3:.0f}] mm,"
        f" dyaw [{np.degrees(yaw[0]):.0f}, {np.degrees(yaw[1]):.0f}] deg"
    )
    print(f"[vis] open http://localhost:{args.port}  (object model: {obj_urdf.name}, traj_aug={policy.traj_aug})")

    # -- ROS (listen only)
    rclpy.init()
    robot = RobotInterface(args, policy.joint_names, commands=False, node_name="regrind_vis")
    executor = SingleThreadedExecutor()
    executor.add_node(robot)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        while rclpy.ok():  # rclpy shuts the context down on SIGINT/SIGTERM
            t0 = time.time()
            if robot.ee_pose is not None:
                hand.set(*robot.ee_pose, robot.joints)
                gui_tcp.value = np.array2string(robot.ee_pose[0], precision=3)
            missing = [
                n
                for n, v in [
                    ("TCP pose", robot.ee_pose),
                    ("hand joints", robot.joints),
                    ("object pose", robot.obj_pose),
                ]
                if v is None
            ]
            if policy.is_articulated and robot.obj_joint is None:
                missing.append("object joint")
            gui_inputs.content = "All inputs received." if not missing else "Waiting for: " + ", ".join(missing)

            if robot.obj_pose is not None:
                pos, quat = robot.obj_pose
                obj.set(pos, quat, [robot.obj_joint or 0.0] if policy.is_articulated else None)
                dpos, dyaw, inside = policy.object_offset(pos - offset, quat)
                gui_dxy.value = f"{dpos[0] * 1e3:+.1f}, {dpos[1] * 1e3:+.1f}"
                gui_dyaw.value = f"{np.degrees(dyaw):+.1f}"
                gui_dz.value = f"{(pos[2] - offset[2] - ref_pos[2]) * 1e3:+.1f}"
                if policy.is_articulated and robot.obj_joint is not None:
                    gui_djoint.value = f"{robot.obj_joint - ref_joint:+.3f}"
                if policy.traj_aug:
                    gui_status.content = (
                        "**OK** - within the " if inside else "**Move the object** - outside the "
                    ) + rng
                    policy.set_initial_object_pose(pos - offset, quat, verbose=False)  # what deploy.py will do
                else:
                    gui_status.content = "Trained without augmentation: match the green pose as closely as possible."

            # Reference preview at step t (shifted by the measured object offset, like deploy.py).
            if gui_play.value:
                gui_t.value = (gui_t.value + 1) % policy.episode_len
            prev_hand.show(gui_show.value)
            prev_obj.show(gui_show.value)
            if gui_show.value:
                t = int(gui_t.value)
                base = policy.base_target(t)
                prev_hand.set(base[:3] + offset, R.from_euler("XYZ", base[3:6]).as_quat(scalar_first=True), base[6:])
                p, q, j = policy.object_target(t)
                prev_obj.set(p + offset, q, [j] if policy.is_articulated else None)

            time.sleep(max(0.0, policy.dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.try_shutdown()
        sys.stdout.flush()
        os._exit(0)  # viser runs non-daemon threads


if __name__ == "__main__":
    main()

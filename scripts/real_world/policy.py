"""Hardware-agnostic wrapper around an exported REGRIND policy.

Everything needed at deployment is read from the training run directory:

    <run_dir>/model_XXXX.pt                     (checkpoint; only its name is used)
    <run_dir>/exported/model_XXXX_policy.pt     (TorchScript policy, written by scripts/rsl_rl/play.py)
    <run_dir>/params/env.yaml                   (env config dumped at training time)

plus the retargeted trajectory referenced by ``commands.motion.retargeted_traj_path``.

All quantities here are in the *simulation* frame. The caller converts robot-frame measurements
to the sim frame (see ``deploy.py``).
"""

from __future__ import annotations

import math
import os
import re

import numpy as np
import torch
import yaml
from regrind.data.arctic import load_retargeted_traj
from regrind.utils.traj_aug_numpy import (
    compute_augmented_object_state_at_timestep,
    compute_base_qpos_at_timestep,
    compute_traj_aug_step_bounds,
    measure_traj_aug_delta_from_poses,
)
from scipy.spatial.transform import Rotation as R


def rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix, row-major (matches ``mdp.object_ori``)."""
    return R.from_quat(quat_wxyz, scalar_first=True).as_matrix()[:, :2].reshape(-1)


def _match_regex_dict(names: list[str], table, default: float | None = None) -> np.ndarray:
    """Resolve an IsaacLab ``{regex: value}`` dict (or scalar) into a per-joint vector."""
    if not isinstance(table, dict):
        return np.full(len(names), float(table), dtype=np.float64)
    out = np.empty(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        for pattern, value in table.items():
            if re.fullmatch(pattern, name):
                out[i] = float(value)
                break
        else:
            if default is None:
                raise KeyError(f"No entry for joint '{name}' in {list(table)}")
            out[i] = default
    return out


class RegrindPolicy:
    """Builds policy observations and converts actions into wrist-pose / finger-joint targets."""

    def __init__(self, checkpoint: str, retargeted_traj: str | None = None, device: str = "cpu"):
        run_dir = os.path.dirname(os.path.abspath(checkpoint))
        model_name = os.path.splitext(os.path.basename(checkpoint))[0]
        exported = os.path.join(run_dir, "exported", f"{model_name}_policy.pt")
        if not os.path.isfile(exported):
            raise FileNotFoundError(f"{exported} not found. Run scripts/rsl_rl/play.py once to export the policy.")
        self.device = device
        self.policy = torch.jit.load(exported, map_location=device).eval()

        with open(os.path.join(run_dir, "params", "env.yaml")) as f:
            cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.env_cfg = cfg

        # -- timing: phi = t / episode_len, exactly as MotionCommand.phi
        self.dt = cfg["sim"]["dt"] * cfg["decimation"]
        self.episode_len = math.ceil(cfg["episode_length_s"] / self.dt)

        # -- actions: [wrist delta pos (3), wrist delta rotvec (3), finger deltas (n)]
        root_cfg, joint_cfg = cfg["actions"]["root_pose"], cfg["actions"]["joint_pos"]
        for term in (root_cfg, joint_cfg):
            assert (
                term["base_action_source"] == "motion_target"
            ), "Only base_action_source='motion_target' is supported."
        self.joint_names: list[str] = list(joint_cfg["joint_names"])
        self.num_fingers = len(self.joint_names)
        self.action_scale = np.concatenate([
            np.full(3, root_cfg["scale_pos"]),
            np.full(3, root_cfg["scale_rot"]),
            _match_regex_dict(self.joint_names, joint_cfg["scale"]),
        ])
        # hand_joint_pos observation is relative to the articulation's default joint positions, which the
        # `init_zero_joint_default_pos` startup event resets to zero (otherwise they come from init_state).
        relative = cfg["observations"]["policy"]["hand_joint_pos"]["params"].get("relative_to_default", False)
        if not relative or cfg["events"].get("init_zero_joint_default_pos") is not None:
            self.default_joint_pos = np.zeros(self.num_fingers)
        else:
            self.default_joint_pos = _match_regex_dict(
                self.joint_names, cfg["scene"]["robot"]["init_state"]["joint_pos"], default=0.0
            )

        # -- observation layout (see ObservationsCfg.PolicyCfg in dexterous_env_cfg.py)
        obs_cfg = cfg["observations"]["policy"]
        self.is_articulated = obs_cfg.get("object_joint") is not None
        for term in ("action_base_wrist_pos_and_rot6d", "action_base_hand_joint_pos"):
            assert obs_cfg.get(term) is not None, f"Policy obs term '{term}' missing; checkpoint is not supported."

        # -- reference motion (raw retargeted trajectory, interpolated by episode progress)
        motion = cfg["commands"]["motion"]
        path = retargeted_traj or os.path.expandvars(motion["retargeted_traj_path"])
        if not os.path.isfile(path) and "REGRIND_DATA_DIR" in os.environ and "/retargeted_traj/" in path:
            # trained on another machine: re-root the path under the local data dir
            path = os.path.join(
                os.environ["REGRIND_DATA_DIR"], "retargeted_traj", path.split("/retargeted_traj/", 1)[1]
            )
        print(f"[policy] reference motion: {path}")
        raw = load_retargeted_traj(path)
        keys = ["object_pos", "object_quat", "robot_pos", "robot_quat", "robot_euler_XYZ", "robot_joints"]
        if self.is_articulated:
            keys.append("object_joint")
        self.ref = {k: np.asarray(raw[k], dtype=np.float64) for k in keys}
        assert self.ref["robot_joints"].shape[1] == self.num_fingers

        # -- trajectory augmentation: shift the reference to the measured initial object pose
        self.traj_aug = bool(motion.get("traj_aug_enabled", False)) and bool(motion.get("traj_aug_apply_to_hand", True))
        self.aug_start, self.aug_use = compute_traj_aug_step_bounds(
            self.episode_len, motion.get("traj_aug_start", 0.0), motion.get("traj_aug_use", 1.0)
        )
        self.aug_bounds = (
            tuple(motion.get("traj_aug_translation_lower_bound", (0.0, 0.0, 0.0))),
            tuple(motion.get("traj_aug_translation_upper_bound", (0.0, 0.0, 0.0))),
            tuple(motion.get("traj_aug_yaw_range", (0.0, 0.0))),
        )
        self.aug_delta_pos = np.zeros(3)
        self.aug_delta_yaw = 0.0
        self.reset()

    # ------------------------------------------------------------------ reference
    def _frac(self, t: int) -> float:
        # MotionCommand resamples the demo to exactly `episode_len` env steps, so step t is at t / (len - 1).
        return min(t, self.episode_len - 1) / (self.episode_len - 1)

    def base_target(self, t: int) -> np.ndarray:
        """Reference [wrist_pos(3), wrist_euler_XYZ(3), fingers(n)] at step t (the action base)."""
        return compute_base_qpos_at_timestep(
            self.ref, self._frac(t), float(t), self.aug_start, self.aug_use, self.aug_delta_pos, self.aug_delta_yaw
        ).astype(np.float64)

    def object_target(self, t: int) -> tuple[np.ndarray, np.ndarray, float | None]:
        """Reference object (pos, quat_wxyz, joint) at step t."""
        return compute_augmented_object_state_at_timestep(
            self.ref, self._frac(t), float(t), self.aug_start, self.aug_use, self.aug_delta_pos, self.aug_delta_yaw
        )

    def object_offset(self, obj_pos: np.ndarray, obj_quat: np.ndarray, clamp: bool = False):
        """Planar offset (dpos with dz=0, dyaw) of an object pose from the reference start, and whether training
        trajectory augmentation covered it."""
        ref_pos, ref_quat = self.ref["object_pos"][0], self.ref["object_quat"][0]
        dpos, dyaw = measure_traj_aug_delta_from_poses(obj_pos, obj_quat, ref_pos, ref_quat, *self.aug_bounds, clamp)
        lo, hi, yaw = self.aug_bounds
        inside = bool(np.all(dpos[:2] >= np.array(lo[:2])) and np.all(dpos[:2] <= np.array(hi[:2])))
        inside = inside and yaw[0] <= dyaw <= yaw[1]
        return dpos, dyaw, inside

    def set_initial_object_pose(self, obj_pos, obj_quat, clamp: bool = False, verbose: bool = True) -> None:
        """Measure the SE(2) offset of the real object w.r.t. the reference and shift the reference by it."""
        if not self.traj_aug:
            return
        self.aug_delta_pos, self.aug_delta_yaw, inside = self.object_offset(obj_pos, obj_quat, clamp)
        if verbose:
            print(
                f"[policy] object offset: dxy={self.aug_delta_pos[:2].round(4)}, dyaw={self.aug_delta_yaw:.3f} rad"
                + ("" if inside else "  (WARNING: outside training range)")
            )

    # ------------------------------------------------------------------ control
    def reset(self) -> None:
        self.prev_action = np.zeros(6 + self.num_fingers, dtype=np.float32)
        self._hist: list[np.ndarray] | None = None

    @torch.inference_mode()
    def warmup(self, n: int = 10) -> None:
        obs, _ = self._build_obs(
            0, np.zeros(3), np.array([1.0, 0, 0, 0]), self.default_joint_pos, np.zeros(3), np.array([1.0, 0, 0, 0]), 0.0
        )
        for _ in range(n):
            self.policy(torch.from_numpy(obs).to(self.device))
        self.reset()

    def _build_obs(self, t, wrist_pos, wrist_quat, joints, obj_pos, obj_quat, obj_joint):
        # history terms (length 2, oldest first); on the first step both entries are the current sample
        cur = [wrist_pos, rot6d(wrist_quat), np.asarray(joints) - self.default_joint_pos]
        prev = cur if self._hist is None else self._hist
        base = self.base_target(t)
        base_quat = R.from_euler("XYZ", base[3:6]).as_quat(scalar_first=True)
        terms = [obj_pos, rot6d(obj_quat)]
        if self.is_articulated:
            terms.append([obj_joint])
        for p, c in zip(prev, cur):
            terms += [p, c]
        terms += [self.prev_action, [t / self.episode_len], base[:3], rot6d(base_quat), base[6:]]
        obs = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in terms])
        return obs, (cur, base)

    @torch.inference_mode()
    def step(self, t, wrist_pos, wrist_quat, joints, obj_pos, obj_quat, obj_joint=None):
        """One control step. Inputs are sim-frame measurements; quaternions are (w, x, y, z).

        Returns (target_wrist_pos, target_wrist_quat_wxyz, target_finger_joints, obs, raw_action).
        """
        obs, (cur, base) = self._build_obs(t, wrist_pos, wrist_quat, joints, obj_pos, obj_quat, obj_joint)
        action = self.policy(torch.from_numpy(obs).to(self.device)).reshape(-1).cpu().numpy()
        self._hist = cur
        self.prev_action = action.astype(np.float32)  # last_action obs uses the unclipped action

        # Same composition as SE3ImpedanceActionTerm / ClippedRelativeJointPositionAction
        delta = np.clip(action, -1.0, 1.0) * self.action_scale
        target_pos = base[:3] + delta[:3]
        target_rot = R.from_rotvec(delta[3:6]) * R.from_euler("XYZ", base[3:6])
        target_joints = base[6:] + delta[6:]
        return target_pos, target_rot.as_quat(scalar_first=True), target_joints, obs, action

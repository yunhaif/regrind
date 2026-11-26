"""NumPy/SciPy helpers mirroring MotionCommand trajectory augmentation (commands.py)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp


def compute_traj_aug_step_bounds(demo_len: int, start_frac: float, use_frac: float) -> tuple[int, int]:
    """Match MotionCommand._sample_traj_aug start_step / use_step from demo length and fractions."""
    max_offset = max(demo_len - 1, 1)
    start_scalar = min(max(float(start_frac), 0.0), 1.0)
    use_scalar = min(max(float(use_frac), 0.0), 1.0)
    start_step = int(start_scalar * max_offset)
    start_step = min(start_step, max(0, max_offset - 1))
    use_step = int(use_scalar * max_offset)
    use_step = max(use_step, start_step + 1)
    use_step = min(use_step, max_offset)
    return start_step, use_step


def traj_aug_weight(t_in_demo: float, start_step: int, use_step: int) -> float:
    """Same piecewise schedule as MotionCommand._traj_aug_weight (scalar)."""
    blend_len = max(use_step - start_step, 1)
    if t_in_demo < start_step:
        return 1.0
    phase = (t_in_demo - float(start_step)) / float(blend_len)
    w = 1.0 - phase
    return float(np.clip(w, 0.0, 1.0))


def _interp_rows(arr: np.ndarray, phi: float) -> np.ndarray:
    n = arr.shape[0]
    if n == 1:
        return arr[0].astype(np.float64, copy=True)
    f = float(phi) * float(n - 1)
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, n - 1)
    alpha = f - float(i0)
    return ((1.0 - alpha) * arr[i0] + alpha * arr[i1]).astype(np.float64, copy=False)


def interp_quat_wxyz_slerp(quats_wxyz: np.ndarray, phi: float) -> np.ndarray:
    """Interpolate unit quaternions (w, x, y, z) along demo phase phi in [0, 1]."""
    n = quats_wxyz.shape[0]
    if n == 1:
        q = quats_wxyz[0].astype(np.float64, copy=True)
        return q / (np.linalg.norm(q) + 1e-12)
    f = float(phi) * float(n - 1)
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, n - 1)
    alpha = f - float(i0)
    rots = R_scipy.from_quat(np.stack([quats_wxyz[i0], quats_wxyz[i1]]), scalar_first=True)
    slerp = Slerp([0.0, 1.0], rots)
    q_out = slerp(alpha).as_quat(scalar_first=True)
    return np.asarray(q_out, dtype=np.float64).reshape(4)


def reference_state_at_phi(raw: dict[str, np.ndarray], phi: float) -> dict[str, np.ndarray]:
    """Unaugmented retargeted reference at phase phi (linear index in raw trajectory)."""
    if "robot_quat" in raw:
        # Slerp the wrist orientation like MotionCommand does; linearly interpolating Euler angles breaks near
        # gimbal lock (pitch ~ +-pi/2), where roll/yaw jump between neighbouring frames.
        robot_quat = interp_quat_wxyz_slerp(raw["robot_quat"], phi)
        robot_euler = R_scipy.from_quat(robot_quat, scalar_first=True).as_euler("XYZ", degrees=False)
    else:
        robot_euler = _interp_rows(raw["robot_euler_XYZ"], phi)
    return {
        "object_pos": _interp_rows(raw["object_pos"], phi),
        "object_quat": interp_quat_wxyz_slerp(raw["object_quat"], phi),
        "robot_pos": _interp_rows(raw["robot_pos"], phi),
        "robot_euler_XYZ": robot_euler,
        "robot_joints": _interp_rows(raw["robot_joints"], phi),
    }


def _wrap_to_pi(angle: float) -> float:
    """Map angle (rad) to (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _world_z_heading_rad(quat_wxyz: np.ndarray) -> float:
    """World +Z compass heading (rad) from body +x projected onto fixed world XY.

    With world-from-body rotation matrix R (columns are body axes in world), projection
    of body +x onto the horizontal plane yields atan2(x_y, x_x). Matches "spin about world
    +Z" used in traj augmentation when pose changes are planar.
    """
    r = R_scipy.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True)
    m = r.as_matrix()
    x_w = m[:, 0].astype(np.float64, copy=False)
    rho = float(np.hypot(float(x_w[0]), float(x_w[1])))
    if rho < 1e-10:
        y_w = m[:, 1]
        return float(np.arctan2(float(y_w[1]), float(y_w[0])))
    return float(np.arctan2(float(x_w[1]), float(x_w[0])))


def planar_relative_yaw(quat_a_wxyz: np.ndarray, quat_b_wxyz: np.ndarray) -> float:
    """Heading difference about fixed world +Z axis (wrapped to (-pi, pi]).

    ``quat_*`` orient world-from-body like Isaac / retargeted H5.

    Returned value is heading(quat_a) - heading(quat_b), consistent with traj-aug left-multiply
    by rotation about ``[0, 0, 1]``: it is **not** ``atan2`` of entries of ``R_a R_b^{-1}``, which is
    not a world-Z angle when tilt relative to gravity differs between the two poses.
    """
    yaw_a = _world_z_heading_rad(quat_a_wxyz)
    yaw_b = _world_z_heading_rad(quat_b_wxyz)
    return _wrap_to_pi(yaw_a - yaw_b)


def measure_traj_aug_delta_from_poses(
    obj_pos_cur_sim: np.ndarray,
    obj_quat_cur_wxyz: np.ndarray,
    ref_obj_pos: np.ndarray,
    ref_obj_quat_wxyz: np.ndarray,
    pos_lower: tuple[float, float, float],
    pos_upper: tuple[float, float, float],
    yaw_range: tuple[float, float],
    clamp: bool,
) -> tuple[np.ndarray, float]:
    """SE(2)-style delta: xy translation in world/sim frame; yaw = world +Z heading difference; z delta forced to 0."""
    delta_pos = np.zeros(3, dtype=np.float64)
    delta_pos[:2] = np.asarray(obj_pos_cur_sim[:2], dtype=np.float64) - np.asarray(ref_obj_pos[:2], dtype=np.float64)
    delta_yaw = planar_relative_yaw(obj_quat_cur_wxyz, ref_obj_quat_wxyz)
    if clamp:
        lb = np.asarray(pos_lower, dtype=np.float64)
        ub = np.asarray(pos_upper, dtype=np.float64)
        delta_pos[0] = float(np.clip(delta_pos[0], lb[0], ub[0]))
        delta_pos[1] = float(np.clip(delta_pos[1], lb[1], ub[1]))
        delta_pos[2] = float(np.clip(delta_pos[2], lb[2], ub[2]))
        delta_yaw = float(np.clip(delta_yaw, yaw_range[0], yaw_range[1]))
    return delta_pos.astype(np.float32), delta_yaw


def augment_wrist_to_base(
    ref_obj_pos: np.ndarray,
    ref_obj_quat_wxyz: np.ndarray,
    wrist_pos: np.ndarray,
    wrist_euler_xyz_intrinsic: np.ndarray,
    delta_pos: np.ndarray,
    delta_yaw: float,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply same rigid-body augmentation as _apply_traj_aug_to_points + target_hand_wrist_quat."""
    yaw_w = float(delta_yaw) * float(weight)
    dpos = np.asarray(delta_pos, dtype=np.float64) * float(weight)
    r_delta = R_scipy.from_rotvec(np.array([0.0, 0.0, yaw_w], dtype=np.float64))
    obj_p = np.asarray(ref_obj_pos, dtype=np.float64)
    wrist_p = np.asarray(wrist_pos, dtype=np.float64)
    wrist_aug = r_delta.apply(wrist_p - obj_p) + obj_p + dpos
    r_wrist = R_scipy.from_euler("XYZ", wrist_euler_xyz_intrinsic, degrees=False)
    r_aug = r_delta * r_wrist
    euler_aug = r_aug.as_euler("XYZ", degrees=False)
    return wrist_aug.astype(np.float32), euler_aug.astype(np.float32)


def compute_augmented_object_state_at_timestep(
    raw: dict[str, np.ndarray],
    phi: float,
    t_in_demo: float,
    start_step: int,
    use_step: int,
    delta_pos: np.ndarray,
    delta_yaw: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Traj-aug target object pose at (phi, t), matching MotionCommand._target_object_pos/quat_at."""
    ref = reference_state_at_phi(raw, phi)
    w = traj_aug_weight(t_in_demo, start_step, use_step)
    pos_aug = np.asarray(ref["object_pos"], dtype=np.float64) + np.asarray(delta_pos, dtype=np.float64) * w
    yaw_w = float(delta_yaw) * w
    r_delta = R_scipy.from_rotvec(np.array([0.0, 0.0, yaw_w], dtype=np.float64))
    r_obj = R_scipy.from_quat(np.asarray(ref["object_quat"], dtype=np.float64), scalar_first=True)
    quat_aug = r_delta * r_obj
    quat_aug = quat_aug.as_quat(scalar_first=True)
    ref_joint = None
    if "object_joint" in raw:
        joint = _interp_rows(raw["object_joint"], phi)
        ref_joint = float(np.asarray(joint, dtype=np.float64).reshape(-1)[0])
    return pos_aug.astype(np.float64), np.asarray(quat_aug, dtype=np.float64), ref_joint


def compute_base_qpos_at_timestep(
    raw: dict[str, np.ndarray],
    phi: float,
    t_in_demo: float,
    start_step: int,
    use_step: int,
    delta_pos: np.ndarray,
    delta_yaw: float,
) -> np.ndarray:
    """Full motion base qpos [wrist_pos(3), wrist_euler_intrinsic_XYZ(3), finger joints]."""
    ref = reference_state_at_phi(raw, phi)
    w = traj_aug_weight(t_in_demo, start_step, use_step)
    wrist_p, euler = augment_wrist_to_base(
        ref["object_pos"],
        ref["object_quat"],
        ref["robot_pos"],
        ref["robot_euler_XYZ"],
        delta_pos,
        delta_yaw,
        w,
    )
    joints = np.asarray(ref["robot_joints"], dtype=np.float32).reshape(-1)
    return np.concatenate([wrist_p, euler, joints], axis=0).astype(np.float32)

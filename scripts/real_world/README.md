# Real-world deployment

A minimal setup for running a trained REGRIND policy on a **UR5e arm + WUJI hand** or a **UR5e arm + LEAP Hand**.
These scripts are a cleaned-up version of what we used for the paper's experiments. Expect to calibrate and
tune them for your own cell.

| File | Role |
|------|------|
| `policy.py` | Hardware-agnostic. Loads the exported policy and `params/env.yaml`, rebuilds the reference motion, builds observations, and turns actions into targets. |
| `deploy.py` | ROS 2 node that runs the 30 Hz control loop. |
| `ur5e_node.py` | ROS 2 bridge for the UR5e over [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/). |
| `leap_hand_node.py` | ROS 2 bridge for the LEAP Hand. It uses the official [LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API). |
| `vis.py` | Browser 3D view ([viser](https://viser.studio/)) of the hand, the object and the reference, for placing the object and checking frames. |
| `fake_robot.py` | Simulated arm, hand and object on the same topics, for dry runs without hardware. |

The WUJI hand uses the official driver, [wujihandros2](https://github.com/wuji-technology/wujihandros2).

## How it works

```
 object pose (mocap / pose estimator) ─┐
 UR5e TCP pose ────────────────────────┼─► deploy.py ──► policy ──► wrist SE(3) target ──► ur5e_node.py ──► servoL
 hand joint positions ─────────────────┘   (30 Hz)                └► finger targets ────► hand driver
```

* **Actions are residuals on the retargeted reference.** At step `t` the policy outputs `a ∈ [-1, 1]^(6+n)`. The targets are:
  * wrist position: `p = p_ref(t) + s_pos · a[0:3]`
  * wrist rotation: `R = Exp(s_rot · a[3:6]) · R_ref(t)`
  * fingers: `q = q_ref(t) + s_q · a[6:]`

  This is the same composition as the simulation action terms. The scales come from `env.yaml`.
* **Wrist SE(3) target → UR5e.** The target pose is streamed to the UR controller with `servoL`. This is
  Cartesian servoing, and the UR controller solves the IK. The defaults are `time = 1/30 s`, `lookahead = 0.1 s`
  and `gain = 300`. The policy was trained with a Cartesian impedance-controlled floating hand, so
  Cartesian servoing on the TCP is the closest match. The start pose is reached with `moveL`.
* **Arm/hand synchronization.** Each control step publishes the arm target and the finger targets back to back.
  Both use best-effort QoS with depth 1. The arm and hand controllers then track the latest target independently,
  with no hardware-level synchronization. Training randomizes observation delays, which makes the
  policy tolerant to this kind of latency.
* **Reference motion.** The reference is interpolated from the retargeted trajectory at progress
  `t / (episode_len - 1)`, matching how `MotionCommand` resamples the demo to the episode length. If the
  checkpoint was trained with trajectory augmentation, the measured initial object pose shifts the reference by
  an SE(2) offset, as in training. Disable this with `--no_traj_aug`.

## Setup

1. **Export the policy** (this writes `<run_dir>/exported/model_XXXX_policy.pt`):
   ```bash
   python scripts/rsl_rl/play.py --task [task] --headless --checkpoint <run_dir>/model_XXXX.pt
   ```
   Copy `<run_dir>` to the robot PC. Only `model_XXXX.pt`'s name, `exported/` and `params/env.yaml` are needed,
   plus the retargeted `.h5`. The `.h5` is found through `$REGRIND_DATA_DIR`, or you can pass it with `--retargeted_traj`.
2. **Robot PC environment.** ROS 2 comes from [RoboStack](https://robostack.github.io/) (conda), and the
   rest from the `hardware` extra. Isaac Sim is **not** needed.
   ```bash
   conda create -n regrind_hw -c conda-forge -c robostack-kilted ros-kilted-ros-base
   conda activate regrind_hw
   pip install -e "source/regrind[hardware]"
   ```
   The extra installs the default (CUDA) PyTorch build. The policy is a small MLP that runs on the CPU, so you can
   run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first to get the much smaller CPU build.
   For the LEAP Hand, also clone [LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API) and add its `python/`
   folder to `PYTHONPATH`. Then set the FTDI USB latency timer to 1 ms, as the LEAP Hand authors recommend
   (it resets when the adapter is re-plugged):
   ```bash
   echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   # ttyUSB0 = your LEAP adapter
   ```
3. **Frames.** Everything the policy sees is in the simulation frame. To map the robot frame onto it:
   * **Base axes.** Align the UR5e base axes with the sim world axes. Then `p_robot = p_sim + sim2real_offset`
     and orientations are identical. The offset is the position of the sim world origin in the UR5e base frame.
     It depends on your cell (where the table is relative to the robot) and on the demo's world frame, so it is
     stored per setup in `<run_dir>/real/setup.yaml`, which `deploy.py` and `vis.py` both read:
     ```yaml
     sim2real_offset: [0.0, -0.4, -0.845]
     ```
     Create it by passing `--sim2real_offset X Y Z --save_setup` to either script. `--sim2real_offset` alone
     overrides the file for one run, and `--setup <file>` points to a different file.
   * **TCP.** On the teach pendant, set the TCP to the hand root frame used in simulation
     (the `root` body of the free hand asset).
   * **Object pose.** Publish the object pose in the UR5e base frame. Use the same object frame as the object
     asset in simulation (e.g. via the mocap rigid-body pivot).

   A quick sanity check: open `vis.py` and check that the green reference object sits on your table, and that
   the preview hand at t = 0 is just above it.

## Running

```bash
# Arm
python scripts/real_world/ur5e_node.py --robot_ip <ROBOT_IP>

# Hand (one of)
ros2 launch wujihand_bringup wujihand.launch.py          # WUJI, publishes /hand_0/joint_states
python scripts/real_world/leap_hand_node.py --port /dev/ttyUSB0   # LEAP (LEAP_Hand_API/python on PYTHONPATH)

# Object pose: any node publishing geometry_msgs/PoseStamped on /object_pose
#   (+ std_msgs/Float64 on /object_joint for articulated objects such as scissors)

# Visualizer (optional, open http://localhost:8080). The first time, store the offset for this setup:
python scripts/real_world/vis.py --hand {wuji,leap} --checkpoint <run_dir>/model_XXXX.pt \
    --sim2real_offset X Y Z --save_setup

# Policy (reads the offset from <run_dir>/real/setup.yaml)
python scripts/real_world/deploy.py --hand {wuji,leap} --checkpoint <run_dir>/model_XXXX.pt
```

**Placing the object with the visualizer.** `vis.py` draws the hand at the measured TCP pose and the measured
object in blue. The reference start pose of the object is drawn in green. Move the real object until blue
matches green; the panel shows the remaining offset (dx, dy, dyaw, dz, and the joint for scissors). For
checkpoints trained with trajectory augmentation, it also says whether the offset is within the training range.
The translucent preview shows the reference hand and object at any step (slider or Play). At t = 0 it is the
start pose that `deploy.py` will move the arm to. It is also a quick check of the sim2real offset and the TCP
setup: the measured hand should overlap the preview hand once the arm is at the start pose.

`deploy.py` waits for all inputs and prints the reference initial object pose. After you place the object,
it moves the hand and then the arm to the start of the reference. When you press Enter it runs one episode and
saves a log to `<run_dir>/real/episode_*.npz`.

**Safety.** `deploy.py` clamps the TCP height to `--min_z`. `ur5e_node.py` ignores a servo target that jumps
more than `--max_jump` (default 10 cm) from the previous target, or from the TCP for the first target after a
goto.

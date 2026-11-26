# REGRIND

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-silver.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)

Implementation of ["A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation"](https://www.yunhaifeng.com/REGRIND/).

![Method overview](data/figures/method_overview.png)

## Installation

First follow the [IsaacLab official instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) to install IsaacSim 5.1.0 and IsaacLab 2.3.0. 

Then set up the `regrind` conda environment with the following commands:

```bash
cd /path/to/IsaacLab
./isaaclab.sh --conda regrind

cd /path/to/regrind
conda activate regrind
isaaclab -i rsl_rl
python -m pip install -e source/regrind
source scripts/set_path.sh
```

## Retargeting

Note: The retargeted trajectories are also provided in this repository. You can directly 
[start training](#rl-training) without running retargeting.

### Dependencies

The retargeter relies on [Drake](https://drake.mit.edu/) (`pydrake`), which is **not** part of the
core install (so `import regrind` works without it). Install it into the `regrind` env:

```bash
conda activate regrind
python -m pip install -e "source/regrind[retargeting]"   # installs the `drake` extra
# or simply: python -m pip install drake
```

### License (Mosek)

The optimizer defaults to the [Mosek](https://www.mosek.com/) QP solver, which requires a license:

```bash
python -m pip install mosek
```

Mosek offers free [personal academic licenses](https://www.mosek.com/products/academic-licenses/)
and trial licenses. Download the license file and point Drake at it explicitly:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

(Drake also ships the license-free Clarabel solver; the retargeter exposes a `solver=` argument if
you prefer to use it instead of Mosek.)

### Running retargeting

Make sure the path environment variables are set (`source scripts/set_path.sh`) so demos, keypoints,
and assets resolve, then run:

```bash
python scripts/retarget_hand_object.py --robot {leaphand,wujihand}  --object {scissors,screwdriver}
```

The output `.h5` follows the schema expected by `load_retargeted_traj` (keys `robot_pos`,
`robot_quat`, `robot_joints`, `object_pos`, `object_quat`, `object_joint`, `robot_keypoints`,
`mano_joint_coords`). Visualize / sanity-check it against an environment by overriding the
trajectory path:

```bash
python scripts/replay_retargeted_traj.py --task Regrind-LeapHand-Scissors-Play-v0 \
    --headless --video --num_envs 1 --retargeted_traj_path /path/to/out.h5
```

## RL training

### Exploring the environments

To list available tasks/environments:

```bash
python scripts/list_envs.py 
```

You can add `--suite [suite]` to only show tasks from a specified suite (robot), where `[suite]` 
can be either `LeapHand` or `WujiHand`.

(From now on, we will use `[task]` to denote the task name. Each task is named as 
`Regrind-[suite]-[object]-v0`, e.g., `Regrind-LeapHand-Scissors-v0`, 
`Regrind-WujiHand-Screwdriver-v0`, etc. The environments used for play / evaluation are named 
as `Regrind-[suite]-[object]-Play-v0`.)

To run a zero-action / random agent and visualize:

```bash
python scripts/zero_agent.py --task [task] --num_envs 16
python scripts/random_agent.py --task [task] --num_envs 16
```

### Training

```bash
python scripts/rsl_rl/train.py --task [task] --headless --num_envs 4096
```
To enable wandb logging, add `--logger wandb --log_project_name regrind`.  

### Evaluation

```bash
python scripts/rsl_rl/play.py --task [task] --headless --video
```

Note that we should use the play environments (with the suffix `-Play-v0`). 
Use `--checkpoint` to specify the checkpoint to evaluated. The latest checkpoint will be used if not specified.
When evaluating an intermediate checkpoint, use `--auto_gravity_from_ckpt` to infer the corresponding gravity since we perform gravity curriculum.

## Real-world deployment

[scripts/real_world](scripts/real_world) contains a minimal ROS 2 deployment for a UR5e arm with a WUJI hand or a LEAP Hand.
The policy's wrist SE(3) target is tracked by the UR controller with `servoL`, and the finger targets go to the hand driver in the same 30 Hz control step.
The robot PC does not need Isaac Sim, only a [RoboStack](https://robostack.github.io/) ROS 2 env plus the `hardware` extra.
Running `play.py` once exports the policy; then:

```bash
conda create -n regrind_hw -c conda-forge -c robostack-kilted ros-kilted-ros-base
conda activate regrind_hw
pip install -e "source/regrind[hardware]"

# first run: --sim2real_offset X Y Z --save_setup stores the offset in <run_dir>/real/setup.yaml for later runs
python scripts/real_world/deploy.py --hand {wuji,leap} --checkpoint <run_dir>/model_XXXX.pt --sim2real_offset X Y Z --save_setup
```

See [scripts/real_world/README.md](scripts/real_world/README.md) for the frame conventions, ROS topics and the arm/hand driver nodes.

"""Installation script for the 'regrind' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "wandb==0.21.1",  # Note: temporary fix for compatibility with protobuf
                      # (https://github.com/isaac-sim/IsaacLab/issues/3138)
    "git-lfs",
]

EXTRAS_REQUIRE = {
    "retargeting": ["drake"],
    # Real-robot deployment (scripts/real_world); ROS 2 itself comes from conda (RoboStack), not pip.
    "hardware": [
        "torch",
        "numpy",
        "scipy>=1.14",  # Rotation.from_quat(..., scalar_first=True)
        "h5py",
        "pyyaml",
        "ur-rtde",  # UR5e
        "dynamixel-sdk",  # LEAP Hand (used by LEAP_Hand_API)
        "viser",  # scripts/real_world/vis.py
        "yourdfpy",
    ],
}

# Installation operation
setup(
    name="regrind",
    packages=["regrind"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)

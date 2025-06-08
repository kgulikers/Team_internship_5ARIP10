# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the Avular Origin One project.
#
# Based on code from the WheeledLab project (https://github.com/NVlabs/wheeledlab)
# Copyright (c) 2025–2027, The Wheeled Lab Project Developers.
#
# Modified by the Avulab Project Developers (5ARIP10), 2025–2025.
# Modifications include: 


import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def root_euler_xyz(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root Euler XYZ angles in the environment frame."""
    xyz_tuple = math_utils.euler_xyz_from_quat(mdp.root_quat_w(env, asset_cfg))
    return torch.stack(xyz_tuple, dim=-1)

# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the Avular Origin One project.
#
# Copyright (c) 2025–2025, Avulab Project Developers (5ARIP10).
# All rights reserved.
#
# This file was created specifically for the Origin One platform.


from isaaclab.utils import configclass
from avulab.envs.mdp import DifferentialDriveActionCfg


@configclass
class OriginActionCfg:
    throttle = DifferentialDriveActionCfg(
        left_wheel_joint_names = ["left_front_wheel_joint", "left_rear_wheel_joint"],
        right_wheel_joint_names = ["right_front_wheel_joint", "right_rear_wheel_joint"],
        wheel_radius=0.12,
        wheel_base=0.5,       
        scale=(1.0, 1.0),   
        bounding_strategy="clip",
        no_reverse=False,
        asset_name="robot",
    )

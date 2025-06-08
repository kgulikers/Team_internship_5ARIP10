"""
Adapted from
https://github.com/Lukas-Malthe-MSc/orbit/blob/project-wrap-up/source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/mdp/actions/actions_cfg.py
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import differential_drive_actions

@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = differential_drive_actions.DifferentialDriveAction

    left_wheel_joint_names: list[str] = MISSING
    right_wheel_joint_names: list[str] = MISSING
    wheel_radius: float = 1.0
    wheel_base: float = 1.0
    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (0.0, 0.0)
    bounding_strategy: str | None = "tanh"
    no_reverse: bool = False
    asset_name: str = "robot"
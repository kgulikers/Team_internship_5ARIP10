# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the Avular Origin One project.
#
# Based on code from the WheeledLab project (https://github.com/NVlabs/wheeledlab)
# Copyright (c) 2025–2027, The Wheeled Lab Project Developers.
#
# Modified by the Avulab Project Developers (5ARIP10), 2025–2025.
# Modifications include: Changed to Origin One

import gymnasium as gym

########################################
############ NAVIGATION ENVS ###########
########################################

from .navigation import OriginOneNavigationRLEnvCfg, OriginOneNavigationPlayEnvCfg
import avulab_tasks.navigation.config.agents.mushr as originone_nav_agents

gym.register(
    id="IsaacLab-OriginOneNavigation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OriginOneNavigationRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{originone_nav_agents.__name__}.rsl_rl_ppo_cfg:OriginOnePPORunnerCfg",
        "play_env_cfg_entry_point": OriginOneNavigationPlayEnvCfg,
    }
)

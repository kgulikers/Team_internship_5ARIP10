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
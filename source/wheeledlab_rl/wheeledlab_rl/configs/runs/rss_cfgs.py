from isaaclab.utils import configclass

from wheeledlab_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)

@configclass
class RSS_NAV_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=32,
        task_name="IsaacLab-OriginOneNavigation"
    )
    train = RLTrainConfig(
        num_iterations=1000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            no_log     = False,        
            no_wandb   = False,          
            log_every  = 1,           
            video          = True,
            video_length   = 300,       
            video_interval = 15000,       
            no_checkpoints   = False,
            checkpoint_every = 10,   
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_DRIFT_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrDriftRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            video_interval=15000
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )


@configclass
class RSS_VISUAL_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=2,
        task_name="Isaac-MushrVisualRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_ELEV_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1,
        task_name="Isaac-MushrElevationRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

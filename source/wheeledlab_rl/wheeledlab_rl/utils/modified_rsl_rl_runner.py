import warnings
import rsl_rl
from rsl_rl import runners
import os
import numpy as np
import threading
try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:

    tqdm = None

import time
import torch
from collections import deque


class OnPolicyRunner(runners.OnPolicyRunner):
    """Override for logging purposes"""

    def __init__(self, env, agent_cfg, log_cfg, device="cpu"):
        super().__init__(env, agent_cfg, log_cfg.run_log_dir, device)
        self.no_log = log_cfg.no_log
        self.no_wandb = log_cfg.no_wandb
        self.logger_type = None
        self.log_every = getattr(log_cfg, "log_every", 1)
        self.ckpt_every = getattr(log_cfg, "checkpoint_every", 50)
        if not self.no_wandb:
            self.logger_type = "wandb"
        # self.pbar = tqdm(total=self.cfg.get("rl_max_iterations", 0))

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
                # quickly find out where env.step() is implemented:
        import inspect
        raw_env = self.env
        while hasattr(raw_env, "env"):
            raw_env = raw_env.env
        print("→ Actual step() lives in:", inspect.getsourcefile(raw_env.step))

        # initialize writer
        if not self.no_log and self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(
                log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
            )
            self.writer.log_config(
                self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
            )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        


        for it in tqdm(range(start_iter, tot_iter)):
            start_collect = time.time()
            if obs.isnan().any():
                raise ValueError("NaN in the initial observation (obs)")
            if critic_obs.isnan().any():
                raise ValueError("NaN in the initial observation (critic_obs)")

            policy_time_sum = 0.0
            step_time_sum = 0.0
            obsnorm_time_sum = 0.0
            todev_time_sum = 0.0
            process_time_sum = 0.0
            bookkeep_time_sum = 0.0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    t0 = time.time()
                    actions = self.alg.act(obs, critic_obs)
                    t1 = time.time()
                    policy_time_sum += (t1 - t0)

                    t2 = time.time()
                    obs_raw, rewards, dones, infos = self.env.step(actions)
                    t3 = time.time()
                    step_time_sum += (t3 - t2)

                    t4 = time.time()
                    if actions.isnan().any():
                        raise ValueError("NaN in actions")
                    obs = self.obs_normalizer(obs_raw)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(
                            infos["observations"]["critic"]
                        )
                    else:
                        critic_obs = obs
                    t5 = time.time()
                    obsnorm_time_sum += (t5 - t4)

                    t6 = time.time()
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    t7 = time.time()
                    todev_time_sum += (t7 - t6)

                    t8 = time.time()
                    self.alg.process_env_step(rewards, dones, infos)
                    t9 = time.time()
                    process_time_sum += (t9 - t8)

                    if not self.no_log:
                        t10 = time.time()
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        t11 = time.time()
                        bookkeep_time_sum += (t11 - t10)

                end_collect = time.time()
                collection_time = end_collect - start_collect

                # Learning step
                start_learn = end_collect
                self.alg.compute_returns(critic_obs)

            t12 = time.time()
            loss_dict = self.alg.update()
            end_learn = time.time()
            learn_time = end_learn - start_learn

            self.current_learning_iteration = it
            if (not self.no_log and not self.no_wandb) and (it % self.log_every == 0):
                start_log = time.time()
                self.log(locals())
                end_log = time.time()
                log_time = end_log - start_log
                print(
                    f"[Iter {it:5d}]  "
                    f"policy: {policy_time_sum*1000:7.2f} ms  |  "
                    f"env.step: {step_time_sum*1000:7.2f} ms  |  "
                    f"obsnorm: {obsnorm_time_sum*1000:7.2f} ms  |  "
                    f"todev: {todev_time_sum*1000:7.2f} ms  |  "
                    f"process: {process_time_sum*1000:7.2f} ms  |  "
                    f"bookkeep: {bookkeep_time_sum*1000:7.2f} ms  |  "
                    f"collection_all: {collection_time*1000:7.2f} ms  |  "
                    f"learn: {learn_time*1000:7.2f} ms  |  "
                    f"log: {log_time*1000:7.2f} ms  |  "
                    f"total: {((collection_time+learn_time+log_time)*1000):7.2f} ms"
                )
                rewbuffer.clear()
                lenbuffer.clear()


            #if (self.ckpt_every > 0) and (it % self.ckpt_every == 0):
            #    model_path = os.path.join(self.log_dir, "models", f"model_{it}.pt")
            #    os.makedirs(os.path.dirname(model_path), exist_ok=True)
            #    # Use a background thread so `torch.save` cannot block main loop
            #    threading.Thread(
            #        target=torch.save,
            #        args=(self.alg.policy.actor, model_path),
            #        daemon=True,
            #    ).start()
            #ep_infos.clear()

            if (self.ckpt_every > 0) and (it % self.ckpt_every == 0):
                model_dir = os.path.join(self.log_dir, "models")
                os.makedirs(model_dir, exist_ok=True)

                # 1) Save full actor model
                full_model_path = os.path.join(model_dir, f"model_{it}.pt")
                threading.Thread(
                    target=torch.save,
                    args=(self.alg.policy.actor, full_model_path),
                    daemon=True,
                ).start()

                # 2) Save only state_dict
                state_dict_path = os.path.join(model_dir, f"model_{it}_state_dict.pt")
                threading.Thread(
                    target=torch.save,
                    args=(self.alg.policy.actor.state_dict(), state_dict_path),
                    daemon=True,
                ).start()

            ep_infos.clear()



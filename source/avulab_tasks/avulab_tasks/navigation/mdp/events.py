import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab.envs import ManagerBasedEnv
import isaaclab.sim.utils as sim_utils



class reset_root_state_along_track(ManagerTermBase):
    """Creates track path, saves reference poses and resets the asset root state along the track path
    during environment resets with additive noise.
    """

    def _init_(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super()._init_(cfg, env)
        self.track_radius = torch.tensor(cfg.params.get("track_radius", 0.8), device=self.device)
        self.track_straight_dist = torch.tensor(cfg.params.get("track_straight_dist", 0.8), device=self.device)
        self.num_points = cfg.params.get("num_points", 20)

        # Pre-generate reference points and sample around them for performance
        self.reference_poses = self.generate_reference_poses()

    def generate_reference_poses(self):
        dist_track = 2. * torch.pi * self.track_radius + 4. * self.track_straight_dist
        dists = torch.rand(self.num_points, device=self.device) * dist_track

        # Case 1
        case1_pos = torch.stack([self.track_radius.repeat_interleave(dists.shape[0]).to(self.device),
                                dists - self.track_straight_dist.repeat_interleave(dists.shape[0]).to(self.device),
                                torch.zeros_like(dists)], dim=-1)
        case1_ori = torch.stack([torch.zeros_like(dists), torch.zeros_like(dists), torch.full_like(dists, 90)], dim=-1)

        # Case 2
        angle = (dists - 2 * self.track_straight_dist) / self.track_radius
        case2_pos = torch.stack([self.track_radius * torch.cos(angle),
                                 self.track_straight_dist + self.track_radius * torch.sin(angle),
                                 torch.zeros_like(dists)], dim=-1)
        case2_ori = torch.stack([torch.zeros_like(dists),
                                 torch.zeros_like(dists),
                                 90 + angle * 180 / torch.pi], dim=-1)

        # Case 3
        remaining = dists - 2 * self.track_straight_dist - torch.pi * self.track_radius
        case3_pos = torch.stack([-self.track_radius.repeat_interleave(dists.shape[0]).to(self.device),
                                self.track_straight_dist.repeat_interleave(dists.shape[0]).to(self.device) - remaining,
                                torch.zeros_like(dists)], dim=-1)
        case3_ori = torch.stack([torch.zeros_like(dists),
                                 torch.zeros_like(dists),
                                 torch.full_like(dists, 270)], dim=-1)

        # Case 4
        angle2 = (dists - 4 * self.track_straight_dist - torch.pi * self.track_radius) / self.track_radius
        case4_pos = torch.stack([-self.track_radius * torch.cos(angle2),
                                 -self.track_straight_dist - self.track_radius * torch.sin(angle2),
                                 torch.zeros_like(dists)], dim=-1)
        case4_ori = torch.stack([torch.zeros_like(dists),
                                 torch.zeros_like(dists),
                                 270 + angle2 * 180 / torch.pi], dim=-1)

        # Combine cases using torch.where
        pos = torch.where(
            (dists < 2 * self.track_straight_dist).unsqueeze(-1),
            case1_pos,
            torch.where(
                (dists < 2 * self.track_straight_dist + torch.pi * self.track_radius).unsqueeze(-1),
                case2_pos,
                torch.where(
                    (dists < 4 * self.track_straight_dist + torch.pi * self.track_radius).unsqueeze(-1),
                    case3_pos,
                    case4_pos
                )
            )
        )

        ori = torch.where(
            (dists < 2 * self.track_straight_dist).unsqueeze(-1),
            case1_ori,
            torch.where(
                (dists < 2 * self.track_straight_dist + torch.pi * self.track_radius).unsqueeze(-1),
                case2_ori,
                torch.where(
                    (dists < 4 * self.track_straight_dist + torch.pi * self.track_radius).unsqueeze(-1),
                    case3_ori,
                    case4_ori
                )
            )
        )

        reference_poses = torch.stack([pos, ori], dim=1)
        return reference_poses

    def _call_(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        track_radius: float,
        track_straight_dist: float,
        num_points: int,
        asset_cfg: SceneEntityCfg,
        pos_noise: float = 0.0,
        yaw_noise: float = 0.0,
    ):
        """Reset the asset root state by sampling around reference poses along the track path.
        """
        # access the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # Sample around the reference poses
        idx = torch.randint(self.num_points, (len(env_ids),), device=env.device)
        ref_points = self.reference_poses[idx]

        xy_noise = (2 * torch.rand((len(env_ids), 2), device=env.device) - 1.) * pos_noise
        add_pos_noise = torch.cat((xy_noise, torch.zeros_like(xy_noise[..., 0:1])), dim=-1)
        posns = ref_points[:, 0, :] + add_pos_noise

        yaw_noise = (2 * torch.rand(len(env_ids), device=env.device) - 1.) * yaw_noise
        oris = ref_points[:, 1, :]
        roll, pitch, yaw = torch.unbind(torch.deg2rad(oris), dim=-1)
        yaw += yaw_noise
        oris = math_utils.quat_from_euler_xyz(roll=roll, pitch=pitch, yaw=yaw)

        asset.write_root_pose_to_sim(torch.cat([posns, oris], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)

class reset_root_state_new(ManagerTermBase):
    """
    Reset the robot at a fixed start pose each episode.
    Params:
      - asset_cfg: the robot’s SceneEntityCfg
      - pos: [x, y, z] start coordinates (your point A)
      - rot: [qx, qy, qz, qw] start orientation
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        pos: list[float],
        rot: list[float],
    ):
        # Fetch the instantiated asset from the scene by name
        asset: Articulation | RigidObject = env.scene[asset_cfg.name]

        # Create a [batch_size x 3] tensor for position
        pos_t = torch.tensor(pos, device=env.device).view(1, 3).repeat(len(env_ids), 1)
        # Create a [batch_size x 4] tensor for rotation
        rot_t = torch.tensor(rot, device=env.device).view(1, 4).repeat(len(env_ids), 1)
        # Concatenate to [batch_size x 7]
        root_state = torch.cat([pos_t, rot_t], dim=-1)

        # Apply the pose to the simulator
        asset.write_root_pose_to_sim(root_state, env_ids=env_ids)
        # Zero out any residual velocity
        asset.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=env.device),
            env_ids=env_ids,
        )
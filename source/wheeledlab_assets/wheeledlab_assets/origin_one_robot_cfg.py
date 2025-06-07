


import os

from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg


WHEEL_RADIUS = 0.1175
MAX_WHEEL_SPEED = 2  

OriginRobotCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "Robots", "origin_v18.urdf")
),
        joint_drive=None,
        fix_base=False,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=True,
        root_link_name="main_body",
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={},  # optionally set joints like "left_front_wheel_joint": 0.0
    ),
    actuators={
        "wheel_act": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=50,
            damping=5,
            effort_limit=10.0,
            velocity_limit=MAX_WHEEL_SPEED / WHEEL_RADIUS,
        )
    },
)
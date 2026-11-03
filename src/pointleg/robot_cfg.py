# =============================================================================
# robot_cfg.py — 12 自由度纯点足四足机器人实体配置
# =============================================================================
"""12-DoF point-foot quadruped entity configuration.

The policy action order is leg-major:
``fl, fr, rl, rr`` and, inside each leg, ``abduction, pitch, knee``.
MJLab retains the XML joint order. The position action validates this order
before either the default RSL-RL runner or the optional HIM runner can start.
"""

import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from pointleg.motor_spec import LEG_MOTOR_SPEC


_PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / "mjcf").is_dir()),
    Path(__file__).resolve().parent,
)


ROBOT_XML = Path(
    os.environ.get("POINTLEG_XML", str(_PROJECT_ROOT / "mjcf" / "pointleg.xml"))
).expanduser().resolve()

# XML待核对：四条腿的前缀、三个关节后缀及其正负方向必须与最终 XML 一致。
LEG_NAMES = ("fl", "fr", "rl", "rr")
LEG_JOINT_SUFFIXES = (
    "hip_abduction_joint",
    "hip_pitch_joint",
    "knee_joint",
)
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}" for leg in LEG_NAMES for joint in LEG_JOINT_SUFFIXES
)

# 正则仅选择 12 个腿关节；MJLab 保留 XML 的自然顺序，必须与 LEG_JOINT_NAMES 一致。
LEG_JOINT_PATTERNS = tuple(
    f".*_{joint}" for joint in LEG_JOINT_SUFFIXES
)

# XML待核对：base body、足端刚体和足端 site 均需逐项核对后再开始训练。
BASE_BODY_NAME = "base_link"
FOOT_BODY_NAMES = tuple(f"{leg}_foot_Link" for leg in LEG_NAMES)
FOOT_SITE_NAMES = tuple(f"{leg}_foot_site" for leg in LEG_NAMES)

# XML待核对：初始机身高度和关节角必须由最终足端几何及关节轴向重新标定。
STANDING_BASE_HEIGHT = 0.42
CRAWL_BASE_HEIGHT = 0.20
CRAWL_KNEE_ANGLE = -2.50


def get_spec() -> mujoco.MjSpec:
    """加载机器人MJCF模型，由MJLab重新生成执行器和训练初始姿态

    为什么要删掉 xml 里的 actuator：
      MJCF 自带的一般是 position/velocity/motor 等简单驱动，参数（力矩上限、PD 增益）
      写死在 xml 里，改起来要么重新生成 xml 要么手改一堆数字。删除后由下面的
      LEG_ACTUATOR_CFG 用代码批量生成，训练时还能配合域随机化
      直接改 stiffness/damping，方便调参与 sim2real 对齐。

    该函数以 spec_fn 形式传给 EntityCfg，属于延迟加载：真正建场景时才调用一次。

    Returns:
        mujoco.MjSpec: 处理完成的MjSpec内存模型对象
    """
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(
            f"Point-foot XML is not ready: {ROBOT_XML}. "
            "Set POINTLEG_XML to the completed model path before simulation."
        )
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    LEG_MOTOR_SPEC.validate_simulation_effort_limit(EFFORT_LEG)
    # 关节总力矩限幅与执行器共用配置，避免外部 XML 的旧限幅覆盖训练参数。
    for joint in spec.joints:
        if joint.name in LEG_JOINT_NAMES:
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.actfrcrange = [-EFFORT_LEG, EFFORT_LEG]
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    # XML 站立关键帧仅供独立查看，训练姿态由各任务的 init_state 生成。
    for keyframe in list(spec.keys):
        spec.delete(keyframe)
    return spec


# 十二个关节均使用 N6014B-12.6，用户确认实机供电为 48 V。
MOTOR_SUPPLY_VOLTAGE_V = 48.0
ESTIMATED_NO_LOAD_SPEED_RAD_S = LEG_MOTOR_SPEC.estimate_no_load_speed(
    MOTOR_SUPPLY_VOLTAGE_V
)

# 仿真对称力矩上限为 25 N·m，低于电机峰值，PD 增益仍待实机标定。
# 空载转速估算仅供核对，不作为带载速度约束；当前不模拟电压、热或反电动势。
STIFFNESS_LEG = 50.0
DAMPING_LEG = 1.5
EFFORT_LEG = 25.0
LEG_MOTOR_SPEC.validate_simulation_effort_limit(EFFORT_LEG)

# 参数已在减速器输出端定义，关节驱动不再额外乘以减速比。
LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=LEG_JOINT_PATTERNS,
    stiffness=STIFFNESS_LEG,
    damping=DAMPING_LEG,
    effort_limit=EFFORT_LEG,
)


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, STANDING_BASE_HEIGHT),
    joint_pos={
        ".*_hip_abduction_joint": 0.0,
        ".*_hip_pitch_joint": 0.550,
        ".*_knee_joint": -1.125,
    },
    joint_vel={".*": 0.0},
)

# ====================== 碰撞配置 ======================
# XML待核对：足端 geom 名称必须匹配下列正则，geom 名称与 body 名称分别核对。
# 足端使用带摩擦的三维接触，机身和腿杆保留法向接触。
COLLISION_CFG = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim={".*_foot_Link.*": 3, ".*": 1},
    priority={".*_foot_Link.*": 1},
    friction={".*_foot_Link.*": (0.8, 0.01, 0.001)},
)


ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(LEG_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.95,
)


def get_robot_cfg() -> EntityCfg:
    """Return the normal standing configuration for the 12-DoF point-foot robot."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(COLLISION_CFG,),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )

# ====================== 机器人初始状态（低矮匍匐姿态） ======================
# 用于 PointLeg-Crawl-v0，机身初始高度为 0.20m。
# 髋俯仰和膝角形成低姿构型，默认关节角保持在软限位内。
#   - 髋外展 ±0.4 让左右腿向外张开，增大支撑面防止侧翻
# 左右符号相反（左正右负）是因为两侧关节轴向镜像，同号反而会让两腿朝同一边倒。
def get_robot_crawl_cfg() -> EntityCfg:
    """Return the low-body configuration used by the crawl task."""
    crawl_init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, CRAWL_BASE_HEIGHT),
        joint_pos={
            "(fl|rl)_hip_abduction_joint": 0.4,
            "(fr|rr)_hip_abduction_joint": -0.4,
            ".*_hip_pitch_joint": 1.65,
            ".*_knee_joint": CRAWL_KNEE_ANGLE,
        },
        joint_vel={".*": 0.0},
    )
    return EntityCfg(
        init_state=crawl_init_state,
        collisions=(COLLISION_CFG,),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )

# ====================== RL策略action缩放系数 ======================
# Action scale说明：
# 策略使用高斯动作分布，动作没有固定的正负一边界。
# 动作乘以缩放系数后叠加默认关节角，最终目标裁剪到关节软限位。
LEG_POS_SCALE = 0.25
# 髋外展的缩放系数为 0.125，其余关节在 env_cfgs.py 中使用 LEG_POS_SCALE。

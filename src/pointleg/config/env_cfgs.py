"""Robot Environment Configurations for RL Locomotion Training.
This module defines the Manager-Based RL Environment Configurations for a
12-DoF point-foot quadruped. It handles sensors, actuators, command generators,
observation/critic terms, event randomizations, rewards, and terminations for
flat ground, rough terrains, and crawling tasks.

【本文件的组织方式：一个模板 + 三次改配置】
    _make_base_env_cfg()  基础模板：传感器、观测、动作、指令、域随机化、奖励、终止
        ├── flat_env_cfg()    在模板上只改：机器人姿态(站立)、地形(全平地)、回合时长
        ├── rough_env_cfg()   在模板上大改：地形发生器、课程、奖励重构、指标扩展
        └── crawl_env_cfg()   在模板上大改：机器人姿态(匍匐)、地形(低杆)、匍匐高度奖励

    三个函数都是"先 cfg = _make_base_env_cfg() 再逐项增删改"的写法，读代码时
    要特别注意 pop / 覆盖 的顺序——后半段的操作会覆盖前半段，改动容易互相抵消。
    例如 rough 里先把 base_height_l2 的 weight 设为 0，再用 sensor_cfg 指到高度雷达，
    实际起作用的只是后面那一步。

【管理器模型（ManagerBased RL）】
    mjlab 把 RL 环境的各个部分拆成独立 manager，每个 manager 下挂若干 term：
      scene(场景/机器人/地形/传感器)  actions(动作)  observations(观测,分 actor/critic 组)
      commands(指令)  rewards(奖励)  terminations(终止)  events(域随机化)
      curriculum(课程)  metrics(诊断指标,不参与训练只用来看)
    本文件干的事就是把这些 manager 的 term 表填满。
    每个 term 都是 XxxTermCfg(func=..., weight=/params=...)，func 可以来自 mjlab 内置，
    也可以是本仓库 mdp/ 下的自定义函数。

【Actor / Critic 观测分离（非对称 actor-critic）】
    actor 组只给策略能真实获得的量（带噪声），critic 组额外附加特权信息
    （机身真实线速度、脚部接触、地形高度扫描），因为价值网络只在训练时用，
    部署时不需要，可以"作弊"以获得更准确的 V(s) 估计。
"""

#以下分为4个环境模板，后三个由第一个得到
import math

# ====================== MJLab 核心模块导入 ======================
# RL环境基类：管理器式强化学习环境配置
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp          # MDP基础组件
from mjlab.envs.mdp import dr as envs_dr        # 域随机化 Domain Randomization
# 仿真器配置
from mjlab.sim import SimulationCfg, MujocoCfg

# 各管理器配置项
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

# 场景配置
from mjlab.scene import SceneCfg
# 传感器套件：接触传感器、射线扫描、网格图案
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    GridPatternCfg,
)

# 速度任务MDP：命令生成器
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

# MJLab 内置地形库
from mjlab.terrains import (
    TerrainEntityCfg,
    TerrainGeneratorCfg,
    BoxFlatTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfPerlinNoiseTerrainCfg,
)
# 自定义竞赛障碍物地形
from ..terrains import RCLowBarTerrainCfg

# 工具：均匀噪声配置
from mjlab.utils.noise import UniformNoiseCfg as Unoise
# 可视化查看器配置
from mjlab.viewer import ViewerConfig

# 机器人本体配置：站立/匍匐两套初始状态、动作缩放系数
from ..robot_cfg import (
    BASE_BODY_NAME,
    FOOT_BODY_NAMES,
    FOOT_SITE_NAMES,
    LEG_JOINT_NAMES,
    LEG_JOINT_PATTERNS,
    LEG_POS_SCALE,
    STANDING_BASE_HEIGHT,
    get_robot_cfg,
    get_robot_crawl_cfg,
)

# 自定义MDP组件：带延迟低通的动作模型
from ..mdp.lowpass_actions import JointPositionDelayedLowPassActionCfg
# 自定义MDP：持续扰动注入
from ..mdp.disturbances import apply_continuous_disturbance
# 自定义MDP：纯正奖励机制开关
from ..mdp.only_positive_rewards import configure_positive_rewards

# 自定义奖励函数集
from ..mdp.rewards import (
    track_linear_velocity,
    track_linear_velocity_l1,
    track_linear_velocity_x,
    track_linear_velocity_y,
    track_angular_velocity,
    track_angular_velocity_z,
    base_height_l2,
    base_clearance_outside_range_l2,
    command_aligned_progress_reward,
    command_progress_deficit_penalty,
    command_orthogonal_drift_penalty,
    TranslationStanceOverstayPenalty,
    TranslationFootWorkspacePenalty,
    turn_progress_deficit_penalty,
    turn_linear_drift_penalty,
    turn_yaw_error_degrees,
    TurnStanceTwistPenalty,
    TurnFootWorkspacePenalty,
    command_aligned_progress_metric,
    terrain_type_metric,
    terrain_level_metric,
    termination_reason_metric,
    safe_base_lin_vel,
    safe_foot_contact,
    safe_height_scan,
    adaptive_leg_motion_penalty,
    leg_symmetry,
    stand_still,
    hip_deviation,
    hip_abduction_excess,
    joint_deviation_l2,
    knee_extension_penalty,
    flat_orientation_l2,
    pitch_excess_penalty,
    lin_vel_z_l2,
    crawl_height_reward,
    terrain_level_bonus,
    action_rate_curriculum_l2,
    variable_posture,
    joint_pos_penalty,
    joint_mirror,
    feet_contact_without_cmd,
    long_foot_air_time_penalty,
    FootSwingHeightReward,
    short_stance_penalty,
    StanceBalancePenalty,
    FootLateralPlacementPenalty,
    ForwardFootTouchdownPenalty,
    ForwardFootSwingPenalty,
    support_deficit_penalty,
    supported_feet_air_time,
    upright_roll_only,
    upward,
    joint_power,
    ang_vel_xy_l2,
    contact_forces,
    tracking_lin_vel_error,
    tracking_yaw_vel_error,
    tracking_lin_vel_x_error,
    tracking_lin_vel_y_error,
    tracking_lin_vel_along_command_error,
    actual_lin_vel_orthogonal_command_mean,
    command_lin_vel_mean,
    command_yaw_vel_abs_mean,
    actual_lin_vel_mean,
    tracking_lin_vel_error_band_mean,
    tracking_lin_vel_axis_error_band_mean,
    command_band_active,
    command_direction_active,
    upright_metric,
    base_ground_contact_metric,
)
from ..mdp.terminations import joint_pos_out_of_limits, root_height_out_of_range
from .evaluation import configure_evaluation

# 自定义课程学习函数集
from ..mdp.curriculums import (
    command_axis_levels_vel,
    rough_terrain_levels,
    terrain_levels_ramp_strict,
    terrain_levels_vel_strict,
)
# 自定义命令生成器：带阈值的均匀速度命令
from ..mdp.commands import UniformThresholdVelocityCommandCfg


# ======================================================================
# 基础环境模板：所有环境均基于此配置扩展
# ======================================================================
def _make_base_env_cfg() -> ManagerBasedRlEnvCfg:
    """构建基础RL环境配置模板
    包含通用传感器、观测空间、动作空间、命令生成、域随机化、奖励函数、终止条件等基础组件
    平地、复杂地形、匍匐环境均基于此基础配置修改扩展

    每个 manager 段的阅读顺序建议：
      sensors → observations → actions → commands → events → rewards → terminations
      因为这正好是"感知 → 决策 → 执行 → 扰动 → 评价 → 判负"的闭环顺序。

    Returns:
        ManagerBasedRlEnvCfg: 基础环境配置对象
    """
    
    # ====================== 传感器定义 ======================
    # XML待核对：FOOT_BODY_NAMES 必须和四个点足刚体名称一一对应。
    # 足端接地接触传感器：检测四个点足与地面的接触状态与接触力
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=FOOT_BODY_NAMES,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # 检测与地形的接触
        fields=("found", "force"),  # 输出字段：是否接触、接触力大小
        reduce="none",
        num_slots=1,
        history_length=3,  # 保留3帧历史数据
        track_air_time=True,  # 统计离地时长，用于步态节律奖励
    )

    # 底盘接地传感器：检测base_link是否触地，用于摔倒判定
    base_ground_cfg = ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern=BASE_BODY_NAME, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # 腿部碰撞传感器：检测腿杆与地形的碰撞
    body_collision_cfg = ContactSensorCfg(
        name="body_collision",
        primary=ContactMatch(
            mode="body",
            pattern=(".*_hip_abduction_Link", ".*_hip_pitch_Link", ".*_knee_Link"),  # 匹配腿部所有连杆
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # 地形高度扫描雷达：向前方发射射线阵列，扫描前方地形高度（critic特权信息）
    terrain_scan = RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name=BASE_BODY_NAME, entity="robot"),  # 挂载在底盘base_link上
        pattern=GridPatternCfg(resolution=0.08, size=(1.6, 1.0)),  # 网格扫描范围与分辨率
        ray_alignment="yaw",  # 射线随机身偏航角对齐
        max_distance=5.0,  # 最大扫描距离5m
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,  # 关闭调试可视化
    )

    # XML待核对：地形使用 geom group 0，机器人几何体应使用其他 group，避免测到自身。
    base_height_scan = RayCastSensorCfg(
        name="base_height_sensor",
        frame=ObjRef(type="body", name=BASE_BODY_NAME, entity="robot"),
        pattern=GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
        ray_alignment="world",
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    # ====================== 观测空间配置 ======================
    # ---- Actor 观测（策略网络输入，含噪声模拟真实传感器） ----
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel, scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),  # 角速度观测噪声
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 重力投影观测噪声
        ),
        "command": ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "twist"},  # 读取twist速度命令
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 关节位置观测噪声
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
            scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5),  # 关节速度缩放与噪声
        ),
        "actions": ObservationTermCfg(func=velocity_mdp.last_action),  # 上一时刻动作值
    }

    # ---- Critic 观测（价值网络输入，包含特权信息，无噪声） ----
    # 当前尺寸为 325：actor 45 + 机身线速度 3 + 四足接触 4 + 高度扫描 273。
    critic_terms = {
        **actor_terms,  # 继承actor全部观测
        "base_lin_vel": ObservationTermCfg(func=safe_base_lin_vel, scale=2.0),  # 机身线速度（特权信息）
        "foot_contact": ObservationTermCfg(
            func=safe_foot_contact, params={"sensor_name": "feet_ground_contact"},  # 足端接触状态
        ),
        "height_scan": ObservationTermCfg(
            func=safe_height_scan, params={"sensor_name": "height_scanner"},
            clip=(-1.0, 1.0),  # 高度扫描值裁剪
        ),
    }

    # 观测分组输出
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms, concatenate_terms=True,
            enable_corruption=True,  # 启用观测噪声扰动
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False,  # critic不添加噪声
        ),
    }

    # ====================== 动作空间与速度命令 ======================
    # 策略输出经过随机延迟和一阶低通后，作为默认关节角的增量位置指令。
    #   use_default_offset=True 表示以实体的默认关节角为基准（增量控制），
    #   这样策略只要输出接近 0 就能保持站立姿态，学习负担远小于直接输出绝对角度。
    actions: dict[str, ActionTermCfg] = {
        # 腿部关节位置动作：带延迟低通滤波，模拟真实执行器迟滞
        "leg_joint_pos": JointPositionDelayedLowPassActionCfg(
            entity_name="robot",
            actuator_names=LEG_JOINT_PATTERNS,
            expected_joint_names=LEG_JOINT_NAMES,
            # 动作缩放：髋外展缩放0.125rad，其余腿部关节缩放0.25rad
            scale={".*_hip_abduction_joint": 0.125, "^(?!.*_hip_abduction_joint).*": LEG_POS_SCALE},
            use_default_offset=True,
            clip_to_joint_limits=True,
            control_frequency=50.0, cut_off_frequency=5.0,  # 控制频率50Hz，低通截止5Hz
            min_delay=0, max_delay=2,  # 执行延迟0~2步
        ),
    }

    # 速度命令生成器
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformThresholdVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(10.0, 10.0),  # 命令每10s重采样一次
            rel_standing_envs=0.15, rel_heading_envs=1.0, heading_command=True,
            heading_control_stiffness=0.6,
            rel_forward_envs=0.40,  # 40%环境为纯前向命令
            ranges=UniformThresholdVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5),  # 线速度范围x±1m/s，y±0.5m/s
                ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),  # 角速度±1rad/s，航向±π
            ),
        )
    }

    # ====================== 域随机化事件 ======================
    # mode 决定执行时机：
    #   "startup"  整个训练只执行一次（用于随机化物理参数，如质量、摩擦、刚度）
    #   "reset"    每次环境 reset 时执行（用于随机化初始位姿与关节状态）
    #   "interval" 按时间间隔周期执行（用于中途施加推力扰动）
    # startup 随机化在环境初始化时执行，reset 随机化在回合重置时执行。
    events = {
        # 重置事件：场景重置为默认状态
        "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        # 重置事件：机器人初始位姿与速度均匀随机化
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform, mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # 间隔事件：周期性给机器人施加推力扰动，增强鲁棒性
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=(10.0, 15.0),  # 每10~15s触发一次
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # 机身按密度比例同步缩放质量与惯量，并在同一事件中随机化质心。
        "base_inertia": EventTermCfg(
            func=envs_dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
                "alpha_range": (0.5 * math.log(0.85), 0.5 * math.log(1.45)),
                "t_range": (-0.05, 0.05),
            },
        ),
        # 启动事件：机身摩擦系数随机化
        "body_friction": EventTermCfg(
            func=envs_dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)), "operation": "abs", "ranges": (0.3, 1.0)},
        ),
        # 执行器位置控制的比例与微分增益分别采样。
        "actuator_pd_gains": EventTermCfg(
            func=envs_dr.pd_gains, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.9, 1.1),
                    "kd_range": (0.9, 1.1), "operation": "scale", "distribution": "log_uniform"},
        ),
    }

    # ====================== 奖励函数配置 ======================
    # 【奖励权重的量级关系（读这张表先看这一段）】
    #   正奖励：跟踪类 ≈ 1.0~2.5（指数奖励上限为 1，所以 weight 就是满分价值）
    #   负惩罚：小的 1e-7~1e-4（数值量大、每步都产生，需要小权重）
    #           中的 0.1~2.0 （姿态、平滑类）
    #           大的 -200     （is_terminated，摔倒是一次性大事件）
    #   所有负权重项最终都会乘上直立加权，所以摔倒过程中的惩罚会自然衰减。
    rewards = {
        # 主奖励：线速度跟踪
        "track_lin_vel": RewardTermCfg(func=track_linear_velocity, weight=2.5, params={"std": 0.5, "command_name": "twist"}),
        # 主奖励：角速度跟踪
        "track_ang_vel": RewardTermCfg(func=track_angular_velocity, weight=2.5, params={"std": 0.5, "command_name": "twist"}),
        # 奖励：机身直立姿态保持
        "upright": RewardTermCfg(func=velocity_mdp.upright, weight=1.0, params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))}),
        # 惩罚：机身高度偏离目标值
        "base_height_l2": RewardTermCfg(func=base_height_l2, weight=-2.0, params={"target_height": STANDING_BASE_HEIGHT}),
        # 惩罚：机身角速度过大
        "body_ang_vel": RewardTermCfg(func=velocity_mdp.body_angular_velocity_penalty, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))}),
        # 惩罚：回合终止（摔倒、超时等）
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
        # 惩罚：关节力矩能耗
        "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-2.0e-4),
        # 惩罚：关节加速度（平滑性）
        "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
        # 惩罚：动作变化率（动作平滑性）
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        # 惩罚：关节位置超限
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-10.0),
        # 奖励：足端离地时长（步态节律）
        "feet_air_time": RewardTermCfg(func=velocity_mdp.feet_air_time, weight=0.5, params={"sensor_name": "feet_ground_contact", "threshold_min": 0.1, "threshold_max": 0.5, "command_name": "twist", "command_threshold": 0.1}),
        # 惩罚：自适应腿部运动惩罚
        "leg_motion_penalty": RewardTermCfg(func=adaptive_leg_motion_penalty, weight=-0.02, params={"command_name": "twist", "sensor_name": "feet_ground_contact", "command_threshold": 0.05, "tilt_relax_start": 0.08, "tilt_relax_end": 0.30, "contact_target": 0.85, "min_penalty_scale": 0.2, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_abduction_joint", ".*_hip_pitch_joint", ".*_knee_joint"))}),
        # 惩罚：静止不动
        "stand_still": RewardTermCfg(func=stand_still, weight=-0.2, params={"command_name": "twist", "command_threshold": 0.1}),
        # 惩罚：腿部碰撞
        "body_collision": RewardTermCfg(func=velocity_mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "body_collision"}),
    }

    # ====================== 终止条件配置 ======================
    # 触发的任一条件都会结束该环境的 episode 并 reset。
    # time_out 是"正常到点"，其余是"失败"——HIMLOCO 风格的纯正奖励机制下，
    # 终止本身不再给负奖励（rough 里 is_terminated 权重也被置 0），
    # 靠"中断累积正奖励"来间接惩罚早退。
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),  # 回合超时
        "bad_orientation": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": 1.0}),  # 姿态倾斜超限
        "base_ground_contact": TerminationTermCfg(func=velocity_mdp.illegal_contact, params={"sensor_name": "base_ground_contact"}),  # 底盘触地判定摔倒
        "joint_pos_out_of_limits": TerminationTermCfg(
            func=joint_pos_out_of_limits,
            params={"limit_margin": 0.02, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
        ),
        "base_height_out_of_range": TerminationTermCfg(
            func=root_height_out_of_range,
            params={"minimum_height": 0.18, "maximum_height": 0.65,
                    "sensor_name": "base_height_sensor"},
        ),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),  # 数值异常检测
    }

    # ====================== 课程学习与评估指标 ======================
    curriculum: dict[str, CurriculumTermCfg] = {}  # 基础配置无课程，子类环境按需添加
    metrics = {"mean_leg_action_acc": MetricsTermCfg(func=velocity_mdp.mean_action_acc)}  # 基础指标

    # ====================== 组装基础环境配置 ======================
    # decimation=4 的含义：仿真步长 0.005s，每 4 个仿真步做一次策略推理与控制更新，
    # 即控制频率 = 1/(0.005×4) = 50Hz。这与动作低通滤波里配的 control_frequency=50.0 必须一致，
    # 改任何一个都要同步改另一个，否则滤波截止频率会和设计值不符。
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=2048, env_spacing=2.5,  # 并行环境数量2048，环境间距2.5m
            terrain=TerrainEntityCfg(terrain_type="generator", terrain_generator=TerrainGeneratorCfg(
                size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
                layout_shape=(4, 4),
                sub_terrains={"flat": BoxFlatTerrainCfg(proportion=1.0)},  # 基础默认全平地
            )),
            sensors=(feet_ground_cfg, base_ground_cfg, body_collision_cfg, terrain_scan, base_height_scan),
        ),
        commands=commands, actions=actions, observations=observations,
        rewards=rewards, terminations=terminations, events=events,
        metrics=metrics, curriculum=curriculum, decimation=4, episode_length_s=20.0,  # 控制步频=仿真步频/4，单回合20s
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005, impratio=100, cone="elliptic")),  # 仿真步长5ms
        viewer=ViewerConfig(body_name=BASE_BODY_NAME, distance=3.0, elevation=-20.0, azimuth=45.0),  # 查看器视角
    )


# ======================================================================
# 平地环境：基础行走训练与评估
# ======================================================================
def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """平地训练/评估环境配置
    Args:
        play: 是否为回放模式；回放模式下关闭随机化、课程，延长回合时间
    Returns:
        ManagerBasedRlEnvCfg: 平地环境配置
    """
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}  # 加载站立姿态机器人配置

    # 活动环境中分配独立后退组，保持当前航向并采样低速负向指令。
    cfg.commands["twist"].rel_backward_envs = 0.20
    cfg.commands["twist"].backward_velocity_range = (-0.4, -0.2)

    # 平移时抑制垂直于指令方向的串速，前进指令下即约束横向速度。
    cfg.rewards["command_orthogonal_drift"] = RewardTermCfg(
        func=command_orthogonal_drift_penalty,
        weight=-1.5,
        params={
            "command_name": "twist",
            "minimum_command_speed": 0.06,
            "speed_tolerance": 0.04,
            "deviation_scale": 0.15,
        },
    )

    # 四个髋外展关节允许小幅调整，持续偏离标准站姿时按超出角度的平方扣分。
    cfg.rewards["hip_abduction_excess"] = RewardTermCfg(
        func=hip_abduction_excess,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_abduction_joint",)),
            "tolerance": 0.12,
            "deviation_scale": 0.25,
        },
    )
    # 静止命令下要求四足接地，避免策略用三足支撑换取局部奖励。
    cfg.rewards["feet_contact_without_cmd"] = RewardTermCfg(
        func=feet_contact_without_cmd,
        weight=0.25,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    )
    # 静止时更严格地拉回标准关节姿态，运动时保留步态所需的关节活动空间。
    cfg.rewards["static_joint_posture"] = RewardTermCfg(
        func=joint_pos_penalty,
        weight=-0.15,
        params={
            "command_name": "twist",
            "stand_still_scale": 8.0,
            "velocity_threshold": 0.25,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS),
        },
    )
    # 只在静止命令下约束左右腿镜像，避免干扰运动时的交替摆腿。
    cfg.rewards["static_leg_symmetry"] = RewardTermCfg(
        func=joint_mirror,
        weight=-0.35,
        params={
            "mirror_joints": [
                ["fl_(hip_pitch|knee)_joint", "fr_(hip_pitch|knee)_joint"],
                ["rl_(hip_pitch|knee)_joint", "rr_(hip_pitch|knee)_joint"],
            ],
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )
    # 左右髋外展角异号才是镜像，防止同向偏腿被当成对称姿态。
    cfg.rewards["static_abduction_symmetry"] = RewardTermCfg(
        func=joint_mirror,
        weight=-0.35,
        params={
            "mirror_joints": [
                ["fl_hip_abduction_joint", "fr_hip_abduction_joint"],
                ["rl_hip_abduction_joint", "rr_hip_abduction_joint"],
            ],
            "mirror_sign": -1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )
    # 运动时限制单条腿连续悬空，保留正常摆腿但阻止三足退化步态。
    cfg.rewards["long_foot_air_time"] = RewardTermCfg(
        func=long_foot_air_time_penalty,
        weight=-3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "max_air_time": 0.80,
            "command_threshold": 0.1,
        },
    )
    # 扩大有效摆腿时间窗口，鼓励跨步而非仅靠短促碎步前进。
    cfg.rewards["feet_air_time"].weight = 0.4
    cfg.rewards["feet_air_time"].params["threshold_max"] = 0.7
    cfg.rewards["feet_air_time"].func = supported_feet_air_time
    cfg.rewards["feet_air_time"].params["min_support_feet"] = 2
    # 平地步行要求至少两足支撑，单足支撑和四足腾空持续扣分。
    cfg.rewards["support_deficit"] = RewardTermCfg(
        func=support_deficit_penalty,
        weight=-2.0,
        params={"sensor_name": "feet_ground_contact", "min_support_feet": 2},
    )
    # 摆腿达到目标高度并持续支撑后结算，降低频繁点地的收益。
    cfg.rewards["foot_swing_height"] = RewardTermCfg(
        func=FootSwingHeightReward,
        weight=0.08,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "target_height": 0.10,
            "height_std": 0.05,
            "command_threshold": 0.1,
            "ground_sensor_name": "base_height_sensor",
            "min_support_feet": 2,
            "min_stance_time": 0.16,
            "min_cycle_time": 0.50,
        },
    )
    # 短暂触地后立即抬脚会扣分，推动足端形成连续支撑相。
    cfg.rewards["short_stance"] = RewardTermCfg(
        func=short_stance_penalty,
        weight=-0.10,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "min_stance_time": 0.16},
    )
    # 直线运动中比较左右腿跨周期支撑占比，允许交替摆腿及短时差异。
    cfg.rewards["stance_balance"] = RewardTermCfg(
        func=StanceBalancePenalty,
        weight=-0.5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    # 支撑足相对地面尽量保持静止，避免用刨地代替承重推进。
    cfg.rewards["feet_slip"] = RewardTermCfg(
        func=velocity_mdp.feet_slip,
        weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True),
        },
    )
    # 直行时约束前后足对的横向落点中心，保留交替步态与足间宽度调整。
    cfg.rewards["foot_lateral_placement"] = RewardTermCfg(
        func=FootLateralPlacementPenalty,
        weight=-0.5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    
    if play:
        configure_evaluation(cfg)
    return cfg


# ======================================================================
# 复杂地形环境：越野障碍训练
# ======================================================================
def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建随机粗糙地形和随机凸块的迁移训练与评估配置。"""
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}  # 加载站立姿态机器人
    tile_edge_width = 0.05
    bump_grid_width = (8.0 - 2.0 * tile_edge_width) / 17

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(8.0, 8.0),
            border_width=2.0,
            num_rows=10,
            num_cols=2,
            layout_shape=(4, 4),
            curriculum=True,
            shuffle_curriculum_columns=True,
            difficulty_range=(0.15, 1.0),
            sub_terrains={
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.52,
                    noise_range=(-0.085, 0.085),
                    noise_step=0.01,
                    horizontal_scale=0.10,
                    downsampled_scale=0.20,
                    border_width=0.0,
                    base_thickness_ratio=100.0,
                    scale_with_difficulty=True,
                    size=(8.0, 8.0),
                ),
                "low_random_bumps": BoxRandomGridTerrainCfg(
                    proportion=0.48,
                    grid_width=bump_grid_width,
                    grid_height_range=(0.0, 0.085),
                    border_width=tile_edge_width,
                    platform_width=3.0 * bump_grid_width,
                    size=(8.0, 8.0),
                ),
            },
        ),
        max_init_terrain_level=2,
    )

    # 移除默认速度课程，自定义地形与命令课程
    # 注：基础模板的 curriculum 本来是空字典，这一句是为兼容 mjlab 默认速度任务
    # 可能注入的 command_vel 课程而写的防御性删除，没有也不影响。
    cfg.curriculum.pop("command_vel", None)

    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=rough_terrain_levels,
        params={
            "command_name": "twist",
            "initial_level_cap": 2,
            "window_size": 100,
            "success_threshold": 0.80,
            "success_progress_ratio": 0.60,
            "demote_progress_ratio": 0.40,
            "minimum_command_speed": 0.15,
        },
    )

    # 速度指令沿用平地的全范围采样，仅地形保留表现型课程。
    cfg.commands["twist"].heading_command = True
    cfg.commands["twist"].rel_heading_envs = 1.0
    cfg.commands["twist"].heading_control_stiffness = 0.6
    cfg.commands["twist"].ranges.heading = (-math.pi, math.pi)
    cfg.commands["twist"].resampling_time_range = (10.0, 10.0)
    cfg.commands["twist"].rel_standing_envs = 0.15
    cfg.commands["twist"].rel_forward_envs = 0.40
    cfg.commands["twist"].rel_backward_envs = 0.20
    cfg.commands["twist"].backward_velocity_range = (-0.4, -0.2)
    cfg.commands["twist"].rel_lateral_envs = 0.35
    cfg.commands["twist"].rel_yaw_envs = 0.35
    cfg.commands["twist"].flat_lateral_probability = 0.35
    cfg.commands["twist"].flat_yaw_probability = 0.35
    cfg.commands["twist"].ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)
    # 小幅速度命令保留用于慢速机动，明确静止组仍发送零指令。
    cfg.commands["twist"].linear_deadband = 0.02
    cfg.commands["twist"].angular_deadband = math.radians(2.0)

    # ====================== 重置与启动随机化调整 ======================
    cfg.events.pop("joint_friction", None)
    # 重置事件：关节状态重置
    cfg.events["reset_joints"] = EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS),
        },
    )
    
    # 重置高度沿用机器人默认姿态，pose_range 仅提供相对默认位置的偏移。
    cfg.events["reset_base"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"z": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 推力扰动：调整为5~10s一次
    cfg.events["push_robot"] = EventTermCfg(
        func=envs_mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "asset_cfg": SceneEntityCfg("robot")},
    )

    # ====================== 奖励函数调整 ======================
    # 移除基础速度跟踪奖励，替换为分轴独立跟踪奖励
    cfg.rewards.pop("track_lin_vel", None)
    cfg.rewards.pop("track_ang_vel", None)

    # X轴线速度跟踪奖励
    cfg.rewards["track_lin_vel_x_exp"] = RewardTermCfg(
        func=track_linear_velocity_x,
        weight=1.0,
        params={"std": 0.25, "command_name": "twist"},
    )
    # Y轴线速度跟踪奖励
    cfg.rewards["track_lin_vel_y_exp"] = RewardTermCfg(
        func=track_linear_velocity_y,
        weight=1.0,
        params={"std": 0.25, "command_name": "twist"},
    )
    # Z轴角速度跟踪奖励
    cfg.rewards["track_ang_vel_z_exp"] = RewardTermCfg(
        func=track_angular_velocity_z,
        weight=1.0,
        params={"std": 0.25, "command_name": "twist"},
    )
    cfg.rewards["command_aligned_progress"] = RewardTermCfg(
        func=command_aligned_progress_reward,
        weight=1.0,
        params={"command_name": "twist", "minimum_command_speed": 0.06},
    )
    # 有移动指令却无法前进时扣分，避免用高跷静止姿态规避粗糙地形。
    cfg.rewards["command_progress_deficit"] = RewardTermCfg(
        func=command_progress_deficit_penalty,
        weight=-3.0,
        params={
            "command_name": "twist",
            "minimum_command_speed": 0.06,
            "minimum_progress_ratio": 0.6,
        },
    )
    # 按指令方向的垂线计算串速，不惩罚正确的斜向移动。
    cfg.rewards["command_orthogonal_drift"] = RewardTermCfg(
        func=command_orthogonal_drift_penalty,
        weight=-1.5,
        params={
            "command_name": "twist",
            "minimum_command_speed": 0.06,
            "speed_tolerance": 0.04,
            "deviation_scale": 0.15,
        },
    )
    cfg.rewards["translation_stance_overstay"] = RewardTermCfg(
        func=TranslationStanceOverstayPenalty,
        weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "minimum_command_speed": 0.06,
            "stance_distance": 0.18,
            "minimum_stance_limit": 0.30,
            "maximum_stance_limit": 1.20,
            "deviation_scale": 0.30,
            "minimum_air_time": 0.08,
        },
    )
    cfg.rewards["translation_foot_workspace"] = RewardTermCfg(
        func=TranslationFootWorkspacePenalty,
        weight=-0.4,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "minimum_command_speed": 0.06,
            "nominal_forward": 0.035,
            "nominal_outward": 0.105,
            "forward_margin": 0.24,
            "lateral_margin": 0.12,
            "deviation_scale": 0.08,
        },
    )
    
    # 惩罚：机身Z向速度（上下颠簸）
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=lin_vel_z_l2, weight=-2.0)
    # 惩罚：机身XY平面角速度
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(func=ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")})

    cfg.rewards["upright"].weight = 0.5
    # 前后俯仰超过 5 度后持续扣分，允许小幅地形适应。
    cfg.rewards["pitch_excess"] = RewardTermCfg(
        func=pitch_excess_penalty,
        weight=-0.5,
        params={
            "tolerance_degrees": 5.0,
            "deviation_scale_degrees": 10.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards.pop("roll_penalty", None)

    # 移除地形等级奖励与动作率课程奖励
    cfg.rewards.pop("terrain_level_bonus", None)
    cfg.rewards.pop("action_rate_curriculum", None)
    # 调整动作变化率惩罚权重
    cfg.rewards["action_rate"].weight = -0.01
    # 调整关节力矩惩罚权重
    cfg.rewards["joint_torques"].weight = -2.5e-5
    # 新增：关节功率惩罚
    cfg.rewards["joint_power"] = RewardTermCfg(func=joint_power, weight=-2.0e-5)

    cfg.rewards.pop("joint_acc", None)
    # 分关节加速度惩罚：仅作用于 12 个腿部关节。
    cfg.rewards["leg_joint_acc_l2"] = RewardTermCfg(
        func=envs_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
    )

    # 调整关节位置超限惩罚
    cfg.rewards["joint_pos_limits"].weight = -5.0
    cfg.rewards.pop("leg_motion_penalty", None)
    cfg.rewards["is_terminated"].weight = -50.0
    cfg.rewards.pop("leg_symmetry", None)

    # 移除静止奖励陷阱，替换为静止惩罚
    cfg.rewards.pop("stand_still", None)
    cfg.rewards["stand_still"] = RewardTermCfg(func=stand_still, weight=-2.0, params={"command_name": "twist", "command_threshold": 0.02})

    cfg.rewards.pop("hip_deviation", None)
    cfg.rewards.pop("variable_posture", None)
    # 髋外展允许偏离默认姿态 12 度，超过后以弱惩罚抑制持续侧偏。
    cfg.rewards["hip_abduction_excess"] = RewardTermCfg(
        func=hip_abduction_excess,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_abduction_joint",)),
            "tolerance": math.radians(12.0),
            "deviation_scale": 0.25,
        },
    )
    # 膝角超过 -25° 后按超出角度平方扣分，正常弯腿不受限制。
    cfg.rewards["knee_extension"] = RewardTermCfg(
        func=knee_extension_penalty,
        weight=-0.5,
        params={
            "maximum_angle": math.radians(-25.0),
            "transition_width": math.radians(10.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
        },
    )

    # 奖励：无命令时足端接触
    cfg.rewards["feet_contact_without_cmd"] = RewardTermCfg(
        func=feet_contact_without_cmd,
        weight=0.1,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact", "command_threshold": 0.02}
    )
    # 降低摆腿时间奖励，避免用大幅前摆和长时间悬空换取奖励。
    cfg.rewards["feet_air_time"].weight = 0.05
    # 至少两足支撑时才给予摆腿时间奖励。
    cfg.rewards["feet_air_time"].func = supported_feet_air_time
    cfg.rewards["feet_air_time"].params["min_support_feet"] = 2
    cfg.rewards.pop("base_height_l2", None)
    cfg.rewards["local_ground_clearance"] = RewardTermCfg(
        func=base_clearance_outside_range_l2,
        weight=-20.0,
        params={
            "minimum_height": 0.28,
            "maximum_height": 0.44,
            "sensor_name": "base_height_sensor",
        },
    )
    cfg.rewards["support_deficit"] = RewardTermCfg(
        func=support_deficit_penalty,
        weight=-0.5,
        params={"sensor_name": "feet_ground_contact", "min_support_feet": 2},
    )
    # 直行时比较跨周期左右支撑占比，容许短时地形适应和交替摆腿。
    cfg.rewards["stance_balance"] = RewardTermCfg(
        func=StanceBalancePenalty,
        weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "smoothing_time": 1.0,
            "warmup_time": 1.0,
            "duty_tolerance": 0.15,
            "deviation_scale": 0.25,
            "command_threshold": 0.1,
            "lateral_threshold": 0.1,
            "yaw_threshold": 0.15,
        },
    )
    # 短暂点地后立即离地时扣分，允许粗糙地形上的快速换脚。
    cfg.rewards["short_stance"] = RewardTermCfg(
        func=short_stance_penalty,
        weight=-0.03,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "min_stance_time": 0.10,
        },
    )
    # 单足悬空超过正常摆腿时长时扣分，防止三足静止并长期抬起一条腿。
    cfg.rewards["long_foot_air_time"] = RewardTermCfg(
        func=long_foot_air_time_penalty,
        weight=-1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "max_air_time": 0.80,
            "command_threshold": 0.1,
        },
    )
    # 直行时限制触地脚过度前伸，保持可实现的正常跨步长度。
    cfg.rewards["forward_foot_touchdown"] = RewardTermCfg(
        func=ForwardFootTouchdownPenalty,
        weight=-0.08,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "nominal_foot_x": (0.43, 0.43, -0.05, -0.05),
            "forward_margin": 0.20,
            "deviation_scale": 0.08,
            "command_threshold": 0.15,
            "lateral_threshold": 0.10,
            "yaw_threshold": 0.15,
        },
    )
    # 原地转向时约束撑地扭身和足端活动范围，保留正常支撑阶段。
    turn_gate = {
        "command_name": "twist",
        "minimum_yaw_degrees": 8.0,
        "maximum_linear_speed": 0.08,
    }
    cfg.rewards["turn_stance_twist"] = RewardTermCfg(
        func=TurnStanceTwistPenalty,
        weight=-0.75,
        params={
            **turn_gate,
            "sensor_name": "feet_ground_contact",
            "tolerance_degrees": 12.0,
            "deviation_scale_degrees": 10.0,
            "minimum_air_time": 0.08,
        },
    )
    cfg.rewards["turn_foot_workspace"] = RewardTermCfg(
        func=TurnFootWorkspacePenalty,
        weight=-0.4,
        params={
            **turn_gate,
            "sensor_name": "feet_ground_contact",
            "nominal_forward": 0.035,
            "nominal_outward": 0.105,
            "forward_margin": 0.24,
            "lateral_margin": 0.12,
            "deviation_scale": 0.08,
        },
    )
    cfg.rewards["turn_progress_deficit"] = RewardTermCfg(
        func=turn_progress_deficit_penalty,
        weight=-1.5,
        params={**turn_gate, "minimum_progress_ratio": 0.6},
    )
    cfg.rewards["turn_linear_drift"] = RewardTermCfg(
        func=turn_linear_drift_penalty,
        weight=-0.5,
        params={**turn_gate, "speed_tolerance": 0.05, "deviation_scale": 0.15},
    )
    for direction, sign in (("left", 1), ("right", -1)):
        cfg.metrics[f"turn_{direction}_yaw_error_deg_s"] = MetricsTermCfg(
            func=turn_yaw_error_degrees, params={**turn_gate, "sign": sign}
        )
    # 移动和转向时持续惩罚悬空足过度前伸，接地足不受本项约束。
    cfg.rewards["forward_foot_swing"] = RewardTermCfg(
        func=ForwardFootSwingPenalty,
        weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "nominal_foot_x": (0.43, 0.43, -0.05, -0.05),
            "forward_margin": 0.20,
            "deviation_scale": 0.08,
            "command_threshold": 0.1,
        },
    )
    cfg.rewards["feet_slip"] = RewardTermCfg(
        func=velocity_mdp.feet_slip,
        weight=-0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True),
        },
    )
    # 惩罚：接触力过大
    cfg.rewards["contact_forces"] = RewardTermCfg(func=contact_forces, weight=-1.5e-4, params={"sensor_name": "feet_ground_contact", "threshold": 100.0})

    # 移除不适用的奖励项
    for key in ("body_ang_vel", "terrain_level_bonus", "action_rate_curriculum"):
        cfg.rewards.pop(key, None)

    cfg.episode_length_s = 20.0
    # 仿真配置调整：增加接触传感器匹配数、迭代次数
    # contact_sensor_maxmatch=128：MuJoCo 默认的接触检测上限较小，地形复杂时
    # 每步的接触点会超过默认值而被丢弃，表现为"偶尔穿模"。调到 128 是为了兜住这种情况。
    # ccd_iterations=80：连续碰撞检测迭代次数，越障时足端高速撞墙容易穿透，
    # 加大迭代能减少穿透，代价是每步仿真变慢。
    cfg.sim = SimulationCfg(contact_sensor_maxmatch=128, mujoco=MujocoCfg(timestep=0.005, impratio=100, cone="elliptic", ccd_iterations=80))

    # 移除姿态终止条件，允许机器人学习从倾倒恢复
    cfg.seed = 42  # 固定随机种子
                   # 固定种子使地形生成、指令采样、域随机化都可复现，
                   # 便于对比两次实验的差异到底是改动引起的还是随机性引起的。

    # 地形并行环境数量与间距
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = 2048
        cfg.scene.terrain.env_spacing = 2.5

    # ====================== 评估指标扩展 ======================
    # metrics 与 rewards 的区别：metrics 不参与反向传播，只把数值写进 TensorBoard，
    # 因此可以放心地写"信息量大但不可微"的统计量（比如条件均值、符号一致率）。
    # 这一组指标的设计意图是：当 mean_reward 不涨时，能立刻定位到
    # "是哪个速度段跟不上 / 是哪个轴跟得差 / 还是频繁摔倒重置"。
    cfg.metrics.update(
        {
            "terrain_type": MetricsTermCfg(func=terrain_type_metric),
            "terrain_level": MetricsTermCfg(func=terrain_level_metric),
            "effective_progress": MetricsTermCfg(
                func=command_aligned_progress_metric, params={"command_name": "twist"}
            ),
            "feet_slip": MetricsTermCfg(
                func=velocity_mdp.feet_slip,
                params={
                    "sensor_name": "feet_ground_contact",
                    "command_name": "twist",
                    "command_threshold": 0.1,
                    "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES, preserve_order=True),
                },
            ),
            "contact_overload": MetricsTermCfg(
                func=contact_forces,
                params={"sensor_name": "feet_ground_contact", "threshold": 100.0},
            ),
            "joint_power": MetricsTermCfg(func=joint_power),
            "failure_bad_orientation": MetricsTermCfg(
                func=termination_reason_metric, params={"term_name": "bad_orientation"}
            ),
            "failure_base_contact": MetricsTermCfg(
                func=termination_reason_metric, params={"term_name": "base_ground_contact"}
            ),
            "failure_joint_limit": MetricsTermCfg(
                func=termination_reason_metric, params={"term_name": "joint_pos_out_of_limits"}
            ),
            "failure_height": MetricsTermCfg(
                func=termination_reason_metric, params={"term_name": "base_height_out_of_range"}
            ),
            "failure_nan": MetricsTermCfg(
                func=termination_reason_metric, params={"term_name": "nan_detection"}
            ),
            # 速度跟踪误差
            "tracking_lin_vel_error": MetricsTermCfg(func=tracking_lin_vel_error, params={"command_name": "twist"}),
            "tracking_lin_vel_x_error": MetricsTermCfg(func=tracking_lin_vel_x_error, params={"command_name": "twist"}),
            "tracking_lin_vel_y_error": MetricsTermCfg(func=tracking_lin_vel_y_error, params={"command_name": "twist"}),
            "tracking_lin_vel_along_cmd_error": MetricsTermCfg(
                func=tracking_lin_vel_along_command_error, params={"command_name": "twist"}
            ),
            "actual_lin_vel_orthogonal_cmd": MetricsTermCfg(
                func=actual_lin_vel_orthogonal_command_mean, params={"command_name": "twist"}
            ),
            "tracking_yaw_vel_error": MetricsTermCfg(func=tracking_yaw_vel_error, params={"command_name": "twist"}),
            # 命令统计
            "cmd_lin_vel": MetricsTermCfg(func=command_lin_vel_mean, params={"command_name": "twist"}),
            "cmd_yaw_vel": MetricsTermCfg(func=command_yaw_vel_abs_mean, params={"command_name": "twist"}),
            "actual_lin_vel": MetricsTermCfg(func=actual_lin_vel_mean),
            # 分速度段跟踪误差
            "tracking_lin_vel_error_cmd_0_03": MetricsTermCfg(
                func=tracking_lin_vel_error_band_mean,
                params={"command_name": "twist", "min_speed": 0.0, "max_speed": 0.3},
            ),
            "tracking_lin_vel_error_cmd_03_07": MetricsTermCfg(
                func=tracking_lin_vel_error_band_mean,
                params={"command_name": "twist", "min_speed": 0.3, "max_speed": 0.7},
            ),
            "tracking_lin_vel_error_cmd_07_up": MetricsTermCfg(
                func=tracking_lin_vel_error_band_mean,
                params={"command_name": "twist", "min_speed": 0.7, "max_speed": 10.0},
            ),
            # 分轴分速度段跟踪误差
            "tracking_lin_vel_x_error_cmd_0_03": MetricsTermCfg(
                func=tracking_lin_vel_axis_error_band_mean,
                params={"axis": 0, "command_name": "twist", "min_speed": 0.0, "max_speed": 0.3},
            ),
            "tracking_lin_vel_x_error_cmd_03_07": MetricsTermCfg(
                func=tracking_lin_vel_axis_error_band_mean,
                params={"axis": 0, "command_name": "twist", "min_speed": 0.3, "max_speed": 0.7},
            ),
            "tracking_lin_vel_x_error_cmd_07_up": MetricsTermCfg(
                func=tracking_lin_vel_axis_error_band_mean,
                params={"axis": 0, "command_name": "twist", "min_speed": 0.7, "max_speed": 10.0},
            ),
            "tracking_lin_vel_y_error_cmd_0_03": MetricsTermCfg(
                func=tracking_lin_vel_axis_error_band_mean,
                params={"axis": 1, "command_name": "twist", "min_speed": 0.0, "max_speed": 0.3},
            ),
            "tracking_lin_vel_y_error_cmd_03_07": MetricsTermCfg(
                func=tracking_lin_vel_axis_error_band_mean,
                params={"axis": 1, "command_name": "twist", "min_speed": 0.3, "max_speed": 0.7},
            ),
            "tracking_lin_vel_y_error_cmd_07_up": MetricsTermCfg(
                func=tracking_lin_vel_axis_error_band_mean,
                params={"axis": 1, "command_name": "twist", "min_speed": 0.7, "max_speed": 10.0},
            ),
            # 命令速度段激活统计
            "cmd_band_0_03": MetricsTermCfg(
                func=command_band_active, params={"command_name": "twist", "min_speed": 0.0, "max_speed": 0.3}
            ),
            "cmd_band_03_07": MetricsTermCfg(
                func=command_band_active, params={"command_name": "twist", "min_speed": 0.3, "max_speed": 0.7}
            ),
            "cmd_band_07_up": MetricsTermCfg(
                func=command_band_active, params={"command_name": "twist", "min_speed": 0.7, "max_speed": 10.0}
            ),
            # 姿态与接触统计
            "upright": MetricsTermCfg(func=upright_metric),
            "base_ground_contact_rate": MetricsTermCfg(
                func=base_ground_contact_metric, params={"sensor_name": "base_ground_contact"}
            ),
        }
    )

    # 记录实际生效的六方向指令占比，混合指令可同时计入多个方向。
    for direction, axis, sign in (
        ("forward", 0, 1), ("backward", 0, -1),
        ("left", 1, 1), ("right", 1, -1),
        ("turn_left", 2, 1), ("turn_right", 2, -1),
    ):
        cfg.metrics[f"cmd_{direction}_active"] = MetricsTermCfg(
            func=command_direction_active,
            params={"command_name": "twist", "axis": axis, "sign": sign},
        )

    # 回放在重置实体状态前切换地形；固定评估通过独立入口冻结测试用例。
    if play:
        configure_evaluation(cfg)
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.events = {
            "randomize_terrain": EventTermCfg(
                func=envs_mdp.randomize_terrain, mode="reset"
            ),
            **cfg.events,
        }
    return cfg


def elevation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """前进方向的上下楼梯和坡面训练，保留原有本体观测接口。"""
    from .elevation import make_elevation_env_cfg

    return make_elevation_env_cfg(jump=False, play=play)


def bridge_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """单木桥环境占位，统一保留训练与回放接口。

    待实现桥坐标系误差、足端边缘余量、短边进出和掉桥判定，
    正常入口出口接地与桥内失足分开处理，初版不安排主动跳跃。
    完成后在 TASK_SPECS 中启用 PointLeg-Bridge-v0。
    """
    raise NotImplementedError("PointLeg-Bridge-v0 的地形、规则与奖励尚未实现")


def stepping_stones_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """梅花桩精准踏点训练环境。"""
    from .stepping_stones import make_stepping_stones_env_cfg

    return make_stepping_stones_env_cfg(play=play)


def jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """触发后固定前跳，跨平台间隙及中墙，落地四足稳定。"""
    from .jump import make_jump_env_cfg

    return make_jump_env_cfg(play=play)


# ======================================================================
# 匍匐钻杆环境：低矮障碍穿越训练
# ======================================================================
def crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """匍匐钻杆训练/评估环境配置
    针对低矮障碍穿越场景，机器人压低机身以蹲伏姿态通过横杠

    与 flat/rough 最本质的区别：本环境的目标不是"走得好"，而是"趴得低还能走"。
    因此奖励上拆掉了所有"要求姿态标准"的项（joint_deviation 之类改成"要求贴住匍匐姿态"），
    终止条件也放宽：姿态角阈值放到 80°，并移除了底盘触地的终止判定
    （否则机器人压低机身时底盘蹭地就直接判失败，根本学不会钻杆）。

    Args:
        play: 是否为回放模式
    Returns:
        ManagerBasedRlEnvCfg: 匍匐环境配置
    """
    cfg = _make_base_env_cfg()
    cfg.events["positive_rewards"] = EventTermCfg(func=configure_positive_rewards, mode="startup")
    cfg.scene.entities = {"robot": get_robot_crawl_cfg()}  # 加载匍匐姿态机器人配置

    # ====================== 地形配置 ======================
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20, curriculum=True,
            layout_shape=(4, 4),
            sub_terrains={
                "flat": BoxFlatTerrainCfg(proportion=0.25),  # 平地25%
                # 自定义低杆障碍：占比35%，离地间隙0.24~0.32m
                "rc_low_bar": RCLowBarTerrainCfg(proportion=0.35, clearance_range=(0.24, 0.32)),
                # 随机网格：占比20%，高度0~0.05m
                "random_grid": BoxRandomGridTerrainCfg(proportion=0.20, grid_width=0.45, grid_height_range=(0.0, 0.05)),
                # 柏林噪声：占比20%，高度0~0.05m
                "perlin_noise": HfPerlinNoiseTerrainCfg(proportion=0.20, height_range=(0.0, 0.05), octaves=2, persistence=0.4, lacunarity=2.0, horizontal_scale=0.20, resolution=0.20, border_width=0.50, base_thickness_ratio=100.0),
            },
        ),
        max_init_terrain_level=0,  # 初始地形难度等级0
    )

    # 命令配置调整
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=velocity_mdp.terrain_levels_vel, params={"command_name": "twist"})
    cfg.commands["twist"].heading_command = False  # 关闭航向命令，直线穿越
    cfg.commands["twist"].rel_heading_envs = 0.0
    cfg.commands["twist"].ranges.heading = None
    cfg.commands["twist"].rel_standing_envs = 0.05   # 静止命令占比5%
    cfg.commands["twist"].rel_forward_envs = 0.40    # 纯前向命令占比40%，低杆地形以直穿为主

    # ====================== 事件与重置配置 ======================
    # 启动事件：关节摩擦随机化
    cfg.events["joint_friction"] = EventTermCfg(func=envs_dr.joint_friction, mode="startup", params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.7, 1.3), "operation": "scale"})
    # 重置事件：关节位置小范围随机偏移
    cfg.events["reset_joints"] = EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS),
        },
    )
    
    # 重置事件：机器人初始位姿（匍匐低高度）
    cfg.events["reset_base"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"z": (0.00, 0.04), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 推力扰动：力度减小，适合匍匐场景
    cfg.events["push_robot"] = EventTermCfg(
        func=envs_mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, "asset_cfg": SceneEntityCfg("robot")},
    )

    # ====================== 奖励函数调整 ======================
    # 主奖励权重调整：线速度权重提高，角速度权重降低
    cfg.rewards["track_lin_vel"].weight = 3.0
    cfg.rewards["track_lin_vel"].params["std"] = 0.5
    cfg.rewards["track_ang_vel"].weight = 1.5
    cfg.rewards["track_ang_vel"].params["std"] = 0.5
    
    # 惩罚：Z向速度
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=lin_vel_z_l2, weight=-1.0)
    # 惩罚：机身XY角速度
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(func=velocity_mdp.body_angular_velocity_penalty, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))})
    
    # 直立奖励权重调整
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = 0.5
    # 动作变化率惩罚减轻
    cfg.rewards["action_rate"].weight = -0.001
    # 关节力矩惩罚减轻
    cfg.rewards["joint_torques"].weight = -1e-4
    # 关节加速度惩罚置0
    cfg.rewards["joint_acc"].weight = 0.0
    # 关节位置超限惩罚减轻
    cfg.rewards["joint_pos_limits"].weight = -1.0
    # 腿部运动惩罚加重，保持匍匐姿态稳定
    cfg.rewards["leg_motion_penalty"].weight = -5.0
    # 终止惩罚减轻
    cfg.rewards["is_terminated"].weight = -50.0

    # 匍匐高度奖励：机身低于0.22m时获得最大奖励
    cfg.rewards.pop("base_height_l2", None)
    cfg.rewards["crawl_height_reward"] = RewardTermCfg(
        func=crawl_height_reward,
        weight=1.5,
        params={"target_height": 0.22, "std": 0.05, "sensor_name": "base_height_sensor"}
    )

    cfg.rewards.pop("stand_still", None)
    
    cfg.rewards.pop("hip_deviation", None)
    # 腿部关节偏离惩罚：加重，约束匍匐姿态
    cfg.rewards["leg_joint_deviation"] = RewardTermCfg(
        func=joint_deviation_l2,
        weight=-15.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
    )

    # 机身碰撞惩罚减轻，匍匐允许轻微蹭地
    if "body_collision" in cfg.rewards:
        cfg.rewards["body_collision"].weight = -0.1
    # 惩罚：机身水平姿态
    cfg.rewards["flat_orientation"] = RewardTermCfg(func=flat_orientation_l2, weight=-1.0)

    # 移除不适用的奖励项
    for key in ("feet_air_time", "body_ang_vel"):
        cfg.rewards.pop(key, None)

    cfg.episode_length_s = 30.0  # 匍匐回合时间延长
    # 仿真配置调整
    cfg.sim = SimulationCfg(contact_sensor_maxmatch=128, mujoco=MujocoCfg(timestep=0.005, impratio=100, cone="elliptic", ccd_iterations=80))

    # 放宽姿态终止阈值：80度，允许匍匐大角度倾斜
    # 底盘触地终止被移除的代价是"无法用摔倒判定结束 episode"，
    # 姿态阈值与局部地面高度、关节限位一起约束失稳状态。
    cfg.terminations["bad_orientation"].params["limit_angle"] = math.radians(80.0)
    cfg.terminations["base_height_out_of_range"].params.update(
        {"minimum_height": 0.10, "maximum_height": 0.38}
    )
    
    # 移除底盘接地终止条件，方便钻杆时低姿态通过
    cfg.terminations.pop("base_ground_contact", None)

    # 回放模式配置
    if play:
        configure_evaluation(cfg)
    return cfg

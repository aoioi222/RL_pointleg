# =============================================================================
# rewards.py — 纯点足四足机器人训练用 MDP 奖励函数库
#
# 【模块定位】
#   本文件是 mdp 组件库中的「奖励（Reward）」侧，向上挂载到 RewardManager，
#   在 env_cfgs.py 的 rewards / metrics 段中以函数名 + 参数被引用，例如：
#       cfg.rewards["track_lin_vel"] = RewardTermCfg(
#           func=track_linear_velocity, weight=2.5, params={...})
#
# 【函数分类】（按功能分 5 组）
#   1) 速度跟踪奖励（主任务）  : track_linear_velocity(_x/_y), track_angular_velocity(_z)
#   2) 姿态/高度/运动惩罚      : base_height_l2, flat_orientation_l2, lin_vel_z_l2,
#                                ang_vel_xy_l2, joint_deviation_l2, joint_power ...
#   3) 足式运动约束            : feet_clearance, soft_landing,
#                                contact_fraction_reward, adaptive_leg_motion_penalty ...
#   4) 模式条件化奖励          : get_mode_id, crouch_height_reward,
#                                mode_conditioned_posture, crawl_height_reward ...
#   5) 诊断指标（Metrics）     : tracking_lin_vel_error 等（喂给 TensorBoard）
#
# 【通用约定】
#   - 每个函数签名 = (env, 具名参数...)，由 RewardManager 从 RewardTermCfg.params 解包调用；
#   - 返回 [num_envs] 的 tensor（原始奖励值，不与 weight 相乘，weight 在配置里乘）；
#   - 大量函数末尾乘以 `-projected_gravity_b[:,2]` 作为「直立加权」：
#     机器人越接近直立，权重越大；快摔倒时该项趋近 0，避免惩罚信号误导策略；
#   - _finite() 把 NaN/Inf 替换为 0，防止终止瞬间非法物理状态污染训练统计。
# =============================================================================

"""MDP Reward functions for unitree locomotion training."""

# ═══════════════════════════════════════════════════════════════════════════
# 【本文件的关键概念速查（读代码前先看这段）】
#
# 1. 直立加权因子  clamp(-projected_gravity_b[:,2], 0, 0.7) / 0.7
#    projected_gravity_b 是"重力方向在机身坐标系下的投影"，机身完全水平时它 = (0,0,-1)，
#    所以 -g_z 在直立时为 1、翻倒时为 0。乘上它以后，所有姿态/平滑类惩罚
#    在机器人快摔倒时自动衰减到 0 —— 因为那些动作本来就是摔倒过程中不可避免的，
#    硬罚会让策略学不到"如何摔"以外的信息。为什么上限取 0.7 而不是 1.0：
#    留一点余量，避免完全直立时惩罚过于尖锐。
#
# 2. 指数型奖励  exp(-err / std^2)
#    误差为 0 时奖励 = 1，误差增大按高斯衰减。std 是"容差"：
#    std 小 = 要求精确；std 大 = 允许粗糙。本文件所有跟踪类奖励都是这个形式，
#    区别只在"跟什么、哪个轴、用什么误差范数（L1/L2）"。
#
# 3. 只在"有指令"时生效的惩罚
#    很多惩罚会先取指令强度 total = |v_xy| + |ω_z|，再用 total > threshold 做掩码。
#    原因：机器人站着不动时腿本来就会微动，这时罚它等于逼它僵直；
#    只有"被要求运动却乱动"才是真的坏行为。
#
# 4. 地形掩码
#    terrain.terrain_types 是每个环境当前所处子地形的编号，与
#    terrain_generator.sub_terrains 的键顺序一一对应，因此可以按名字查到编号再比对。
#    本文件用它实现"只在台阶上罚横移"这类地形条件化惩罚。
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import TYPE_CHECKING
import math
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.entity import Entity
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.lab_api.math import quat_apply_inverse

from .height import root_ground_clearance
from ..robot_cfg import FOOT_BODY_NAMES, FOOT_SITE_NAMES, STANDING_BASE_HEIGHT
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _finite(value: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
    """Keep diagnostic metrics finite when a terminating env has invalid physics state."""
    # 把 NaN/±Inf 替换为指定值，保证诊断指标在终止瞬间（物理状态非法）仍可计算
    return torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 速度跟踪奖励（主任务项）
# ─────────────────────────────────────────────────────────────────────────────


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking commanded horizontal xy velocity.

    x、y 联合跟踪（一个误差项算两个轴）。相比 x/y 分轴版本，联合版不会出现
    "x 跟得准、y 完全跑偏"却仍拿高分的漏洞，但两个轴互相牵扯，调权重不够灵活。
    Rough 环境用的是分轴版（track_linear_velocity_x/y），Flat/Crawl 用这个联合版。
    """
    # 主奖励：跟踪指令的水平速度（x+y 联合）
    # 误差平方 / std^2 后取 exp：误差为 0 时奖励=1，误差越大指数衰减
    # 再乘直立加权：倾斜时奖励打折，防止"躺着也能刷速度分"的钻空子行为
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    reward = torch.exp(-xy_error / std**2)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def track_linear_velocity_x(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Independent x velocity tracking reward.

    gravity_z_power 用于改变直立加权的方式：
      - 传 None（默认）：加权系数 = (-g_z)，线性，直立为 1；
      - 传数值 p      ：加权系数 = (-g_z)^p。p=2 时更严厉（稍微倾斜就大幅掉分），
        p<1 时更宽容。爬楼梯/翻墙阶段需要允许大幅俯仰，通常给个小的 p。
    """
    # x 轴速度单独跟踪，与 y 轴解耦以便分别调节权重。
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    x_error = torch.square(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])
    reward = torch.exp(-x_error / std**2)
    if gravity_z_power is not None:
        # 显式指定重力投影幂次时：直立加权 = (-g_z)^power
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0) ** gravity_z_power
    else:
        reward *= -asset.data.projected_gravity_b[:, 2]
    return reward


def track_linear_velocity_y(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Independent y velocity tracking reward.

    横向运动通常比前向运动更难，因此可为 y 轴配置更低权重或更大容差。
    """
    # y 轴（横移）速度单独跟踪，逻辑同 track_linear_velocity_x
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    y_error = torch.square(command[:, 1] - asset.data.root_link_lin_vel_b[:, 1])
    reward = torch.exp(-y_error / std**2)
    if gravity_z_power is not None:
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0) ** gravity_z_power
    else:
        reward *= -asset.data.projected_gravity_b[:, 2]
    return reward


def track_angular_velocity_z(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Independent yaw velocity tracking reward.

    对比 track_angular_velocity（联合版）：这个版本没有 0.7 截断，且提供
    gravity_z_power 选项，方便在爬越阶段放宽对航向精度的要求。
    """
    # z 轴（偏航角速度）单独跟踪，逻辑同 x/y
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    z_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    reward = torch.exp(-z_error / std**2)
    if gravity_z_power is not None:
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0) ** gravity_z_power
    else:
        reward *= -asset.data.projected_gravity_b[:, 2]
    return reward


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking commanded yaw rate.

    带 0.7 截断的直立加权版：机身倾斜超过约 45° 后权重就压到 0，
    避免机器人"歪着原地打转"也拿满分。
    """
    # 主奖励：跟踪指令偏航角速度（与 track_angular_velocity_z 功能近似，带直立加权）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    reward = torch.exp(-z_error / std**2)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 2. 地形感知惩罚 / 高度控制
# ─────────────────────────────────────────────────────────────────────────────


def stair_lateral_yaw_drift_l2(
    env: ManagerBasedRlEnv,
    terrain_names: tuple[str, ...] = ("pyramid_stairs", "pyramid_stairs_inv"),
    y_scale: float = 1.0,
    yaw_scale: float = 1.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize sideways velocity and yaw-rate drift only on stair terrains."""
    # 台阶地形专属惩罚：只在爬楼梯时惩罚横移速度与偏航角速度
    # 与 commands.py 中"台阶上锁 y/z 指令"形成互补——指令锁了，实际速度也必须惩罚
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_cfg = getattr(terrain, "cfg", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_types is None or terrain_generator is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 构造台阶地形掩码：terrain_types 属于指定子地形之一即为 True
    from .terrain_utils import terrain_type_mask

    mask = terrain_type_mask(env, terrain_names)

    y_vel = asset.data.root_link_lin_vel_b[:, 1]
    yaw_vel = asset.data.root_link_ang_vel_b[:, 2]
    penalty = y_scale * torch.square(y_vel) + yaw_scale * torch.square(yaw_vel)
    return _finite(torch.where(mask, penalty, torch.zeros_like(penalty)))


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float = STANDING_BASE_HEIGHT,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize base height deviation from target height, relative to terrain or origin.

    为什么搞两套测量通道：
      激光高度扫描（height_scan）给出的是"机身正下方地形表面的相对高度"，
      在地形起伏时才是真实离地高度，但射到悬崖边缘外会返回无效值（很大或很小），
      因此要过滤有效射线；没有装扫描雷达的平地任务则退化成
      "root 的世界 z − 环境原点 z"，即相对出生平面的高度。

    注意 Rough 环境里这一项的 weight 被改成 0，实际由高度扫描配合
    base_height_l2 的 sensor_cfg 参数间接约束（见 env_cfgs.py）。
    """
    # 惩罚机身高度偏离目标值（双通道测量）：
    #   - 有高度传感器：用激光测距均值估计实际离地高度（过滤悬崖/贴地异常射线）
    #   - 无传感器    ：用 root_link 世界 z - 环境原点 z 近似
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    
    has_sensor = False
    if sensor_cfg is not None:
        try:
            _ = env.scene[sensor_cfg.name]
            has_sensor = True
        except KeyError:
            has_sensor = False
            
    if has_sensor:
        from mjlab.envs.mdp.observations import height_scan
        # Get height scanner readings (each ray's height relative to base frame Z)
        heights = height_scan(env, sensor_cfg.name, offset=0.0)
        # Clean any invalid values
        heights = torch.nan_to_num(heights, nan=target_height, posinf=target_height, neginf=target_height)
        
        # Filter out rays that represent extreme misses/cliffs (e.g. height < 0 or height > 1.5)
        # 仅将有效范围内的地面射线纳入平均高度。
        valid_mask = (heights > 0.0) & (heights < 1.5)
        
        # Calculate mean height above terrain ignoring misses
        sum_heights = torch.sum(torch.where(valid_mask, heights, torch.zeros_like(heights)), dim=1)
        count_valid = torch.sum(valid_mask.float(), dim=1)
        measured_height = torch.where(count_valid > 0, sum_heights / torch.clamp(count_valid, min=1.0), torch.full_like(sum_heights, target_height))
        
        error = measured_height - target_height
    else:
        if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
            env_origins_z = env.scene.env_origins[:, 2]
        else:
            env_origins_z = torch.zeros(env.num_envs, device=env.device)
        
        root_z = asset.data.root_link_pos_w[:, 2] - env_origins_z
        error = root_z - target_height
        
    reward = torch.square(error)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def safe_height_scan(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Safely fetch height scanner outputs, replacing NaNs/infs with finite values."""
    # 安全封装：高度扫描数据，NaN→0、+Inf→5、-Inf→-5（供观测/指标复用）
    from mjlab.envs.mdp.observations import height_scan
    result = height_scan(env, sensor_name)
    return torch.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)


def safe_base_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Safely fetch base linear velocity, replacing NaNs/infs with finite values."""
    # 安全封装：机身线速度，NaN→0、±Inf→±100
    from mjlab.envs.mdp.observations import base_lin_vel
    result = base_lin_vel(env)
    return torch.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)


def safe_foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Safely fetch foot air time, replacing NaNs with zeros."""
    # 安全封装：脚离地时间
    from mjlab.tasks.velocity.mdp.observations import foot_air_time
    result = foot_air_time(env, sensor_name)
    return torch.nan_to_num(result, nan=0.0)


def safe_foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Safely fetch foot contact flags, replacing NaNs with zeros."""
    # 安全封装：脚接触标志
    from mjlab.tasks.velocity.mdp.observations import foot_contact
    result = foot_contact(env, sensor_name)
    return torch.nan_to_num(result, nan=0.0)


def safe_foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Safely fetch foot contact forces, replacing NaNs with zeros."""
    # 安全封装：脚接触力
    from mjlab.tasks.velocity.mdp.observations import foot_contact_forces
    result = foot_contact_forces(env, sensor_name)
    return torch.nan_to_num(result, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 足式运动奖励
# ─────────────────────────────────────────────────────────────────────────────


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize deviation from the target foot clearance height.

    注意这里是"惩罚"却返回正值：配置里 weight 需要填负数，或按当前 env_cfgs
    的用法（该函数实际未挂载）自行决定。cost = |足端离地高度 − 目标| × 足端水平速度，
    即"足端在移动且高度不对"才罚，静止时 cost 也被 vel_norm 压成 0。
    """
    # 足端离地高度惩罚：足端偏离目标高度乘以水平速度，只在移动时生效。
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=(".*_foot_Link",))
    asset: Entity = env.scene[asset_cfg.name]
    
    foot_z = asset.data.body_com_pos_w[:, asset_cfg.body_ids, 2]
    foot_vel_xy = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)
    
    cmd = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(cmd[:, :2], dim=1)
    angular_norm = torch.abs(cmd[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    
    reward = cost * active
    return reward


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces on the first landing contact."""
    # 软着陆惩罚：仅惩罚"首次触地"瞬间的冲击力（力 × 首次接触标志）
    # 防止机器人从高处砸下来（降低落地冲击），且只在有运动指令时生效
    from mjlab.sensor import ContactSensor
    contact_sensor: ContactSensor = env.scene[sensor_name]
    forces = contact_sensor.data.force
    force_magnitude = torch.norm(forces, dim=-1)
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)
    
    cmd = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(cmd[:, :2], dim=1)
    angular_norm = torch.abs(cmd[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    
    reward = cost * active
    return reward


def _total_command_magnitude(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Retrieve absolute horizontal command speed."""
    # 工具函数：指令总强度 = |水平速度| + |偏航角速度|（多个奖励共用）
    command = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    return linear_norm + angular_norm


def adaptive_leg_motion_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    command_threshold: float = 0.05,
    tilt_relax_start: float = 0.08,
    tilt_relax_end: float = 0.30,
    contact_target: float = 0.85,
    min_penalty_scale: float = 0.2,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize leg motion less during tilt or low-contact recovery.

    relax 系数取 tilt_relax 与 contact_relax 的较大值（"只要有一个条件成立就放松"），
    然后映射成惩罚倍率 penalty_scale = 1 − 0.8·relax，范围 [0.2, 1.0]。
    也就是说最松的时候仍有 20% 的惩罚，保留"别乱蹬"的底线信号。
    """
    # 自适应腿部运动惩罚：身体倾斜大或足端接触少时自动放松。
    # 动机：越障/腾空瞬间必然需要剧烈摆腿，此时若照常重罚会把策略逼成"僵直腿"
    #   - tilt_relax:    倾斜越大（投影重力 xy 分量越大）越放松
    #   - contact_relax: 足端接触比例越低（腾空）越放松
    #   - penalty_scale: 1.0（全罚）~ min_penalty_scale（最低 0.2 倍罚）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=(
            ".*_hip_abduction_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ))
    asset: Entity = env.scene[asset_cfg.name]
    leg_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    base_cost = torch.mean(torch.square(leg_vel), dim=1)
    active = (_total_command_magnitude(env, command_name) > command_threshold).float()

    tilt_xy = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    # 倾斜因子：0.08→0 松弛度，0.30→1 完全松弛（线性插值后 clamp）
    tilt_relax = torch.clamp(
        (tilt_xy - tilt_relax_start) / max(tilt_relax_end - tilt_relax_start, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    contact_fraction = torch.mean(in_contact, dim=1)
    if contact_fraction.ndim > 1:
        contact_fraction = torch.mean(contact_fraction, dim=1)
    # 接触因子：接触比例越低于 contact_target，松弛度越大
    contact_relax = torch.clamp(
        (contact_target - contact_fraction) / max(contact_target, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    relax = torch.maximum(tilt_relax, contact_relax)
    penalty_scale = 1.0 - (1.0 - min_penalty_scale) * relax
    reward = base_cost * active * penalty_scale
    return reward


def contact_fraction_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Return the fraction of feet currently in ground contact."""
    # 返回持续足地接触比例，供足式任务按需配置。
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    reward = torch.mean(in_contact, dim=1)
    return reward


def stand_still(
    env,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize joint deviation from default posture when no command is active.

    与上面的 joint_pos_penalty 功能重叠（后者是带倍率的更细版本），
    Rough 环境先把基础模板里的 stand_still 移除，再以 weight=-2.0 重新挂上，
    本质是同一条规则调权重，保留是为了让权重可被单独观察。
    """
    # 静止惩罚：指令为 0（无运动指令）时，关节偏离默认姿态要罚
    # 防止"指令静止但腿乱动"的抖动，训练干净利落的待机姿态
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=(
            ".*_hip_abduction_joint", ".*_hip_pitch_joint", ".*_knee_joint",
        ))
    asset = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cost = torch.mean(torch.abs(diff), dim=1)
    command = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    inactive = (linear_norm + angular_norm < command_threshold).float()
    reward = cost * inactive
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 4. 关节位置 / 姿态惩罚
# ─────────────────────────────────────────────────────────────────────────────


def hip_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize hip abduction joints deviating from default."""
    # 惩罚髋外展关节偏离默认位（保持两腿对称、不外八字）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=(".*_hip_abduction_joint",))
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(q - q0), dim=1)
    return reward


def hip_abduction_excess(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.12,
    deviation_scale: float = 0.25,
) -> torch.Tensor:
    """Penalize hip abduction outside a small band around the default pose."""
    if tolerance < 0.0 or deviation_scale <= 0.0:
        raise ValueError("Hip tolerance must be nonnegative and deviation scale positive.")
    asset: Entity = env.scene[asset_cfg.name]
    deviation = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    excess = torch.clamp(deviation - tolerance, min=0.0)
    env.extras["log"]["Metrics/hip_abduction_deviation_mean"] = deviation.mean()
    env.extras["log"]["Metrics/hip_abduction_excess_mean"] = excess.mean()
    return torch.sum(torch.square(excess / deviation_scale), dim=1)


def joint_deviation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize all specified joints deviating from default."""
    # 惩罚指定关节偏离默认位（L2 平方误差），默认作用于腿部三关节
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=(".*_hip_abduction_joint", ".*_hip_pitch_joint", ".*_knee_joint"))
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(q - q0), dim=1)
    return reward


def knee_extension_penalty(
    env: ManagerBasedRlEnv,
    maximum_angle: float,
    transition_width: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize knees that extend past a configured safe angle.

    The cost is one-sided: normal knee flexion is unrestricted, while extension
    toward zero and positive angles is increasingly expensive.
    """
    if transition_width <= 0.0:
        raise ValueError("Knee extension transition width must be positive.")
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=(".*_knee_joint",))
    asset: Entity = env.scene[asset_cfg.name]
    knee_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    extension = torch.clamp(knee_angle - maximum_angle, min=0.0)
    return torch.sum(torch.square(extension / transition_width), dim=1)


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize non-flat base orientation (roll/pitch via projected gravity)."""
    # 惩罚机身不水平：用投影重力的 xy 分量平方和衡量倾角
    # 注意：会阻碍故意倾斜越障/匍匐，一般只用于平地阶段或低权重
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward


def pitch_excess_penalty(
    env: ManagerBasedRlEnv,
    tolerance_degrees: float = 5.0,
    deviation_scale_degrees: float = 10.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return a quadratic cost for forward or backward pitch outside a deadband.

    Parameters are expressed in degrees. Projected gravity measures orientation,
    so a constant lean still incurs a cost even when angular velocity is zero.
    """
    if not math.isfinite(tolerance_degrees) or not 0.0 <= tolerance_degrees < 90.0:
        raise ValueError("Pitch tolerance must be finite and in [0, 90) degrees.")
    if not math.isfinite(deviation_scale_degrees) or deviation_scale_degrees <= 0.0:
        raise ValueError("Pitch deviation scale must be finite and positive.")
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0].abs(), torch.linalg.vector_norm(gravity[:, 1:], dim=1))
    excess = torch.clamp(pitch - math.radians(tolerance_degrees), min=0.0)
    return torch.square(excess / math.radians(deviation_scale_degrees))


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize vertical (z) base linear velocity."""
    # 惩罚机身 z 向（上下）速度，抑制弹跳，保持足端稳定接地。
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_link_lin_vel_b[:, 2])
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using the go2w kernel."""
    # 惩罚机身滚转/俯仰角速度（x/y 轴），抑制摇晃，使用 go2w 核（直立加权）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 5. 姿态奖励（类实现，速度相关的容差）
# ─────────────────────────────────────────────────────────────────────────────


class variable_posture:
    """Reward posture tracking with speed-dependent tolerance.

    实现方式：每次 __call__ 先用三个互斥掩码把当前状态分到 standing/walking/running
    三档（running_mask 与 is_tilting 取 max，即"速度快"或"身体歪"都算 running），
    再把三个 std 张量按掩码线性混合成一个逐关节的 std，最后算指数奖励。
    因为是按掩码连续混合（不是硬切换），档位切换时奖励曲线不会跳变。
    """

    # 姿态奖励（类形式）：关节贴不贴默认位，但容差随速度档位变化
    # 三个档位：standing(静止) / walking(行走) / running(奔跑)
    # 速度越快、或身体越倾斜（越障/攀爬中），std 越大 = 容差越宽 = 越不苛求姿态
    # 这样策略在越障时敢大幅摆腿，平地上又能保持标准姿态

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        self.default_joint_pos = default_joint_pos

        # 按关节名解析三档 std（配置里是按关节名的 dict，这里解析成张量）
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"], list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(std_standing, device=env.device, dtype=torch.float32)

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"], list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"], list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing: dict,
        std_walking: dict,
        std_running: dict,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # 参数已由 __init__ 缓存，运行时不重复解析
        asset: Entity = env.scene[asset_cfg.name]
        total_speed = _total_command_magnitude(env, command_name)
        
        # 提取身体倾斜度 (重力在 XY 平面的投影)
        tilt_xy = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
        
        # 如果身体倾斜 > 0.15 (约 8.6 度)，说明正在爬墙、越障或已摔倒
        is_tilting = (tilt_xy > 0.15).float()
        is_running_cmd = (total_speed >= running_threshold).float()
        
        # 【核心越障修改】只要速度极快，或者正在倾斜攀爬，直接赋予最大幅度的动作容忍权 (running_mask)
        running_mask = torch.clamp(is_running_cmd + is_tilting, 0.0, 1.0)
        
        # 剩下的平稳且不倾斜的状态，再去根据低速指令判断是 walking 还是 standing
        not_running_mask = 1.0 - running_mask
        is_walking_cmd = (total_speed >= walking_threshold).float()
        
        walking_mask = not_running_mask * is_walking_cmd
        standing_mask = not_running_mask * (1.0 - is_walking_cmd)

        # 按档位掩码线性混合 std（每个关节的容差）
        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        reward = torch.exp(-torch.mean(error_squared / (std**2), dim=1))
        return reward


# ─────────────────────────────────────────────────────────────────────────────
# 6. 模式条件化奖励（依赖 mode_command.py 的 ModeCommand，防御性读取）
# ─────────────────────────────────────────────────────────────────────────────


def get_mode_id(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Retrieve current integer mode_id tensor from command_manager.

    Safely falls back to zeros (walk mode) if command manager or mode is not initialized.
    """
    # 从 CommandManager 读取当前模式（mode_command.py 的 ModeCommand.get_mode_id）
    # 未启用 mode 指令时防御性返回 0（walk），保证奖励函数不崩
    if not hasattr(env, "command_manager") or env.command_manager is None:
        return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    term = env.command_manager.get_term("mode")
    if term is None or not hasattr(term, "get_mode_id"):
        return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return term.get_mode_id()


def crouch_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.22,
    std: float = 0.05,
    mode_command_name: str = "mode",
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward the robot for keeping body low under target height when in crouch mode (mode_id == 2).

    Uses a single-sided penalty so that going lower than target_height receives full reward.
    """
    # 匍匐高度奖励：仅在 mode_id==2（crouch）时生效
    # 单边惩罚：root_z 高于 target_height 才算误差，越低奖励越满
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]

    if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
        env_origins_z = env.scene.env_origins[:, 2]
    else:
        env_origins_z = torch.zeros(env.num_envs, device=env.device)

    root_z = asset.data.root_link_pos_w[:, 2] - env_origins_z
    error = torch.square(torch.clamp(root_z - target_height, min=0.0))
    reward = torch.exp(-error / std ** 2)

    mode_id = get_mode_id(env)
    active = (mode_id == 2).float()
    reward = reward * active
    return reward


class mode_conditioned_posture:
    """Conditioned posture reward selecting targets/tolerances by active mode_id."""

    # 模式条件化姿态奖励（类形式）：
    #   根据当前 mode_id 选择不同的目标姿态与容差：
    #     walk   → 默认姿态，std_walk
    #     climb  → 默认姿态，std × climb_std_scale（宽容差，允许大幅摆腿越障）
    #     crouch → 若配置了 crouch_pos 则目标改为匍匐姿态，std × crouch_std_scale
    # 让一条策略同时学会三种姿态，由模式指令切换

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        asset = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        self.joint_ids = asset.find_joints(cfg.params["asset_cfg"].joint_names)[0]

        _, _, std_walk = resolve_matching_names_values(
            data=cfg.params["std_walk"], list_of_strings=joint_names,
        )
        self.std_walk = torch.tensor(std_walk, device=env.device, dtype=torch.float32)

        # 匍匐模式的目标关节位（可选，配置了才启用）
        if "crouch_pos" in cfg.params:
            _, _, crouch_pos_vals = resolve_matching_names_values(
                data=cfg.params["crouch_pos"], list_of_strings=joint_names,
            )
            self.crouch_pos_target = torch.tensor(
                crouch_pos_vals, device=env.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            self.crouch_pos_target = None

        # climb / crouch 模式放宽 std 的倍率（越障/匍匐时允许更大幅度的动作）
        climb_scale = cfg.params.get("climb_std_scale", 3.0)
        crouch_scale = cfg.params.get("crouch_std_scale", 4.0)
        self.std_climb = self.std_walk * climb_scale
        self.std_crouch = self.std_walk * crouch_scale

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_walk: dict,
        asset_cfg: SceneEntityCfg,
        climb_std_scale: float = 3.0,
        crouch_std_scale: float = 4.0,
        crouch_pos: dict | None = None,
    ) -> torch.Tensor:
        del std_walk, crouch_pos  # 已在 __init__ 缓存

        asset = env.scene[asset_cfg.name]
        mode_id = get_mode_id(env).float()

        # 构造三模式掩码（unsqueeze(1) 便于与关节维广播）
        walk_mask = (mode_id == 0).float().unsqueeze(1)
        climb_mask = (mode_id == 1).float().unsqueeze(1)
        crouch_mask = (mode_id == 2).float().unsqueeze(1)

        # 按模式混合容差
        std = (
            self.std_walk * walk_mask
            + self.std_climb * climb_mask
            + self.std_crouch * crouch_mask
        )

        current_pos = asset.data.joint_pos[:, self.joint_ids]
        base_desired_pos = self.default_joint_pos[:, self.joint_ids]

        # 匍匐模式下目标姿态替换为 crouch_pos_target
        if self.crouch_pos_target is not None:
            desired_pos = (
                base_desired_pos * (walk_mask + climb_mask)
                + self.crouch_pos_target * crouch_mask
            )
        else:
            desired_pos = base_desired_pos

        error_sq = torch.square(current_pos - desired_pos)
        reward = torch.exp(-torch.mean(error_sq / (std ** 2 + 1e-8), dim=1))
        return reward


def crawl_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.22,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg | None = None,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Reward low root clearance using the same ground as height termination.

    Missing or invalid ground returns zero reward. This is not a whole-body
    obstacle-clearance test; the lower bound is enforced by termination.
    """
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Crawl height reward std must be positive")
    height, valid = root_ground_clearance(env, asset_cfg, sensor_name)
    safe_height = torch.where(valid, height, torch.zeros_like(height))
    error = torch.square(torch.clamp(safe_height - target_height, min=0.0))
    return torch.where(valid, torch.exp(-error / std ** 2), torch.zeros_like(height))


def command_aligned_progress_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    minimum_command_speed: float = 0.1,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward signed velocity along the commanded horizontal direction."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_speed[:, None], min=1.0e-6)
    aligned_velocity = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * command_direction, dim=1)
    normalized_progress = aligned_velocity / torch.clamp(command_speed, min=minimum_command_speed)
    active = command_speed >= minimum_command_speed
    return torch.where(active, torch.clamp(normalized_progress, min=-1.0, max=1.0), 0.0)


def command_progress_deficit_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    minimum_command_speed: float = 0.2,
    minimum_progress_ratio: float = 0.5,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize commanded locomotion that makes insufficient forward progress.

    The signed progress reward is zero when the robot freezes.  This term
    turns that zero-progress state into a cost while leaving low-speed and
    standing commands unaffected.
    """
    if minimum_command_speed <= 0.0:
        raise ValueError("Minimum command speed must be positive.")
    if not 0.0 < minimum_progress_ratio <= 1.0:
        raise ValueError("Minimum progress ratio must lie in (0, 1].")
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_speed[:, None], min=1.0e-6)
    aligned_velocity = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * command_direction, dim=1)
    required_velocity = command_speed * minimum_progress_ratio
    deficit = torch.clamp(required_velocity - aligned_velocity, min=0.0)
    normalized_deficit = torch.clamp(
        deficit / torch.clamp(required_velocity, min=1.0e-6), max=1.0
    )
    active = command_speed >= minimum_command_speed
    return torch.where(active, torch.square(normalized_deficit), torch.zeros_like(command_speed))


def command_orthogonal_drift_penalty(
    env, command_name: str, minimum_command_speed: float = 0.06,
    speed_tolerance: float = 0.04, deviation_scale: float = 0.15,
) -> torch.Tensor:
    """Penalize horizontal motion perpendicular to a translation command."""
    if not all(math.isfinite(v) and v > 0.0 for v in (minimum_command_speed, deviation_scale)):
        raise ValueError("Translation command threshold and drift scale must be finite and positive.")
    if not math.isfinite(speed_tolerance) or speed_tolerance < 0.0:
        raise ValueError("Orthogonal drift tolerance must be finite and nonnegative.")
    command = env.command_manager.get_command(command_name)[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / speed[:, None].clamp_min(1.0e-6)
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    cross_speed = (direction[:, 0] * velocity[:, 1] - direction[:, 1] * velocity[:, 0]).abs()
    cost = ((cross_speed - speed_tolerance).clamp_min(0.0) / deviation_scale).square().clamp_max(4.0)
    return torch.where(speed >= minimum_command_speed, cost, 0.0)


def _translation_mask(env, command_name, minimum_command_speed, include_forward=False):
    if not math.isfinite(minimum_command_speed) or minimum_command_speed <= 0.0:
        raise ValueError("Translation command threshold must be finite and positive.")
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    nonforward = (command[:, 0] < 0.0) | (command[:, 1].abs() >= minimum_command_speed)
    return speed, (speed >= minimum_command_speed) & (nonforward | include_forward)


class TranslationStanceOverstayPenalty:
    """Limit prolonged stance, optionally including forward locomotion."""

    def __init__(self, cfg: RewardTermCfg, env):
        sensor = env.scene[cfg.params["sensor_name"]]
        self.contact_ids = []
        for body in FOOT_BODY_NAMES:
            ids = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            if len(ids) != 1:
                raise ValueError(f"Expected one contact channel for {body}.")
            self.contact_ids.append(ids[0])
        self.stance_age = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.stance_age[env_ids] = 0.0

    def __call__(
        self, env, sensor_name: str, command_name: str,
        minimum_command_speed: float = 0.06, stance_distance: float = 0.18,
        minimum_stance_limit: float = 0.30, maximum_stance_limit: float = 1.20,
        deviation_scale: float = 0.30, minimum_air_time: float = 0.08,
        include_forward: bool = False,
    ) -> torch.Tensor:
        values = (stance_distance, minimum_stance_limit, maximum_stance_limit, deviation_scale, minimum_air_time)
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("Stance timing parameters must be finite and positive.")
        if minimum_stance_limit > maximum_stance_limit:
            raise ValueError("Minimum stance limit must not exceed maximum stance limit.")
        speed, active = _translation_mask(env, command_name, minimum_command_speed, include_forward)
        sensor = env.scene[sensor_name]
        if sensor.data.current_air_time is None:
            raise RuntimeError("Translation stance penalty requires contact air-time tracking.")
        contact = sensor.data.found[:, self.contact_ids] > 0
        released = (~contact) & (sensor.data.current_air_time[:, self.contact_ids] >= minimum_air_time)
        age = self.stance_age + contact.float() * env.step_dt
        self.stance_age = torch.where(active[:, None] & ~released, age, 0.0)
        allowed_time = (stance_distance / speed.clamp_min(minimum_command_speed)).clamp(
            min=minimum_stance_limit, max=maximum_stance_limit
        )
        excess = (self.stance_age - allowed_time[:, None]).clamp_min(0.0)
        cost = (excess / deviation_scale).square().clamp_max(4.0)
        env.extras["log"]["Metrics/nonforward_stance_max_s"] = (
            self.stance_age.max(dim=1).values.sum() / active.sum().clamp_min(1)
        )
        return torch.where(contact & active[:, None], cost, 0.0).mean(dim=1)


def base_clearance_outside_range_l2(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    maximum_height: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize root clearance only when it leaves the locally measured safe band."""
    if not minimum_height < maximum_height:
        raise ValueError("Clearance range must have a positive width")
    clearance, valid = root_ground_clearance(env, asset_cfg, sensor_name)
    error = torch.clamp(minimum_height - clearance, min=0.0)
    error += torch.clamp(clearance - maximum_height, min=0.0)
    return torch.where(valid, torch.square(error), torch.ones_like(error))


def _in_place_turn_mask(env, command_name, minimum_yaw_degrees, maximum_linear_speed):
    if not math.isfinite(minimum_yaw_degrees) or minimum_yaw_degrees <= 0.0:
        raise ValueError("Turn command threshold must be finite and positive.")
    if not math.isfinite(maximum_linear_speed) or maximum_linear_speed < 0.0:
        raise ValueError("Turn linear command threshold must be finite and nonnegative.")
    command = env.command_manager.get_command(command_name)
    active = (
        (command[:, 2].abs() >= math.radians(minimum_yaw_degrees))
        & (torch.linalg.vector_norm(command[:, :2], dim=1) <= maximum_linear_speed)
    )
    return command, active


def turn_progress_deficit_penalty(
    env, command_name: str, minimum_yaw_degrees: float = 8.0,
    maximum_linear_speed: float = 0.08, minimum_progress_ratio: float = 0.6,
) -> torch.Tensor:
    """Penalize insufficient signed yaw velocity during commanded in-place turns."""
    if not math.isfinite(minimum_progress_ratio) or not 0.0 < minimum_progress_ratio <= 1.0:
        raise ValueError("Turn progress ratio must lie in (0, 1].")
    command, active = _in_place_turn_mask(
        env, command_name, minimum_yaw_degrees, maximum_linear_speed
    )
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2] * command[:, 2].sign()
    required = command[:, 2].abs() * minimum_progress_ratio
    deficit = torch.clamp(1.0 - actual / required.clamp_min(1.0e-6), min=0.0, max=2.0)
    return torch.where(active, deficit.square(), 0.0)


def turn_linear_drift_penalty(
    env, command_name: str, minimum_yaw_degrees: float = 8.0,
    maximum_linear_speed: float = 0.08, speed_tolerance: float = 0.05,
    deviation_scale: float = 0.15,
) -> torch.Tensor:
    """Limit horizontal velocity error without demanding motionless support feet."""
    if not math.isfinite(speed_tolerance) or speed_tolerance < 0.0:
        raise ValueError("Turn drift tolerance must be finite and nonnegative.")
    if not math.isfinite(deviation_scale) or deviation_scale <= 0.0:
        raise ValueError("Turn drift scale must be finite and positive.")
    command, active = _in_place_turn_mask(
        env, command_name, minimum_yaw_degrees, maximum_linear_speed
    )
    error = torch.linalg.vector_norm(
        env.scene["robot"].data.root_link_lin_vel_b[:, :2] - command[:, :2], dim=1
    )
    cost = ((error - speed_tolerance).clamp_min(0.0) / deviation_scale).square().clamp_max(4.0)
    return torch.where(active, cost, 0.0)


def turn_yaw_error_degrees(
    env, command_name: str, sign: int, minimum_yaw_degrees: float = 8.0,
    maximum_linear_speed: float = 0.08,
) -> torch.Tensor:
    """Broadcast the active left/right in-place turn error in degrees per second."""
    if sign not in (-1, 1):
        raise ValueError("Turn direction sign must be -1 or 1.")
    command, active = _in_place_turn_mask(
        env, command_name, minimum_yaw_degrees, maximum_linear_speed
    )
    active = active & (command[:, 2] * sign > 0.0)
    error = torch.rad2deg((env.scene["robot"].data.root_link_ang_vel_b[:, 2] - command[:, 2]).abs())
    mean = torch.where(active, _finite(error), 0.0).sum() / active.sum().clamp_min(1)
    return torch.ones_like(error) * mean


def terrain_type_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the active generated-terrain type index."""
    terrain = env.scene.terrain
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, device=env.device)
    return terrain.terrain_types.float()


def terrain_level_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the active generated-terrain difficulty level."""
    terrain = env.scene.terrain
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        return torch.zeros(env.num_envs, device=env.device)
    return terrain.terrain_levels.float()


def command_aligned_progress_metric(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return signed horizontal velocity along the current command."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1)
    direction = command / torch.clamp(speed[:, None], min=1.0e-6)
    progress = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
    return torch.where(speed > 0.0, progress, torch.zeros_like(progress))


def termination_reason_metric(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
    """Expose an individual termination cause as a numeric metric."""
    return env.termination_manager.get_term(term_name).float()


# ─────────────────────────────────────────────────────────────────────────────
# 7. 地形等级相关奖励（配合 curriculums.py 的地形课程使用）
# ─────────────────────────────────────────────────────────────────────────────


def _get_terrain_levels(env: ManagerBasedRlEnv) -> torch.Tensor | None:
    """Safely fetch current active terrain levels from environment."""
    # 工具函数：安全获取当前地形等级（无地形时返回 None）
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and hasattr(terrain, "terrain_levels"):
        return terrain.terrain_levels
    if hasattr(env, "terrain_levels"):
        return env.terrain_levels
    return None


def terrain_level_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Positive step bonus normalized to [0, 1] by max terrain level (num_rows-1 = 9).

    Normalizing prevents the raw level integer (0~9) from dominating the reward
    signal at high difficulty, which would otherwise incentivize level-rushing
    over quality locomotion.
    """
    # 地形等级正奖励：levels / 9 归一化到 [0,1]
    # 若不归一化，高难度时等级数值会主导奖励信号，诱导策略"冲级"而忽略步态质量
    levels = _get_terrain_levels(env)
    if levels is None:
        return torch.zeros(env.num_envs, device=env.device)
    return levels.float() / 9.0


def action_rate_curriculum_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Apply action change rate penalty with curriculum-driven scaling.

    Penalizes rate changes fully at level 0 (flat) for optimal smoothness,
    decaying to 10% penalty at levels 8+ to facilitate explosive dynamic clearing maneuvers.
    """
    # 课程化动作变化率惩罚：
    #   平地(level 0) 全量惩罚动作突变 → 训出平滑步态；
    #   高难度(level 8+) 惩罚衰减到 10% → 允许爆发式越障动作
    # 与 env_cfgs.py 里 action_rate 的区别：那个是固定权重，这个是动态衰减版，
    # 目前两个环境都把它移除了（cfg.rewards.pop("action_rate_curriculum")），属于备用项。
    action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    levels = _get_terrain_levels(env)
    if levels is None:
        reward = action_rate
    else:
        decay = 1.0 - 0.9 * torch.clamp(levels.float() / 8.0, 0.0, 1.0)
        reward = action_rate * decay
        
    return reward


def leg_symmetry(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asymmetry between left and right legs (pitch and knee joints) robustly.

    与后面 joint_mirror 的区别：
      - leg_symmetry 用硬编码的关节名前缀（fl_/fr_/rl_/rr_）字符串搜索，
        且对角配对（fl↔fr、rl↔rr）；找不到名字时 fallback 返回索引 0，
        这会静默地拿"第 0 个关节"去比较，属于隐患（Rough 环境已把它移除，改用 joint_mirror）。
      - joint_mirror 由配置传入关节对，且带索引缓存，更稳妥。
    """
    # 左右腿对称惩罚：左前-右前、左后-右后的 pitch/knee 关节位差平方和
    # 防止策略学出"一腿高、一腿低"的跛行步态（用关节名检索，找不到时兜底返回索引 0）
    asset: Entity = env.scene["robot"]
    joint_names = asset.data.joint_names
    
    def find_idx(side, jtype):
        for i, name in enumerate(joint_names):
            name_lower = name.lower()
            if side in name_lower and jtype in name_lower:
                return i
        # Fallback if not found to avoid crash, though it should find it
        return 0
        
    fl_p = find_idx("fl_", "pitch")
    fr_p = find_idx("fr_", "pitch")
    rl_p = find_idx("rl_", "pitch")
    rr_p = find_idx("rr_", "pitch")
    
    fl_k = find_idx("fl_", "knee")
    fr_k = find_idx("fr_", "knee")
    rl_k = find_idx("rl_", "knee")
    rr_k = find_idx("rr_", "knee")
    
    q = asset.data.joint_pos
    cost = torch.square(q[:, fl_p] - q[:, fr_p]) + \
           torch.square(q[:, rl_p] - q[:, rr_p]) + \
           torch.square(q[:, fl_k] - q[:, fr_k]) + \
           torch.square(q[:, rl_k] - q[:, rr_k])
    return cost


def feet_contact_without_cmd(
    env, command_name: str, sensor_name: str, command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward foot contact count only when no command is active.

    注意返回值是布尔张量乘法再乘权重浮点（reward *= bool_tensor 会做类型提升），
    最终是"接触足端数量 × 直立加权"，再乘权重 0.1，量级很小，
    作用是给静止状态一个"四个足端都贴地"的轻推，防止单足悬空的怪异站姿。
    """
    # 无指令时的足端接触奖励：指令静默时鼓励更多足端接触以稳定姿态。
    from mjlab.sensor import ContactSensor
    contact_sensor = env.scene[sensor_name]
    contact = contact_sensor.data.found > 0
    reward = torch.sum(contact, dim=-1).float()
    cmd = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(cmd[:, :2], dim=1)
    angular_norm = torch.abs(cmd[:, 2])
    reward *= (linear_norm + angular_norm) < command_threshold
    asset = env.scene["robot"]
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def support_deficit_penalty(env, sensor_name: str, min_support_feet: int = 2) -> torch.Tensor:
    """Penalize insufficient ground support, including the all-feet-airborne state."""
    support_count = (env.scene[sensor_name].data.found > 0).sum(dim=1)
    env.extras["log"]["Metrics/support_feet_mean"] = support_count.float().mean()
    env.extras["log"]["Metrics/flight_fraction"] = (support_count == 0).float().mean()
    return torch.clamp(min_support_feet - support_count, min=0).float()


def supported_feet_air_time(
    env, sensor_name: str, threshold_min: float = 0.1, threshold_max: float = 0.7,
    command_name: str = "twist", command_threshold: float = 0.1,
    min_support_feet: int = 2,
) -> torch.Tensor:
    """Reward swing time only when enough other feet remain on the ground."""
    from mjlab.tasks.velocity.mdp.rewards import feet_air_time

    reward = feet_air_time(
        env, sensor_name=sensor_name, threshold_min=threshold_min,
        threshold_max=threshold_max, command_name=command_name,
        command_threshold=command_threshold,
    )
    support_count = (env.scene[sensor_name].data.found > 0).sum(dim=1)
    return reward * (support_count >= min_support_feet)


class FootSwingHeightReward:
    """Reward target-height swings after sustained stance, with a cycle-rate budget."""

    def __init__(self, cfg: RewardTermCfg, env):
        sensor = env.scene[cfg.params["sensor_name"]]
        robot = env.scene["robot"]
        self.site_ids = []
        for name in sensor.primary_names:
            matches = [i for i, body in enumerate(FOOT_BODY_NAMES) if name.endswith(body)]
            if len(matches) != 1:
                raise ValueError(f"Unexpected foot contact body: {name}")
            ids, _ = robot.find_sites(FOOT_SITE_NAMES[matches[0]])
            if len(ids) != 1:
                raise ValueError("Each foot must have exactly one sole site.")
            self.site_ids.append(ids[0])
        if len(self.site_ids) != 4 or len(set(self.site_ids)) != 4:
            raise ValueError("Swing height reward requires four distinct feet.")
        self.peak_height = torch.zeros((env.num_envs, 4), device=env.device)
        self.has_swing = torch.zeros_like(self.peak_height, dtype=torch.bool)
        self.supported_swing = torch.ones_like(self.has_swing)
        self.pending_score = torch.zeros_like(self.peak_height)
        self.pending_valid = torch.zeros_like(self.has_swing)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.peak_height[env_ids] = 0.0
        self.has_swing[env_ids] = False
        self.supported_swing[env_ids] = True
        self.pending_score[env_ids] = 0.0
        self.pending_valid[env_ids] = False

    def __call__(
        self, env, sensor_name: str, command_name: str,
        target_height: float = 0.10, height_std: float = 0.05,
        command_threshold: float = 0.1,
        ground_sensor_name: str = "base_height_sensor",
        min_support_feet: int = 2,
        min_stance_time: float = 0.16,
        min_cycle_time: float = 0.50,
    ) -> torch.Tensor:
        if min(target_height, height_std, min_stance_time, min_cycle_time) <= 0.0:
            raise ValueError("Swing height, tolerance and gait times must be positive.")
        sensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time
        last_air_time = sensor.data.last_air_time
        if contact_time is None or last_air_time is None:
            raise RuntimeError("Swing height reward requires contact and air time tracking.")
        robot = env.scene["robot"]
        clearance, valid = root_ground_clearance(env, sensor_name=ground_sensor_name)
        ground_z = robot.data.root_link_pos_w[:, 2] - clearance
        foot_height = torch.clamp(
            robot.data.site_pos_w[:, self.site_ids, 2] - ground_z[:, None], min=0.0
        )
        command = env.command_manager.get_command(command_name)
        active = (torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs()) > command_threshold
        eligible = (active & valid)[:, None]
        in_air = sensor.data.found == 0
        self.peak_height = torch.where(
            eligible, torch.where(in_air, torch.maximum(self.peak_height, foot_height), self.peak_height), 0.0
        )
        self.has_swing = (self.has_swing | in_air) & eligible
        # 摆腿期间一旦支撑足不足，本次落地不再获得高度奖励。
        enough_support = (~in_air).sum(dim=1) >= min_support_feet
        self.supported_swing &= ~self.has_swing | enough_support[:, None]
        landed = sensor.compute_first_contact(dt=env.step_dt) & self.has_swing
        baseline = math.exp(-(target_height / height_std) ** 2)
        score = torch.clamp(
            (torch.exp(-((self.peak_height - target_height) / height_std) ** 2) - baseline)
            / (1.0 - baseline), min=0.0, max=1.0
        )
        upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
        # 抬脚得分先暂存，重新离地、指令静止或支撑不足时取消本次结算。
        pending_allowed = ~in_air & eligible & enough_support[:, None]
        self.pending_valid &= pending_allowed
        self.pending_score = torch.where(pending_allowed, self.pending_score, 0.0)
        self.pending_score = torch.where(landed, score, self.pending_score)
        self.pending_valid |= landed & self.supported_swing & pending_allowed
        matured = self.pending_valid & (contact_time >= min_stance_time)
        # 短周期按实际周期时长折算，限制通过频繁点地提高单位时间收益。
        cycle_scale = torch.clamp((last_air_time + contact_time) / min_cycle_time, max=1.0)
        reward = (self.pending_score * cycle_scale * matured).sum(dim=1) * upright / env.step_dt
        self.pending_valid &= ~matured
        env.extras["log"]["Metrics/swing_peak_height"] = (
            (self.peak_height * landed).sum() / landed.sum().clamp(min=1)
        )
        self.peak_height = torch.where(in_air, self.peak_height, 0.0)
        self.has_swing &= in_air
        self.supported_swing = torch.where(
            in_air & eligible, self.supported_swing, True
        )
        return reward


def short_stance_penalty(
    env, sensor_name: str, command_name: str,
    min_stance_time: float = 0.16, command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize takeoff after a brief touch instead of sustained stance."""
    if min_stance_time <= 0.0:
        raise ValueError("Minimum stance time must be positive.")
    sensor = env.scene[sensor_name]
    duration = sensor.data.last_contact_time
    if duration is None:
        raise RuntimeError("Short stance penalty requires contact time tracking.")
    detached = sensor.compute_first_air(dt=env.step_dt) & (duration > 0.0)
    command = env.command_manager.get_command(command_name)
    active = (torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs()) > command_threshold
    deficit = torch.clamp(1.0 - duration / min_stance_time, min=0.0)
    return (deficit.square() * detached).sum(dim=1) * active / env.step_dt


class StanceBalancePenalty:
    """Compare left/right contact duty over time during straight-line motion."""

    def __init__(self, cfg: RewardTermCfg, env):
        sensor = env.scene[cfg.params["sensor_name"]]
        self.foot_indices = []
        for body in FOOT_BODY_NAMES:
            ids = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            if len(ids) != 1:
                raise ValueError(f"Expected one contact channel for {body}.")
            self.foot_indices.append(ids[0])
        self.duty = torch.full((env.num_envs, 4), 0.5, device=env.device)
        self.age = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.duty[env_ids] = 0.5
        self.age[env_ids] = 0.0

    def __call__(
        self, env, sensor_name: str, command_name: str,
        smoothing_time: float = 1.0, warmup_time: float = 1.0,
        duty_tolerance: float = 0.10, deviation_scale: float = 0.25,
        command_threshold: float = 0.1, lateral_threshold: float = 0.1,
        yaw_threshold: float = 0.15,
    ) -> torch.Tensor:
        if min(smoothing_time, deviation_scale) <= 0.0 or min(warmup_time, duty_tolerance) < 0.0:
            raise ValueError("Invalid stance balance averaging or tolerance settings.")
        command = env.command_manager.get_command(command_name)
        straight = (
            (command[:, 0].abs() > command_threshold)
            & (command[:, 1].abs() < lateral_threshold)
            & (command[:, 2].abs() < yaw_threshold)
        )
        contact = (env.scene[sensor_name].data.found[:, self.foot_indices] > 0).float()
        alpha = 1.0 - math.exp(-env.step_dt / smoothing_time)
        self.duty = torch.where(straight[:, None], self.duty + alpha * (contact - self.duty), 0.5)
        self.age = torch.where(straight, self.age + env.step_dt, 0.0)
        valid = straight & (self.age >= warmup_time)
        differences = torch.abs(self.duty[:, [0, 2]] - self.duty[:, [1, 3]])
        excess = torch.clamp(differences - duty_tolerance, min=0.0)
        for i, leg in enumerate(("fl", "fr", "rl", "rr")):
            env.extras["log"][f"Metrics/stance_duty_{leg}"] = (
                (self.duty[:, i] * valid).sum() / valid.sum().clamp(min=1)
            )
        return (excess / deviation_scale).square().mean(dim=1) * valid


class FootLateralPlacementPenalty:
    """Keep each left/right foot-pair placement centered during straight motion."""

    def __init__(self, cfg: RewardTermCfg, env):
        robot = env.scene["robot"]
        sensor = env.scene[cfg.params["sensor_name"]]
        self.site_ids = []
        self.contact_ids = []
        for body, site in zip(FOOT_BODY_NAMES, FOOT_SITE_NAMES):
            ids, _ = robot.find_sites(site)
            contacts = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            if len(ids) != 1 or len(contacts) != 1:
                raise ValueError(f"Expected one foot site and contact channel for {body}.")
            self.site_ids.append(ids[0])
            self.contact_ids.append(contacts[0])
        self.mean_lateral = torch.zeros((env.num_envs, 4), device=env.device)
        self.seen = torch.zeros_like(self.mean_lateral, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.mean_lateral[env_ids] = 0.0
        self.seen[env_ids] = False

    def __call__(
        self, env, sensor_name: str, command_name: str,
        center_tolerance: float = 0.02, deviation_scale: float = 0.05,
        smoothing_time: float = 0.5, min_contact_time: float = 0.08,
        command_threshold: float = 0.1, lateral_threshold: float = 0.1,
        yaw_threshold: float = 0.15,
    ) -> torch.Tensor:
        if min(deviation_scale, smoothing_time) <= 0.0 or min(center_tolerance, min_contact_time) < 0.0:
            raise ValueError("Invalid lateral placement tolerance or averaging settings.")
        robot = env.scene["robot"]
        sensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time
        if contact_time is None:
            raise RuntimeError("Lateral placement penalty requires contact time tracking.")
        relative = robot.data.site_pos_w[:, self.site_ids] - robot.data.root_link_pos_w[:, None, :]
        quaternions = robot.data.root_link_quat_w[:, None, :].expand(-1, 4, -1)
        foot_y = quat_apply_inverse(
            quaternions.reshape(-1, 4), relative.reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)[:, :, 1]
        command = env.command_manager.get_command(command_name)
        straight = (
            (command[:, 0].abs() > command_threshold)
            & (command[:, 1].abs() < lateral_threshold)
            & (command[:, 2].abs() < yaw_threshold)
        )
        stable = (
            (sensor.data.found[:, self.contact_ids] > 0)
            & (contact_time[:, self.contact_ids] >= min_contact_time)
            & straight[:, None]
        )
        alpha = 1.0 - math.exp(-env.step_dt / smoothing_time)
        updated = torch.where(
            self.seen, self.mean_lateral + alpha * (foot_y - self.mean_lateral), foot_y
        )
        self.mean_lateral = torch.where(stable, updated, self.mean_lateral)
        self.mean_lateral = torch.where(straight[:, None], self.mean_lateral, 0.0)
        self.seen = (self.seen | stable) & straight[:, None]
        # 前后足对分别计算中心，避免前腿左偏与后腿右偏互相抵消。
        centers = self.mean_lateral.reshape(env.num_envs, 2, 2).mean(dim=-1)
        valid = self.seen.reshape(env.num_envs, 2, 2).all(dim=-1)
        excess = torch.clamp(centers.abs() - center_tolerance, min=0.0)
        for i, pair in enumerate(("front", "rear")):
            env.extras["log"][f"Metrics/{pair}_foot_lateral_center_abs"] = (
                (centers[:, i].abs() * valid[:, i]).sum() / valid[:, i].sum().clamp(min=1)
            )
        return ((excess / deviation_scale).square() * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


class ForwardFootTouchdownPenalty:
    """Penalize forward touchdown positions outside the nominal reachable footprint."""

    def __init__(self, cfg: RewardTermCfg, env):
        robot = env.scene["robot"]
        sensor = env.scene[cfg.params["sensor_name"]]
        self.site_ids = []
        self.contact_ids = []
        for body, site in zip(FOOT_BODY_NAMES, FOOT_SITE_NAMES):
            site_ids, _ = robot.find_sites(site)
            contact_ids = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            if len(site_ids) != 1 or len(contact_ids) != 1:
                raise ValueError(f"Expected one site and contact channel for {body}.")
            self.site_ids.append(site_ids[0])
            self.contact_ids.append(contact_ids[0])

    def _forward_excess(self, env, nominal_foot_x, forward_margin, deviation_scale):
        if len(nominal_foot_x) != 4:
            raise ValueError("Forward foot placement requires four nominal foot positions.")
        if not all(math.isfinite(value) for value in nominal_foot_x):
            raise ValueError("Nominal foot positions must be finite.")
        if not math.isfinite(forward_margin) or forward_margin <= 0.0:
            raise ValueError("Forward margin must be finite and positive.")
        if not math.isfinite(deviation_scale) or deviation_scale <= 0.0:
            raise ValueError("Forward deviation scale must be finite and positive.")
        robot = env.scene["robot"]
        relative = robot.data.site_pos_w[:, self.site_ids] - robot.data.root_link_pos_w[:, None, :]
        quaternions = robot.data.root_link_quat_w[:, None, :].expand(-1, 4, -1)
        foot_x = quat_apply_inverse(
            quaternions.reshape(-1, 4), relative.reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)[:, :, 0]
        maximum_x = foot_x.new_tensor(nominal_foot_x)[None, :] + forward_margin
        return torch.clamp(foot_x - maximum_x, min=0.0)

    def __call__(
        self,
        env,
        sensor_name: str,
        command_name: str,
        nominal_foot_x: tuple[float, float, float, float],
        forward_margin: float = 0.20,
        deviation_scale: float = 0.08,
        command_threshold: float = 0.15,
        lateral_threshold: float = 0.10,
        yaw_threshold: float = 0.15,
    ) -> torch.Tensor:
        if command_threshold <= 0.0:
            raise ValueError("Forward command threshold must be positive.")
        excess = self._forward_excess(env, nominal_foot_x, forward_margin, deviation_scale)
        sensor = env.scene[sensor_name]
        landed = sensor.compute_first_contact(dt=env.step_dt)[:, self.contact_ids]
        command = env.command_manager.get_command(command_name)
        straight_forward = (
            (command[:, 0] > command_threshold)
            & (command[:, 1].abs() < lateral_threshold)
            & (command[:, 2].abs() < yaw_threshold)
        )
        cost = torch.sum(torch.square(excess / deviation_scale) * landed.float(), dim=1)
        return cost * straight_forward.float() / env.step_dt


class TurnStanceTwistPenalty:
    """Limit accumulated body yaw while each foot remains in the same stance."""

    def __init__(self, cfg: RewardTermCfg, env):
        sensor = env.scene[cfg.params["sensor_name"]]
        self.contact_ids = []
        for body in FOOT_BODY_NAMES:
            ids = [i for i, name in enumerate(sensor.primary_names) if name.endswith(body)]
            if len(ids) != 1:
                raise ValueError(f"Expected one contact channel for {body}.")
            self.contact_ids.append(ids[0])
        self.twist = torch.zeros((env.num_envs, 4), device=env.device)
        self.previous_yaw = torch.zeros(env.num_envs, device=env.device)
        self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.twist[env_ids] = 0.0
        self.previous_yaw[env_ids] = 0.0
        self.previous_active[env_ids] = False

    def __call__(
        self, env, sensor_name: str, command_name: str,
        minimum_yaw_degrees: float = 8.0, maximum_linear_speed: float = 0.08,
        tolerance_degrees: float = 12.0, deviation_scale_degrees: float = 10.0,
        minimum_air_time: float = 0.08,
    ) -> torch.Tensor:
        if not math.isfinite(tolerance_degrees) or tolerance_degrees < 0.0:
            raise ValueError("Stance twist tolerance must be finite and nonnegative.")
        if not math.isfinite(deviation_scale_degrees) or deviation_scale_degrees <= 0.0:
            raise ValueError("Stance twist scale must be finite and positive.")
        if not math.isfinite(minimum_air_time) or minimum_air_time <= 0.0:
            raise ValueError("Stance release time must be finite and positive.")
        _, active = _in_place_turn_mask(env, command_name, minimum_yaw_degrees, maximum_linear_speed)
        sensor = env.scene[sensor_name]
        if sensor.data.current_air_time is None:
            raise RuntimeError("Stance twist penalty requires contact air-time tracking.")
        contact = sensor.data.found[:, self.contact_ids] > 0
        released = (~contact) & (sensor.data.current_air_time[:, self.contact_ids] >= minimum_air_time)
        w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        delta = yaw - self.previous_yaw
        delta = torch.atan2(delta.sin(), delta.cos())
        continuous = active & self.previous_active
        accumulated = self.twist + torch.where(continuous, delta, 0.0)[:, None]
        self.twist = torch.where(continuous[:, None] & ~released, accumulated, 0.0)
        self.previous_yaw = yaw.detach().clone()
        self.previous_active = active.clone()
        excess = (self.twist.abs() - math.radians(tolerance_degrees)).clamp_min(0.0)
        cost = (excess / math.radians(deviation_scale_degrees)).square().clamp_max(4.0)
        maximum_twist = torch.rad2deg(self.twist.abs()).max(dim=1).values
        env.extras["log"]["Metrics/turn_stance_twist_max_deg"] = (
            torch.where(active, maximum_twist, 0.0).sum() / active.sum().clamp_min(1)
        )
        return torch.where(contact & active[:, None], cost, 0.0).mean(dim=1)


class TurnFootWorkspacePenalty(ForwardFootTouchdownPenalty):
    """Keep all turning feet inside a soft hip-relative horizontal workspace."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self.hip_ids = []
        for body in FOOT_BODY_NAMES:
            name = f"{body.split('_')[0]}_hip_pitch_Link"
            ids, _ = robot.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected one hip body for {body}.")
            self.hip_ids.append(ids[0])

    def __call__(
        self, env, sensor_name: str, command_name: str,
        minimum_yaw_degrees: float = 8.0, maximum_linear_speed: float = 0.08,
        nominal_forward: float = 0.035, nominal_outward: float = 0.105,
        forward_margin: float = 0.24, lateral_margin: float = 0.12,
        deviation_scale: float = 0.08,
    ) -> torch.Tensor:
        _, active = _in_place_turn_mask(env, command_name, minimum_yaw_degrees, maximum_linear_speed)
        cost = self._workspace_cost(
            env, nominal_forward, nominal_outward, forward_margin, lateral_margin, deviation_scale
        )
        return torch.where(active, cost, 0.0)

    def _workspace_cost(
        self, env, nominal_forward, nominal_outward, forward_margin, lateral_margin, deviation_scale,
    ):
        if not all(math.isfinite(v) and v > 0.0 for v in (forward_margin, lateral_margin, deviation_scale)):
            raise ValueError("Foot workspace margins and scale must be finite and positive.")
        if not all(math.isfinite(v) for v in (nominal_forward, nominal_outward)):
            raise ValueError("Nominal hip-relative foot position must be finite.")
        robot = env.scene["robot"]
        relative = robot.data.site_pos_w[:, self.site_ids] - robot.data.body_link_pos_w[:, self.hip_ids]
        quaternions = robot.data.root_link_quat_w[:, None, :].expand(-1, 4, -1)
        local = quat_apply_inverse(quaternions.reshape(-1, 4), relative.reshape(-1, 3))
        local = local.reshape(env.num_envs, 4, 3)
        side = local.new_tensor([1.0, -1.0, 1.0, -1.0])
        forward_excess = ((local[:, :, 0] - nominal_forward).abs() - forward_margin).clamp_min(0.0)
        lateral_excess = ((local[:, :, 1] * side - nominal_outward).abs() - lateral_margin).clamp_min(0.0)
        cost = (forward_excess.square() + lateral_excess.square()) / deviation_scale ** 2
        return cost.clamp_max(4.0).mean(dim=1)


class TranslationFootWorkspacePenalty(TurnFootWorkspacePenalty):
    """Limit stance and swing overreach during commanded translation."""

    def __call__(
        self, env, sensor_name: str, command_name: str, minimum_command_speed: float = 0.06,
        nominal_forward: float = 0.035, nominal_outward: float = 0.105,
        forward_margin: float = 0.24, lateral_margin: float = 0.12, deviation_scale: float = 0.08,
        include_forward: bool = False,
    ) -> torch.Tensor:
        _, active = _translation_mask(env, command_name, minimum_command_speed, include_forward)
        cost = self._workspace_cost(
            env, nominal_forward, nominal_outward, forward_margin, lateral_margin, deviation_scale
        )
        return torch.where(active, cost, 0.0)


class ForwardFootSwingPenalty(ForwardFootTouchdownPenalty):
    """Penalize airborne foot overreach on every step of commanded locomotion."""

    def __call__(
        self,
        env,
        sensor_name: str,
        command_name: str,
        nominal_foot_x: tuple[float, float, float, float],
        forward_margin: float = 0.20,
        deviation_scale: float = 0.08,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        if not math.isfinite(command_threshold) or command_threshold <= 0.0:
            raise ValueError("Swing command threshold must be finite and positive.")
        excess = self._forward_excess(env, nominal_foot_x, forward_margin, deviation_scale)
        in_air = env.scene[sensor_name].data.found[:, self.contact_ids] == 0
        command = env.command_manager.get_command(command_name)
        active = (torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()) > command_threshold
        cost = torch.sum(torch.square(excess / deviation_scale) * in_air, dim=1)
        return cost * active


def long_foot_air_time_penalty(
    env,
    sensor_name: str,
    command_name: str,
    max_air_time: float = 0.55,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize an individual foot that remains airborne for too long while moving."""
    from mjlab.sensor import ContactSensor

    contact_sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = contact_sensor.data.current_air_time
    if current_air_time is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' must enable track_air_time.")
    excess_air_time = torch.relu(current_air_time - max_air_time)
    penalty = torch.mean(excess_air_time, dim=1)
    command = env.command_manager.get_command(command_name)
    command_magnitude = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return penalty * (command_magnitude > command_threshold).float()


def joint_pos_penalty(
    env,
    command_name: str,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize joint deviation from default; scale up when standing still.

    stand_still_scale 是该函数最关键的参数（Rough 里配 5.0）：
    无指令且机身几乎不动时，偏离默认姿态的惩罚放大 5 倍。
    这条规则把"站着就站标准"的信号做得非常强，代价是机器人静止时腿会绷得比较硬。
    """
    # 关节偏离惩罚（带档位）：有指令/在动 → 按原值罚；无指令且静止 → 放大 stand_still_scale 倍罚
    # 作用：静止时更严格地要求姿态归位，运动中允许一定偏差
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd_val = env.command_manager.get_command(command_name)
    cmd = torch.linalg.norm(cmd_val[:, :2], dim=1) + torch.abs(cmd_val[:, 2])
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def joint_mirror(
    env,
    mirror_joints: list[list[str]],
    asset_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    mirror_sign: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position differences between mirrored joint pairs.

    每个关节对计算左角减去带镜像符号的右角，再平方求平均。
    mirror_sign 为负一时约束左右角度异号，为正一时约束左右角度同号。
    注意是对"关节组"求和（joint_pair[0] 可能是正则匹配出的多个关节），
    所以配 ["fl_(hip_pitch|knee)_joint", "rr_(hip_pitch|knee)_joint"] 时，
    实际约束的是"左前腿与右后腿这对对角腿的构型是否互为镜像"（对角步态对称性）。
    """
    # 镜像关节对称惩罚：对 (左, 右) 关节对逐对求位差平方（结果按对数取均值）
    # 结果缓存 joint 索引对，避免每步重复字符串匹配
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cache = getattr(env, "joint_mirror_joints_cache", None)
    if not isinstance(cache, dict):
        cache = env.joint_mirror_joints_cache = {}
    key = (asset_cfg.name, tuple(tuple(pair) for pair in mirror_joints))
    if key not in cache:
        cache[key] = []
        for joint_pair in mirror_joints:
            p0 = asset.find_joints(joint_pair[0])[0]
            p1 = asset.find_joints(joint_pair[1])[0]
            cache[key].append([p0, p1])
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in cache[key]:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0]] - mirror_sign * asset.data.joint_pos[:, joint_pair[1]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        command_magnitude = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward *= (command_magnitude < command_threshold).float()
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 8. 接触 / 功率 / 姿态惩罚
# ─────────────────────────────────────────────────────────────────────────────


def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize unwanted leg-link contacts above a force threshold.

    传感器 body_collision 监控的对象是腿杆（hip_abduction/hip_pitch/knee 连杆），
    所以这条惩罚的实际含义是"腿杆撞到地形"，正常足端触地不计入。
    优先用 force_history 取历史峰值而不是当前帧：接触是脉冲式的，
    只看当前帧容易漏掉撞击瞬间。
    """
    # 非预期接触惩罚：底盘等不该碰地的部位若受力超阈值则罚（摔倒、趴地检测）
    # 优先用 force_history（历史峰值），没有则用当前 force
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        is_contact = torch.max(force_mag, dim=2)[0] > threshold
    else:
        force_mag = torch.norm(data.force, dim=-1)
        is_contact = force_mag > threshold
    reward = torch.sum(is_contact, dim=1).float()
    asset = env.scene["robot"]
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def contact_forces(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize foot contact forces above threshold.

    返回的是"超出阈值部分的力之和"，而不是总力——
    这意味着正常踩地的力（<100N）不产生惩罚，只有砸地/撞击才罚，
    等价于一个带死区的软约束，比直接罚 ΣF 温和得多（否则策略会不敢用力支撑）。
    """
    # 接触力惩罚：接触力超过阈值部分求和（防硬着陆、防电机过载）
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        peak_force = torch.max(force_mag, dim=2)[0]
    else:
        peak_force = torch.norm(data.force, dim=-1)
    reward = torch.sum(torch.clamp(peak_force - threshold, min=0.0), dim=1)
    asset = env.scene["robot"]
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def upward(env, asset_cfg=None):
    """Return (1 - g_z)^2: upright is 4, horizontal is 1, inverted is 0.

    A positive weight rewards upright orientation. Both roll and pitch affect
    this value; it is not an orientation-error penalty.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg('robot')
    asset = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_power(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalty for total joint mechanical power: sum(|tau * dq|).

    qfrc_actuator 是执行器施加的广义力（关节力矩），乘以关节角速度即瞬时机械功率。
    取绝对值再求和，表示"不管正功负功都算消耗"，对应电机的实际发热/耗电，
    比 joint_torques_l2（只看力矩大小）更贴近真实能耗约束。
    """
    # 关节机械功率惩罚：Σ|力矩 × 关节速度|，约束能耗与电机发热
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.qfrc_actuator * asset.data.joint_vel), dim=1)


def upright_roll_only(env, asset_cfg=None):
    """Penalize roll-axis deviation from upright.

    pitch 与 roll 的区别：机身投影重力的 x 分量对应俯仰（前后倾），y 分量对应横滚（左右倾）。
    足式机器人过障碍时必须允许前后大幅俯仰，但不能左右翻滚，所以只罚 y 分量。
    """
    # 仅惩罚横滚（roll）方向倾斜：投影重力 y 分量平方
    # 区别于 flat_orientation_l2：不惩罚俯仰（pitch），允许前后倾（适合越障）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg('robot')
    asset = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.projected_gravity_b[:, 1])
    return reward


def track_linear_velocity_l1(env, std, command_name):
    """Reward xy velocity tracking with L1 (absolute) error.

    与 L2 版的差别：L2 对"偶尔偏得特别多"的样本惩罚极重（误差平方），
    L1 是线性惩罚，对离群样本更宽容，训练早期梯度更平稳。
    相应地 exp(-err/std) 的分母是 std 的一次方，同一个 std 数值下容差含义不同。
    """
    # L1 版本的速度跟踪：用绝对误差而非平方误差，对离群误差更宽容
    asset = env.scene['robot']
    cmd = env.command_manager.get_command(command_name)
    lin_vel_error = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2] - cmd[:, :2], dim=1)
    reward = torch.exp(-lin_vel_error / std)
    return reward


def pitch_control_penalty(env, max_pitch_rad: float = 0.50, asset_cfg=None) -> torch.Tensor:
    """Penalize excessive pitch orientation beyond a safe threshold.
    
    Allows pitch angles up to max_pitch_rad (e.g. 29 degrees) for climbing obstacles,
    but quadratically penalizes any pitch angle exceeding this threshold (e.g. during flinging).
    """
    # 俯仰超限惩罚：允许俯仰到 max_pitch_rad（约 29°）用于爬障碍，
    # 超出该阈值（例如甩飞、后空翻）后按超出的平方惩罚
    import math
    if asset_cfg is None:
        from mjlab.envs.manager_based_rl_env import SceneEntityCfg
        asset_cfg = SceneEntityCfg('robot')
    asset = env.scene[asset_cfg.name]
    g_x = asset.data.projected_gravity_b[:, 0]
    g_x_threshold = math.sin(max_pitch_rad)
    excessive_pitch = torch.clamp(torch.abs(g_x) - g_x_threshold, min=0.0)
    reward = torch.square(excessive_pitch)
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 9. 诊断指标（Metrics，喂给 TensorBoard 观察训练质量，不参与梯度）
# ─────────────────────────────────────────────────────────────────────────────


def tracking_lin_vel_error(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Current-step xy velocity tracking error."""
    # 指标：当前 xy 速度跟踪误差（L2 范数）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _finite(torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2] - cmd[:, :2], dim=1))


def tracking_yaw_vel_error(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Current-step yaw velocity tracking error."""
    # 指标：当前偏航角速度跟踪误差
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _finite(torch.abs(asset.data.root_link_ang_vel_b[:, 2] - cmd[:, 2]))


def tracking_lin_vel_x_error(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Current-step x velocity tracking error."""
    # 指标：x 轴速度跟踪误差
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _finite(torch.abs(asset.data.root_link_lin_vel_b[:, 0] - cmd[:, 0]))


def tracking_lin_vel_y_error(env, command_name: str = "twist", asset_cfg=None) -> torch.Tensor:
    """Current-step y velocity tracking error."""
    # 指标：y 轴速度跟踪误差
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return _finite(torch.abs(asset.data.root_link_lin_vel_b[:, 1] - cmd[:, 1]))


def tracking_lin_vel_along_command_error(
    env,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Error of velocity projected onto the commanded xy direction."""
    # 指标：速度在指令方向上的投影误差（衡量"跑对方向"而非"跑对大小"）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = cmd[:, :2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    direction = cmd_xy / torch.clamp(cmd_speed.unsqueeze(1), min=1.0e-6)
    actual_along = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
    return _finite(torch.abs(actual_along - cmd_speed))


def actual_lin_vel_orthogonal_command_mean(
    env,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Absolute velocity component perpendicular to the commanded xy direction."""
    # 指标：垂直于指令方向的速度分量（衡量侧滑程度）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = cmd[:, :2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    direction = cmd_xy / torch.clamp(cmd_speed.unsqueeze(1), min=1.0e-6)
    actual = asset.data.root_link_lin_vel_b[:, :2]
    actual_along = torch.sum(actual * direction, dim=1, keepdim=True) * direction
    orthogonal = actual - actual_along
    return _finite(torch.linalg.norm(orthogonal, dim=1))


def command_lin_vel_mean(env, command_name: str = "twist") -> torch.Tensor:
    """Current commanded xy speed magnitude."""
    # 指标：当前指令水平速度大小
    cmd = env.command_manager.get_command(command_name)
    return torch.linalg.norm(cmd[:, :2], dim=1)


def command_yaw_vel_abs_mean(env, command_name: str = "twist") -> torch.Tensor:
    """Current commanded yaw speed magnitude."""
    # 指标：当前指令偏航角速度绝对值
    cmd = env.command_manager.get_command(command_name)
    return torch.abs(cmd[:, 2])


def actual_lin_vel_mean(env, asset_cfg=None) -> torch.Tensor:
    """Current actual xy speed magnitude."""
    # 指标：实际 xy 速度大小
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return _finite(torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1))


def tracking_lin_vel_error_band_mean(
    env,
    command_name: str = "twist",
    min_speed: float = 0.0,
    max_speed: float = 10.0,
    asset_cfg=None,
) -> torch.Tensor:
    """Broadcast the masked mean xy error for a command speed band.

    注意返回的是 torch.full_like(err, mean_err)，即把"全体环境的分段平均值"
    广播成 [num_envs] 的形状。因为 MetricsManager 会对每个 term 再取 mean，
    所以广播不会改变最终记录的数字，只是满足"必须返回逐环境张量"的接口要求。
    """
    # 指标（分速段）：只统计指令速度在 [min_speed, max_speed) 区间内的平均 xy 误差
    # 用于观察"低速段/中速段/高速段"各自的跟踪质量
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(cmd[:, :2], dim=1)
    err = _finite(torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2] - cmd[:, :2], dim=1))
    active = torch.logical_and(speed >= min_speed, speed < max_speed)
    denom = active.float().sum().clamp_min(1.0)
    mean_err = torch.sum(torch.where(active, err, torch.zeros_like(err))) / denom
    return torch.full_like(err, mean_err)


def tracking_lin_vel_axis_error_band_mean(
    env,
    axis: int,
    command_name: str = "twist",
    min_speed: float = 0.0,
    max_speed: float = 10.0,
    asset_cfg=None,
) -> torch.Tensor:
    """Broadcast the masked mean x/y error for a command speed band."""
    # 指标（分速段 + 分轴）：单轴（x=0/y=1）的分速段平均误差
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(cmd[:, :2], dim=1)
    err = _finite(torch.abs(asset.data.root_link_lin_vel_b[:, axis] - cmd[:, axis]))
    active = torch.logical_and(speed >= min_speed, speed < max_speed)
    denom = active.float().sum().clamp_min(1.0)
    mean_err = torch.sum(torch.where(active, err, torch.zeros_like(err))) / denom
    return torch.full_like(err, mean_err)


def command_direction_active(env, axis: int, sign: int, command_name: str = "twist") -> torch.Tensor:
    """Report whether the applied command contains a component in this direction.

    Mixed commands count toward multiple directions; these fractions do not
    represent mutually exclusive sampling groups.
    """
    if axis not in (0, 1, 2) or sign not in (-1, 1):
        raise ValueError("Command direction requires an axis in [0, 2] and sign -1 or 1.")
    command = env.command_manager.get_command(command_name)
    return (command[:, axis] * sign > 1.0e-6).float()


def command_band_active(
    env,
    command_name: str = "twist",
    min_speed: float = 0.0,
    max_speed: float = 10.0,
) -> torch.Tensor:
    """Fraction helper for command speed bands."""
    # 指标：指令处于某速段的占比（0/1 掩码），用于归一化分速段统计
    cmd = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(cmd[:, :2], dim=1)
    return torch.logical_and(speed >= min_speed, speed < max_speed).float()


def upright_metric(env, asset_cfg=None) -> torch.Tensor:
    """1 means upright, 0 means fully inverted according to projected gravity."""
    # 指标：直立程度，1=完全直立，0=倒扣（投影重力 z 分量，截断到 [0,1]）
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)


def base_ground_contact_metric(env, sensor_name: str) -> torch.Tensor:
    """Per-step base contact flag for diagnosing reset-biased metrics.

    为什么要单独看这个指标：底盘一旦触地就会被 termination 判为摔倒并 reset，
    如果每个 episode 都很短，那些"平均值"类指标其实是在大量重置样本上算出来的，
    看着漂亮但没意义。这个指标持续偏高说明重启过于频繁，得回头调终止条件或初始化。
    """
    # 指标：底盘是否触地（每步标志），用于诊断"是否因频繁摔倒重置而污染统计"
    sensor = env.scene[sensor_name]
    contact = sensor.data.found > 0
    while contact.ndim > 1:
        contact = torch.any(contact, dim=-1)
    return contact.float()

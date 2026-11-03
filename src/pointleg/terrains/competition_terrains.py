"""Competition obstacle terrains for locomotion tasks.

Implemented in accordance with the interface specifications in
mjlab/src/mjlab/terrains/primitive_terrains.py. Terrain difficulty (difficulty in [0, 1])
is automatically supplied via TerrainGeneratorCfg and curriculum.
#需要根据比赛的场地二次开发

【三个子地形与它们训练的任务】
  RCWallTerrainCfg         横向矮墙（连续多道）→ Robot-Rough-v0 的越障主力地形
  RCLowBarTerrainCfg       低矮横杆（限高门）  → Robot-Crawl-v0 的匍匐钻杆地形
  RCPyramidStairsTerrainCfg 金字塔台阶（改出生点）→ Robot-Rough-v0 的阶梯地形

【统一接口约定】
  每个子地形都要实现 function(difficulty, spec, rng) -> TerrainOutput：
    - difficulty ∈ [0,1] 由课程按地形等级给出，本文件全部用它做"线性插值难度"
    - 创建的几何体挂到 spec.body("terrain") 下，并放进 geometries 列表（带颜色，供渲染/语义分割）
    - 必须返回 spawn origin（机器人在本块地形上的出生坐标，位于本块局部坐标系）

【尺寸约定】所有子地形的 size 默认为 (8.0, 8.0)，MuJoCo 的 box geom 尺寸传的是
  半尺寸（half-size），所以壁厚 wall_thickness 要除以 2   再传进去。

【设计要点：出生点前移】三个地形的 origin 都设在 [1.5, cy, 0] 附近而不是正中，
  给机器人留出 1.5m 平地加速距离，否则一出生就撞在障碍上，开局必摔。
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
# MJLab地形基础接口：所有自定义子地形必须继承 SubTerrainCfg
# TerrainGeometry：绑定生成的geom对象与渲染颜色
# TerrainOutput：function()方法固定返回结构体，包含spawn原点和几何体列表
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
# 工具函数：生成铺满子地形块的平地地面
from mjlab.terrains.utils import make_plane
# MJLab原生金字塔台阶地形，第三个类会继承该配置
from mjlab.terrains import BoxPyramidStairsTerrainCfg
# 颜色工具：根据difficulty输出渐变颜色
from mjlab.utils.color import brand_ramp

# 颜色常量，与primitive_terrains官方代码风格保持一致
_COLOR_ORANGE = (0.95, 0.55, 0.10)
_COLOR_PURPLE = (0.60, 0.20, 0.80)


@dataclass(kw_only=True)
class RCWallTerrainCfg(SubTerrainCfg):
    """ 用于高墙步态训练的重复横向墙体障碍地形。

        机器人必须反复跨步/腾跃越过横向墙体并继续行进。
        当难度系数从0提升至1时，墙体高度将从wall_height_range[0]线性增长至wall_height_range[1]。

        墙体实际物理参数：宽度1.0米，高度0.3米，厚度0.05米。
        难度与几何的对应关系（三处都要同步改，否则会出现"难度涨了但障碍没变"）：
          difficulty=0 → 墙高 0（实际走 level 0 会直接返回纯平地，见下面的早退分支）
          difficulty=1 → 墙高 wall_height_range[1] = 0.35m

        注意：env_cfgs.py 里把 wall_height_range 覆盖成了 (0.10, 0.35)，
        所以实际训练时最矮的墙也有 10cm，不会退化成平地。
    """
    # 墙体高度范围：difficulty=0高度最低，difficulty=1高度最高，线性插值
    wall_height_range: tuple[float, float] = (0.0, 0.35)
    """Wall height range (m), linearly interpolated with difficulty."""
    wall_thickness: float = 0.05
    """Wall thickness (m)."""
    # 墙体长度占地形块Y方向总宽度比例，不完全封死通道，方便调试可视化
    wall_length_frac: float = 0.8
    """Wall length fraction of the terrain width (leaving gaps for visualization/debugging)."""
    platform_width: float = 1.5
    """Nominal start platform width (m)."""
    # 多道墙体沿前进X轴的中心坐标，固定位置
    wall_centers_x: tuple[float, ...] = (2.1, 3.2, 4.3, 5.4, 6.5)
    """Wall center positions along x."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """按难度生成"平地 + 5 道横向矮墙"。

        墙体沿 Y 轴横向铺开（机器人从 X 负方向往正方向跑），5 道墙的间距 1.1m，
        对点足四足机器人来说刚好够一次完整跨越动作后落地再起跳。
        """
        del rng     #本地形不使用随机发生器，消除未使用变量警告
        body = spec.body("terrain")     # 获取terrain body，所有障碍物挂载于此body下
        geometries: list[TerrainGeometry] = []  # 存储geom+对应颜色，用于渲染、语义分割
        
        # 根据难度线性计算当前墙体真实高度
        wall_height = self.wall_height_range[0] + difficulty * (
            self.wall_height_range[1] - self.wall_height_range[0]
        )

        # --------------------------
        # 生成整块子地形的底层平地地面
        # --------------------------
        floor_boxes = make_plane(body, self.size, 0.0, center_zero=False)
        floor_color = (0.45, 0.45, 0.45, 1.0)
        for box in floor_boxes:
            geometries.append(TerrainGeometry(geom=box, color=floor_color))
        # 难度极低，墙高几乎为0，直接返回平地地形，不生成墙体
        if wall_height < 1e-3:
            origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
            return TerrainOutput(origin=origin, geometries=geometries)
        
        # --------------------------
        # 创建多道横向墙体（墙体沿Y轴方向延伸）
        # --------------------------
        wall_length = self.wall_length_frac * self.size[1]  # 墙体实际Y向长度
        cy = self.size[1] / 2  # 地形块Y方向中心
        wall_color = brand_ramp(_COLOR_ORANGE, difficulty)  # 墙体颜色随难度渐变

        # 循环创建每一面墙
        for cx in self.wall_centers_x:
            wall_geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                # MuJoCo box size传入【半尺寸】: half_x, half_y, half_z
                size=(
                    self.wall_thickness / 2,   # half-size x
                    wall_length / 2,           # half-size y
                    wall_height / 2,           # half-size z
                ),
                pos=(cx, cy, wall_height / 2),# box中心z=墙高/2，保证底面贴地
            )
            geometries.append(TerrainGeometry(geom=wall_geom, color=wall_color))

        # 设置机器人生成原点：放在地形左侧平地，留出加速距离，避免开局直接撞墙
        origin = np.array([1.5, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)

@dataclass(kw_only=True)
class RCLowBarTerrainCfg(SubTerrainCfg):
    """Low clearance bar obstacle terrain.
    A horizontal bar is placed along the y-axis, requiring the robot to crouch and duck under.
    As difficulty scales from 0 to 1, the clearance height decreases linearly from
    clearance_range[1] to clearance_range[0] (harder is lower).
    
    The actual competition spec is 0.30m离地 and 1.0m width.
    Here the range is configured as 0.25m to 0.35m to cover and exceed the standard specs.

    与 RCWallTerrainCfg 的难度方向相反：墙是越高越难，横杆是越低越难
    （可用空隙变小，机器人必须压得更低），所以 clearance 用减法而不是加法插值。
    env_cfgs.py 里覆盖成 (0.24, 0.32)，与 Robot-Crawl-v0 的机身目标高度 0.22m 配套。
    """
    clearance_range: tuple[float, float] = (0.25, 0.35)
    """Bar ground clearance range (m). difficulty=0 -> highest/easiest, difficulty=1 -> lowest/hardest."""
    bar_radius: float = 0.025
    """Bar cross‑section radius (m)."""
    bar_length_frac: float = 0.85
    """Bar length fraction of the terrain width (leaving gaps for visualization/debugging)."""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """子地形生成回调函数，由TerrainGenerator自动调用
        Args:
            difficulty: 课程学习难度 [0,1]，0最简单，1最难
            spec: MuJoCo MjSpec内存模型对象，用于创建几何体
            rng: 随机数生成器，本地形无随机逻辑
        Returns:
            TerrainOutput: 返回机器人生成原点与带颜色的几何体列表
        """
        del rng  # 本地形不使用随机数，消除未使用变量警告
        body = spec.body("terrain")  # 获取terrain刚体，所有障碍物挂载到此刚体下
        geometries: list[TerrainGeometry] = []  # 存储geom与对应颜色，供渲染、语义分割使用

        # 难度越高，横杠离地间隙越小；difficulty=0取上限，difficulty=1取下限
        clearance = self.clearance_range[1] - difficulty * (
            self.clearance_range[1] - self.clearance_range[0]
        )
        bar_z = clearance + self.bar_radius  # 横杠中心z坐标：地面到圆柱下表面 + 圆柱半径
        # --------------------------
        # -- Ground floor 生成子地形块底层平地 --
        # --------------------------
        floor_boxes = make_plane(body, self.size, 0.0, center_zero=False)
        floor_color = (0.45, 0.45, 0.45, 1.0)
        for box in floor_boxes:
            geometries.append(TerrainGeometry(geom=box, color=floor_color))

        # --------------------------
        #创建栏杆
        # --------------------------

        #  (oriented along y‑axis) 创建横向主横杠圆柱 --     
        bar_length = self.bar_length_frac * self.size[1]  # 横杠实际Y方向长度
        cx = self.size[0] / 2  # 地形块X方向中心
        cy = self.size[1] / 2  # 地形块Y方向中心
        bar_color = brand_ramp(_COLOR_PURPLE, difficulty)  # 颜色随难度渐变

        # MuJoCo cylinder默认沿着Z轴；绕X轴旋转90°使其沿Y轴横向摆放
        # 旋转四元数：绕X转90°，q = [cos(pi/4), sin(pi/4), 0, 0]
        bar_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(self.bar_radius, bar_length / 2),  # cylinder size=(半径, 半长)
            pos=(cx, cy, bar_z),
        )
        bar_geom.quat = np.array([
            np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0
        ])
        geometries.append(TerrainGeometry(geom=bar_geom, color=bar_color))

        # -- Side supporting posts 创建横杠两侧竖直支撑柱 --
        post_color = bar_color
        post_half_h = bar_z / 2  # 支撑柱半高，底部贴地，顶端托住横杠
        for y_off in [-bar_length / 2, bar_length / 2]:
            post = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(self.bar_radius, post_half_h),
                pos=(cx, cy + y_off, post_half_h),
            )
            geometries.append(TerrainGeometry(geom=post, color=post_color))

        # Spawn origin：机器人生成位置放在左侧平地，留出加速距离，避免出生在横杠正下方
        origin = np.array([1.5, cy, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)

@dataclass(kw_only=True)
class RCPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
    """Refactored pyramid stairs terrain with the spawn origin shifted to the left edge.
    This ensures the robot can sprint/accelerate on flat ground before ascending,
    substantially reducing early falls or flips on vertical steps.

    这个类只做一件事：复用原生金字塔台阶的全部几何逻辑，只改出生点。
    之所以用继承+覆盖 function 而不是改上游代码，是为了保持 vendored mjlab 干净，
    上游升级时不会冲突。
    """

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """复用父类原生台阶生成逻辑，仅修改机器人生成原点
        Args:
            difficulty: 课程难度 [0,1]
            spec: MjSpec模型构建对象
            rng: 随机数生成器
        Returns:
            TerrainOutput: 修改过spawn原点的地形输出对象
        """
        # 调用父类function，完整生成金字塔台阶几何体
        output = super().function(difficulty, spec, rng)
        cy = self.size[1] / 2  # 【修复bug】原代码缺失cy变量定义
        # Relocate the spawn origin to the left platform zone
        # 将机器人生成原点移动至左侧平地，机器人先加速再上台阶，减少开局摔倒、翻倒
        output.origin = np.array([1.5, cy, 0.0])
        return output

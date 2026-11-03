# =============================================================================
# vec_env.py — 向量化环境抽象接口
#
# 【作用】把"训练循环"与"具体仿真器"解耦：HIMOnPolicyRunner 只依赖这组接口，
#   换成别的仿真器或真机回放只需重写一个子类，runner / 算法 / 存储都不用改。
#
# 【与 Gym 接口的两点不同】
#   1) get_observations() 与 get_privileged_observations() 分开取，
#      对应非对称 actor-critic（普通 Gym 只有一个 obs）
#   2) step() 返回 7 个值而不是 5 个，因为 HIM 需要终止时刻的
#      "重置前特权观测"来训练估计器
#
# 【step() 约定返回】
#   (obs, privileged_obs, rewards, dones, infos,
#    termination_ids, termination_privileged_obs)
# =============================================================================

import torch


class VecEnv:
    """Abstract vectorized environment interface for HIMLoco RSL-RL.

    下面的类属性只做类型声明（供类型检查与 IDE 提示），子类必须自己赋值。
    num_privileged_obs 写成 int | None = None 是为了支持"没有 critic 组"的环境：
    此时它为 None，runner 会退化成用 actor 观测喂 critic。
    """

    num_envs: int
    num_obs: int | None = None
    num_one_step_obs: int | None = None
    num_privileged_obs: int | None = None
    num_one_step_privileged_obs: int | None = None
    num_actions: int
    max_episode_length: int
    device: str

    def get_observations(self) -> torch.Tensor:
        """取 actor 观测（已做历史拼接的扁平张量）。"""
        raise NotImplementedError

    def get_privileged_observations(self) -> torch.Tensor | None:
        """取 critic 特权观测；环境没有 critic 组时返回 None。"""
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> tuple:
        """推进一个控制步，返回 7 元组（见文件头约定）。"""
        raise NotImplementedError

    def reset(self) -> tuple:
        """重置全部环境，返回 (obs, info)。"""
        raise NotImplementedError

    def seed(self, seed: int = -1) -> int:
        """设置随机种子；-1 表示不固定。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放仿真资源（销毁 MuJoCo 数据、关闭渲染窗口等）。"""
        raise NotImplementedError

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """各环境当前 episode 已走的步数，用于随机化回合进度。"""
        raise NotImplementedError

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        raise NotImplementedError
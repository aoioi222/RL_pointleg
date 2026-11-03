# =============================================================================
# him_mjlab_vec_env_wrapper.py — mjlab 环境 → HIM runner 的适配层
#
# 【它解决三处接口不匹配】
#   1) 观测形态：mjlab 返回 dict（{"actor": ..., "critic": ...}），runner 要扁平张量
#   2) 历史拼接：HIM 的 actor 需要 obs 历史，而 mjlab 每步只给当前帧，
#      所以在 wrapper 里维护滑动窗口缓冲
#   3) 返回值个数：mjlab 的 env.step 返回 6 个值，HIM runner 要 7 个，
#      多出来的是"本步终止环境在 reset 之前的特权观测"
#
# 【滑动窗口缓冲的时间顺序】
#       obs_history_buf = cat(当前帧, buf[:, :-单步维度])
#   最旧的被舍弃、新的拼在最前面，所以缓冲内顺序是**从新到旧**。
#   估计器是纯 MLP（无时序结构），顺序本身不影响它，但自己写部署代码时
#   必须保持一致，否则输入分布对不上，策略表现会明显变差。
#
# 【动作顺序】策略与环境均使用逐腿顺序：fl、fr、rl、rr，每条腿依次为外展、俯仰、膝。
# =============================================================================

import torch

from him_mjlab.envs.him_manager_based_rl_env import HimManagerBasedRlEnv
from ..env.vec_env import VecEnv
from ...robot_cfg import LEG_JOINT_NAMES


class HimMjlabVecEnvWrapper(VecEnv):
    """Wraps mjlab HimManagerBasedRlEnv for HIMLoco RSL-RL.

    Key features:
    - Converts dict observations to flat tensors
    - Maintains history buffers for actor observations
    - Tracks terminated envs and their pre-reset privileged observations
    - Returns 7-value step required by HIMOnPolicyRunner
    """

    def __init__(
        self,
        env: HimManagerBasedRlEnv,
        history_length: int = 5,
        privileged_history_length: int = 0,
    ):
        # 类型校验：本 wrapper 依赖 HimManagerBasedRlEnv 特有的
        # obs_before_reset 与 group_obs_dim 接口，换别的环境类会静默出错
        if not isinstance(env.unwrapped, HimManagerBasedRlEnv):
            raise ValueError(
                "The environment must be HimManagerBasedRlEnv. "
                f"Got: {type(env)}"
            )

        self.env = env
        # XML 中的关节自然顺序必须是逐腿顺序，策略无需再重排动作。
        self.reorder_indices = list(range(len(LEG_JOINT_NAMES)))

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        if self.num_actions != len(self.reorder_indices):
            raise ValueError(
                "Point-foot policy and environment must both expose 12 leg actions; "
                f"got {self.num_actions}. Check the XML joint names and actuator mapping."
            )
        leg_action_term = self.unwrapped.action_manager.get_term("leg_joint_pos")
        if tuple(leg_action_term.target_names) != LEG_JOINT_NAMES:
            raise ValueError(
                "The XML joint order must be fl/fr/rl/rr with abduction, pitch, knee "
                f"inside each leg; got {tuple(leg_action_term.target_names)}."
            )

        # Single-step observation dimensions from observation manager groups
        group_obs_dim = self.unwrapped.observation_manager.group_obs_dim
        if "actor" in group_obs_dim:
            self._policy_group_name = "actor"
        elif "policy" in group_obs_dim:
            self._policy_group_name = "policy"
        else:
            raise KeyError("Observation groups must include an actor or policy group.")

        # 单步 actor 观测长度：3+3+3+12+12+12=45。
        self.num_one_step_obs = group_obs_dim[self._policy_group_name][0]
        self.history_length = history_length
        # 总维度 = (历史长度 + 1) × 单步维度：+1 是因为缓冲里还包含"当前帧"
        self.num_obs = self.num_one_step_obs * (self.history_length + 1)

        # History buffer for policy observations
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)

        # Termination tracking
        # 存本步终止的环境索引，供 runner 替换 next_critic_obs 使用
        self._termination_ids = torch.tensor([], dtype=torch.long, device=self.device)

        # Privileged observation setup if critic group exists
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_one_step_privileged_obs = (
                self.unwrapped.observation_manager.group_obs_dim["critic"][0]
            )
            self.privileged_history_length = privileged_history_length
            self.num_privileged_obs = self.num_one_step_privileged_obs * (
                self.privileged_history_length + 1
            )
            self.privileged_obs_history_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device
            )
            self._termination_privileged_obs = torch.zeros(
                0, self.num_privileged_obs, device=self.device
            )
        else:
            self.num_one_step_privileged_obs = None
            self.num_privileged_obs = None
            self.privileged_obs_history_buf = None
            self._termination_privileged_obs = None

        # HIMOnPolicyRunner.learn() 不会主动调 reset，因此构造时填充首帧观测历史。
        self.reset()

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self):
        obs_dict, _ = self.env.reset()
        policy_obs = obs_dict.get(
            self._policy_group_name, obs_dict[next(iter(obs_dict))]
        )
        self.obs_history_buf = policy_obs.repeat(1, self.history_length + 1)
        if self.privileged_obs_history_buf is not None and "critic" in obs_dict:
            self.privileged_obs_history_buf = obs_dict["critic"].repeat(
                1, self.privileged_history_length + 1
            )
        return self.obs_history_buf, {"observations": obs_dict}

    def get_observations(self) -> torch.Tensor:
        return self.obs_history_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        return self.privileged_obs_history_buf

    def compute_termination_observations(
        self, env_ids: torch.Tensor, obs_before_reset: torch.Tensor
    ) -> torch.Tensor | None:
        """取出终止环境在 reset 之前的特权观测。

        obs_before_reset 由环境侧提供（在 reset 覆盖状态之前抓的一帧快照），
        形状是 [num_envs, 单步特权维度]；这里按终止索引切片，
        得到 [num_terminated, 单步特权维度]，供估计器作对比学习的目标。
        没有环境终止时返回形状正确的空张量，避免下游 squeeze 出错。
        """
        if len(env_ids) == 0:
            return torch.zeros(
                0, self.num_one_step_privileged_obs, device=self.device
            )
        return obs_before_reset[env_ids]

    def step(self, actions: torch.Tensor):
        """Execute one time-step and return 7 values for HIMLoco.

        Returns:
            obs: History-stacked policy observations
            privileged_obs: Privileged/critic observations
            rewards: Rewards
            dones: Done flags (terminated | truncated)
            infos: Additional info dict
            termination_ids: Indices of envs that terminated this step
            termination_privileged_obs: Pre-reset privileged obs for terminated envs
        """
        if actions.shape[-1] != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} policy actions, got {actions.shape[-1]}."
            )

        # Keep the validated leg-major policy action order when stepping the environment.
        actions_reordered = actions[:, self.reorder_indices]

        # Step the environment (6-value return)
        # obs_before_reset 是"重置前最后一帧观测"，用于给终止环境做特权观测替换
        obs_dict, obs_before_reset, rewards, terminated, truncated, infos = \
            self.env.step(actions_reordered)

        # dones 合并 terminated 与 truncated：对 PPO 来说两者都要中断轨迹
        dones = (terminated | truncated).to(dtype=torch.long)

        # 有限时域环境下不能把 truncated 当 time_outs（否则自举会重复计入），
        # 只有无限时域才把超时信息透传给 runner 做奖励自举
        if not self.unwrapped.cfg.is_finite_horizon:
            infos["time_outs"] = truncated

        # Extract policy observations from dict
        current_obs = obs_dict.get(self._policy_group_name)
        if current_obs is None:
            # 兼容没有命名分组的版本：取字典里的第一项
            first_key = next(iter(obs_dict.keys()))
            current_obs = obs_dict[first_key]

        # Update policy observation history buffer
        # 滑窗更新：新帧放最前，老内容整体后移，末尾多出来的丢弃
        if self.history_length > 0:
            self.obs_history_buf = torch.cat(
                (
                    current_obs[:, :self.num_one_step_obs],
                    self.obs_history_buf[:, :-self.num_one_step_obs],
                ),
                dim=-1,
            )
        else:
            # history_length=0 时退化成"只保留当前帧"
            self.obs_history_buf = current_obs

        # Track terminated environments
        self._termination_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)

        # Update privileged observation history buffer
        if "critic" in obs_dict and self.privileged_obs_history_buf is not None:
            current_privileged_obs = obs_dict["critic"]
            termination_observation = obs_before_reset.get("critic")
            if termination_observation is None:
                termination_observation = obs_before_reset.get(self._policy_group_name)
            self._termination_privileged_obs = self.compute_termination_observations(
                self._termination_ids, termination_observation
            )
            if self.privileged_history_length > 0:
                self.privileged_obs_history_buf = torch.cat(
                    (
                        current_privileged_obs[:, :self.num_one_step_privileged_obs],
                        self.privileged_obs_history_buf[
                            :, :-self.num_one_step_privileged_obs
                        ],
                    ),
                    dim=-1,
                )
            else:
                self.privileged_obs_history_buf = current_privileged_obs

        # NaN/Inf guard
        # 熔断式检查：NaN 一旦进入观测缓冲，会顺着估计器传到 actor，再经动作回到
        # 环境，最后表现为"训着训着 loss 变 nan 但查不出来源"。宁可在这里直接崩掉。
        if torch.isnan(self.obs_history_buf).any() or torch.isinf(self.obs_history_buf).any():
            raise ValueError("NaN/Inf detected in obs_history_buf!")
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            raise ValueError("NaN/Inf detected in rewards!")

        # 返回顺序必须与 HIMOnPolicyRunner 的解包顺序严格一致
        return (
            self.obs_history_buf,
            self.privileged_obs_history_buf,
            rewards,
            dones,
            infos,
            self._termination_ids,
            self._termination_privileged_obs,
        )

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

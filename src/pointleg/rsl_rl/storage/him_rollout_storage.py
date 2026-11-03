# =============================================================================
# him_rollout_storage.py — HIM 版 PPO 经验回放缓冲区
#
# 【张量形状约定（读这个文件先记住这一条）】
#       每个 buffer 都是 [T, N, D]      T = num_transitions_per_env（24 步）
#                                       N = num_envs（并行环境数）
#                                       D = 该量的维度
#   注意第一维是**时间**、第二维是**环境**，与常见的 [N, T, D] 相反。
#   原因是 GAE 沿时间反向递推，时间在第一维时 self.values[step+1] 就是"下一步"。
#   训练时 mini_batch_generator 用 flatten(0,1) 把它摊平成 [T*N, D]。
#
# 【为什么要额外存 mu / sigma / next_privileged_obs】
#   mu/sigma  —— 算 KL 与 PPO 裁剪时需要"采样那一刻"的分布参数
#   next_privileged_obs —— HIM 估计器做对比学习需要"下一步的特权观测"，
#                比普通 PPO 多存一份，且终止处必须用 reset 之前的观测（由 wrapper 提供）
# =============================================================================

import torch


class HIMRolloutStorage:
    """PPO 的 rollout 缓冲区，额外支持 HIM 估计器所需的下一步特权观测。"""

    class Transition:
        """单步样本的临时容器（每步复用同一个实例，见 HIMPPO.transition）。"""

        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.next_critic_observations = None

        def clear(self):
            # 把所有字段重置为 None（等价于重新 __init__）
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape,
                 privileged_obs_shape, actions_shape, device='cpu'):
        """预分配全部缓冲区。

        参数里传的是 shape 元组（如 [num_obs]）而不是维度整数，
        这样用 *shape 展开就能同时兼容"带历史维的多维观测"。
        privileged_obs_shape[0] is not None 这个判断用于支持"没有 critic 组"的场景：
        此时 privileged_obs_shape 传进来是 [None]，两条特权 buffer 就不分配。
        """
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
            self.next_privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()
        # 用 byte 而不是 bool：兼容老版本 rsl_rl 的存储习惯，占用更小

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        """把一个 Transition 写进第 self.step 个时间片，然后指针 +1。

        这里用 copy_ 做原地写入（而不是重新赋值），保证 buffer 的存储地址不变，
        避免每步都重新分配显存。溢出的断言是保护：如果 step 超过 T，
        说明采样循环次数和 num_transitions_per_env 不一致。
        """
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        if self.next_privileged_observations is not None:
            self.next_privileged_observations[self.step].copy_(
                transition.next_critic_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        # 只重置写指针，不清空数据——下一个 rollout 会覆盖，省去一次全量清零
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """沿时间反向递推计算 GAE 优势与回报。

        递推公式（标准 GAE-λ）：
            δ_t      = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
            A_t      = δ_t + γ·λ·(1−done_t)·A_{t+1}
            returns_t = A_t + V(s_t)
        其中 (1−done_t) 把终止时刻之后的未来截断——episode 结束后
        后面的 V 值不该影响前面的优势估计，这是 GAE 里最容易写错的一处。

        最后对优势做整体标准化（减均值除标准差），因为 PPO 的 surrogate loss
        是线性乘优势的，优势量纲变化会直接改变有效学习率。
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values        # 最后一步用外部传入的自举值
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        """统计平均轨迹长度与平均回报，用于日志。

        做法：先把"终止位置"转成一维索引序列，再对相邻索引求差得到每条轨迹的长度。
        开头补一个 -1 是为了让第一条轨迹的长度也能算出来（索引减 -1）。
        注意 done[-1] = 1 是就地改缓冲区（把最后一步强制标成终止），
        这会让统计口径包含最后那条被截断的轨迹；因为紧跟其后就要 clear()，所以无害。
        """
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64),
             flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """按 epoch × minibatch 逐批产出训练数据。

        关键点：
          - flatten(0,1) 把 [T,N,D] 摊平成 [T*N,D]，让时间和环境混在一起随机打乱；
            这对 PPO 是正确的（on-policy 数据不分时序），但如果以后要做
            时序相关的辅助损失（如 RNN），就不能这么摊平。
          - 索引只用 num_mini_batches * mini_batch_size 个（整除后可能丢掉尾巴上的
            几个样本），牺牲少量样本换取每个 minibatch 大小严格相等。
          - 每次调用 generator 只用一次；跨 epoch 的打乱在 epoch 循环外层做了一次，
            即每个 epoch 的批次划分相同，属于该实现的特点。
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_critic_observations_batch = next_critic_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (
                    obs_batch, critic_observations_batch, actions_batch,
                    next_critic_observations_batch, target_values_batch,
                    advantages_batch, returns_batch,
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                )
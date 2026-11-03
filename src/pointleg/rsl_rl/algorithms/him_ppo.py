# =============================================================================
# him_ppo.py — HIM 版 PPO 算法
#
# 【与标准 PPO 的三点差异】
#   1) 观测分两路：actor 用 obs（历史拼接），critic 用 critic_obs（含特权信息）。
#      采样时两路都要存；计算 GAE 时用 critic_obs，算重要性比率时用 obs。
#   2) 多了一个估计器训练步骤：每个 minibatch 先调用 actor_critic.estimator.update()，
#      它有独立优化器，与主 PPO 优化器并行更新。日志里因此多出
#      estimation_loss 和 swap_loss 两条曲线。
#   3) 需要 next_critic_obs（下一步的特权观测）：估计器要拿"下一步"做对比学习，
#      而终止环境必须用 reset 之前的观测（由 env wrapper 提供 termination_privileged_obs），
#      否则会把 reset 后的初始状态错当成"下一步"。
#
# 【KL 自适应学习率】
#   用新旧高斯策略之间的解析 KL 散度（对角协方差，无需采样）判断更新是否过头：
#   KL 过大 → 学习率降为 1/1.5；KL 过小 → 升 1.5 倍；范围钳在 [1e-5, 1e-2]。
# =============================================================================

import torch

from ..modules import HIMActorCritic
from ..storage import HIMRolloutStorage


class HIMPPO:
    actor_critic: HIMActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later（必须由 init_storage 初始化后才能用）
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # 可复用的 Transition 容器：每步填一次、攒够就写进 storage 再 clear，
        # 避免每步都新建对象（在 24 步 × 数千环境的循环里这个开销不可忽略）。
        self.transition = HIMRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        """采样一个动作，并把这一步需要的所有量暂存进 self.transition。

        训练 PPO 时需要用到"旧策略"的以下信息，所以必须在采样时就地保存：
          - actions / actions_log_prob：算重要性比率的分子
          - action_mean / action_std（mu/sigma）：算 KL 与裁剪用
          - values：算 GAE
        全部 .detach()：这些是"采样时的快照"，不能让梯度穿过它们。
        """
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        """环境走完一步后收尾：存样本、处理超时自举、通知网络重置状态。"""
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        # 超时（time_out）不是真失败，只是回合到点被截断，V(s') 依然有意义，
        # 所以把 γV(s') 补进奖励里，避免策略把"正常到点"误判成"摔倒"而学出保守行为。
        # 真正摔倒（terminated）时 infos 里不会有 time_outs，走不到这一步。
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)   # 非循环网络中为空实现

    def compute_returns(self, last_critic_obs):
        """rollout 采集结束后计算 GAE 与回报。

        last_critic_obs 是最后一步之后的状态，作为自举用的 V(s_T+1)。
        """
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """跑完 num_learning_epochs × num_mini_batches 次梯度更新，返回四个平均损失。"""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, \
            target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch in generator:

            # 用当前（更新中的）策略重新前向，得到新的动作分布
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            # 高斯分布解析 KL：KL(N(μ_old,σ_old) || N(μ,σ))
            #   = log(σ_old/σ) + (σ_old² + (μ_old−μ)²) / (2σ²) − 0.5
            # 分母 σ 里加 1e-5 防止 σ 趋近 0 时除零。axis=-1 按动作维求和。
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 超过目标 KL 两倍：策略变化过猛，降 lr；低于一半：学得太慢，升 lr
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Estimator Update
            # 估计器在同一个 minibatch 上单独更新一次（用当前 lr 同步），
            # 它的梯度不会流入 actor/critic，反之亦然。
            estimation_loss, swap_loss = self.actor_critic.estimator.update(
                obs_batch, next_critic_obs_batch, lr=self.learning_rate
            )

            # Surrogate loss
            # ratio = π_new(a|s) / π_old(a|s)，用 log 差再取 exp 比直接除更数值稳定
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            # 裁剪版：把 ratio 限制在 [1−ε, 1+ε]，取 max 得到悲观下界（PPO 的核心）
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                # 价值函数同样做裁剪：把新 value 相对旧 value 的变化限制在 ±ε 内，
                # 防止 critic 在单个 batch 上更新过猛带偏整个优势估计
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总损失：PPO 的目标是最大化 [surrogate + entropy]，此处取负做最小化
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss

        # 除以更新次数得到平均值；storage 在最后清空，为下一轮 rollout 腾位置
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss


# Import nn here to avoid circular dependency at module level
# nn 只在 update() 里用到一次；放在文件末尾是为了避开模块级循环导入。
import torch.nn as nn
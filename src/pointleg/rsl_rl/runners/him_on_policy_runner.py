# =============================================================================
# him_on_policy_runner.py — HIM 训练主循环
#
# 【职责】把"环境采样 → 算回报 → 网络更新 → 记录日志 → 存档"串成一个循环。
#   它不认识具体的机器人，只通过 VecEnv 接口与环境交互。
#
# 【一轮迭代（iteration）的完整流程】
#   1. 采集阶段：跑 num_steps_per_env 步（默认 24 步），每步 act → env.step
#      → process_env_step，把数据攒进 storage
#   2. 算回报：compute_returns，用最后一步的 critic 观测做自举
#   3. 更新阶段：alg.update()，跑 epoch × minibatch 次梯度更新
#   4. 日志与存档：写 TensorBoard、每 save_interval 轮存一次 .pt
#
# 【一个容易踩的坑：next_critic_obs】
#   第 132 行用 termination_ids 把终止环境的 next_critic_obs 替换成
#   termination_privileged_obs（reset 之前的那一份）。如果不换，
#   估计器会把"重置后的初始状态"当成终止状态的下一步去做对比学习，
#   学出来的隐变量在摔倒/重置边界处会系统性错误。
#
# 【本文件在仓库中的状态】属于备用的 HIM/RSL-RL 训练链，当前 env_cfgs.py 走的是
#   mjlab 内置的 rsl_rl runner，本文件未被 __init__ 导出，需要手动对接才能启用。
# =============================================================================

import time
import os
import statistics
from collections import deque

import torch

from ..algorithms import HIMPPO
from ..modules import HIMActorCritic
from ..env import VecEnv


class HIMOnPolicyRunner:
    """On-policy runner for HIMLoco training.

    Uses the HIM architecture: asymmetric actor-critic with history-based
    estimator for privileged information extraction.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Determine observation dimensions
        # critic 观测优先取特权观测；环境没提供 critic 组时退化成和 actor 同一份
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs

        # Initialize policy network
        actor_critic_class = eval(train_cfg["policy_class_name"])
        actor_critic: HIMActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        # Initialize algorithm
        alg_class = eval(train_cfg["algorithm_class_name"])
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # Training configuration
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]

        # Initialize storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions]
        )

        # Logging
        # 日志目录为 None 时全程不写日志也不存档，等价于"干跑"模式
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # 日志后端从配置里的 logger 字段读，缺省 tensorboard（wandb/neptune 需要额外登录）
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """跑 num_learning_iterations 轮训练。

        init_at_random_ep_len=True 会把各环境的回合进度随机打散，
        避免所有环境同时到点重置（那种同步重置会让每 20s 出现一次
        "全体换新场景"的周期性扰动，训练指标会呈锯齿状）。
        """
        if self.log_dir is not None and self.writer is None:
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.alg_cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                if hasattr(self.writer, "store_config"):
                    env_cfg = getattr(self.env, "cfg", None)
                    if env_cfg is None and hasattr(self.env, "unwrapped"):
                        env_cfg = getattr(self.env.unwrapped, "cfg", None)
                    self.writer.store_config(env_cfg, self.cfg)
            elif self.logger_type == "tensorboard":
                # tensorboard 是纯本地后端，不需要账号，离线训练首选
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError(
                    "Logger type not found. Choose 'neptune', 'wandb' or 'tensorboard'."
                )

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        # 只保留最近 100 个 episode 的回报/长度，用于算滑动平均（避免早期差数据长期拉低均值）
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            # inference_mode 覆盖整个采集阶段：PPO 的数据采集不需要梯度，
            # 关掉 autograd 能显著省显存（否则每步的计算图都会挂住）
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, \
                        termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    # 终止环境改用 reset 前的特权观测（见文件头说明），
                    # 否则估计器会拿重置后的初始状态当"下一步"做对比学习
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

                    if self.log_dir is not None:
                        # 两种 episodes 信息键名并存：不同版本的环境用 'episode' 或 'log'
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # 只有结束的环境才把累计值送进滑动窗口，然后清零重新累计
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                # 注意 compute_returns 也在 inference_mode 里：它只做数值计算，
                # 不需要梯度，留在图里会白占显存
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = \
                self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(
                self.log_dir, f'model_{self.current_learning_iteration}.pt'
            ))

    def log(self, locs, width=80, pad=35):
        """把一轮迭代的统计量写进 TensorBoard 并打印到终端。

        locs 是 learn() 里的 locals() 字典，靠变量名取值（所以 learn() 里
        那些变量名不能随便改，改了这里就 KeyError）。
        日志分四组：Episode/*（环境侧）、Loss/*（算法侧）、Policy/*、
        Perf/*（性能），这也是排查问题时看图的顺序。
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ""
        if locs['ep_infos']:
            # 环境每步上报的 episode 信息是"动态键"的字典，这里做一次归并：
            # 收集所有出现过的键 → 各自拼成一个张量 → 取均值记录
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, locs['it'])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar('Episode/' + key, value, locs['it'])
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs
            / (locs['collection_time'] + locs['learn_time'])
        )

        loss_dict = {
            'value_function': locs['mean_value_loss'],
            'surrogate': locs['mean_surrogate_loss'],
            'estimation': locs['mean_estimation_loss'],
            'swap': locs['mean_swap_loss'],
        }
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', value, locs['it'])

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar(
                'Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it']
            )
            self.writer.add_scalar(
                'Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it']
            )
            if self.logger_type != "wandb":
                self.writer.add_scalar(
                    'Train/mean_reward/time', statistics.mean(locs['rewbuffer']),
                    self.tot_time
                )
                self.writer.add_scalar(
                    'Train/mean_episode_length/time',
                    statistics.mean(locs['lenbuffer']), self.tot_time
                )

        str_header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in loss_dict.items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            log_string += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in loss_dict.items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        """保存模型 checkpoint。

        存了四个部分：网络权重、PPO 优化器状态、估计器优化器状态、当前迭代轮数。
        后两者不能省：只存权重会导致断点续训时动量/二阶矩丢失，
        恢复后最初几百轮表现会明显变差。估计器用的是独立优化器，必须单独存。
        """
        saved_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict':
                self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        torch.save(saved_dict, path)
        if self.cfg.get("upload_model", False) and hasattr(self.writer, "save_model"):
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        """加载 checkpoint。

        weights_only=False 是必须的：默认的 weights_only=True 只允许加载纯张量，
        而这里的 dict 里带了 infos 等非张量对象，会直接报错。
        回放（play）时可以传 load_optimizer=False，省掉优化器状态的加载。
        """
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(
                loaded_dict['estimator_optimizer_state_dict']
            )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """切换到 eval 模式并返回确定性推理函数 act_inference。

        返回的是绑定方法而不是模块：调用方只需 policy(obs_history) 就能拿到动作，
        不用关心网络内部结构。device 参数用于把模型搬到部署设备（如 CPU）。
        """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
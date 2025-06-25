import random
import os
import glob
import datetime
import numpy as np
import json
import torch
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard
import multiprocessing as mp

# Worker function for module-level parallel self-play
def _parallel_self_play_worker(args):
    ai_player, use_dlc = args
    env = HiveEnv(training_mode=True, use_dlc=use_dlc)
    target_agent = ai_player.clone()
    explorer = target_agent.clone()
    explorer.epsilon = min(1.0, target_agent.epsilon * 1.5)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    transitions = []
    while not terminated and not truncated:
        current = target_agent if env.current_player_idx == 0 else explorer
        action = current.select_action(env, env.game, env.board, env.current_player_idx)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        transitions.append((obs, action, reward, next_obs, terminated))
        obs = next_obs
    return transitions

# Module-level worker initializer
def _init_worker():
    import signal
    # 忽略 SIGINT，让主进程处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class AITrainer:
    def __init__(self, model_path=None, force_new=False, custom_dir=None, custom_prefix=None, use_dlc=False):
        self.use_dlc = use_dlc  # 是否启用DLC棋子
        self.base_model_dir = "./models"
        if not os.path.exists(self.base_model_dir):
            os.makedirs(self.base_model_dir)
        if force_new:
            # 强制新建唯一目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 根据DLC标记添加后缀
            suffix = 'dlc' if self.use_dlc else 'nodlc'
            self.run_prefix = f"{timestamp}_{suffix}"
            self.model_dir = os.path.join(self.base_model_dir, self.run_prefix)
            os.makedirs(self.model_dir, exist_ok=True)
            print(f"[新训练] 新建模型目录: {self.model_dir}")
        elif custom_dir and custom_prefix:
            self.model_dir = custom_dir
            self.run_prefix = custom_prefix
            print(f"[断点续训] 继续训练: {self.model_dir}, 前缀: {self.run_prefix}")
        else:
            # 默认行为：查找最新断点
            self.latest_dir, self.latest_prefix = self._find_latest_checkpoint()
            if self.latest_dir and self.latest_prefix:
                self.model_dir = self.latest_dir
                self.run_prefix = self.latest_prefix
                print(f"[断点续训] 检测到历史断点: {self.model_dir}, 前缀: {self.run_prefix}")
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = 'dlc' if self.use_dlc else 'nodlc'
                self.run_prefix = f"{timestamp}_{suffix}"
                self.model_dir = os.path.join(self.base_model_dir, self.run_prefix)
                os.makedirs(self.model_dir, exist_ok=True)
                print(f"[新训练] 新建模型目录: {self.model_dir}")
        self.model_path = model_path
        # 初始化环境和AIPlayer，传入DLC选项
        self.env = HiveEnv(training_mode=True, use_dlc=self.use_dlc)
        self.player1_ai = AIPlayer("AI_Player1", is_first_player=True, epsilon=1.0, use_dlc=self.use_dlc)
        self.player2_ai = AIPlayer("AI_Player2", is_first_player=False, epsilon=1.0, use_dlc=self.use_dlc)
        # 断点续训：尝试加载模型和统计
        self.average_rewards = self._try_load_npy(f"{self.run_prefix}_reward_history.npy")
        if self.average_rewards is None:
            self.average_rewards = []
        self.end_stats_history = self._try_load_npy(f"{self.run_prefix}_end_stats_history.npy", allow_pickle=True)
        if self.end_stats_history is None:
            self.end_stats_history = []
        self.episode_steps_history = self._try_load_npy(f"{self.run_prefix}_steps_history.npy")
        if self.episode_steps_history is None:
            self.episode_steps_history = []
        self.illegal_action_count_history = self._try_load_npy(f"{self.run_prefix}_illegal_history.npy")
        if self.illegal_action_count_history is None:
            self.illegal_action_count_history = []
        self.queenbee_step_history = self._try_load_npy(f"{self.run_prefix}_queenbee_step_history.npy")
        if self.queenbee_step_history is None:
            self.queenbee_step_history = []
        self.start_episode = len(self.average_rewards)
        # 仅在非新建模式下加载模型和meta
        if not force_new:
            model_file = os.path.join(self.model_dir, f"{self.run_prefix}_final.npz")
            if os.path.exists(model_file):
                try:
                    self.player1_ai.neural_network.load_model(model_file)
                    self.player2_ai.neural_network.load_model(model_file)
                    print(f"[断点续训] 已加载模型: {model_file}")
                    self._load_meta()  # 加载超参数
                except Exception as e:
                    print(f"[WARN] 加载模型失败: {e}")

    def _find_latest_checkpoint(self):
        # 查找 models/*/xxx_final.npz，按时间排序取最新
        candidates = glob.glob(os.path.join(self.base_model_dir, "*", "*_final.npz"))
        if not candidates:
            return None, None
        candidates.sort()
        latest = candidates[-1]
        latest_dir = os.path.dirname(latest)
        basename = os.path.basename(latest)
        prefix = basename.split("_final.npz")[0]
        return latest_dir, prefix

    def _try_load_npy(self, filename, allow_pickle=False):
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=allow_pickle).tolist()
            except Exception as e:
                print(f"[WARN] 加载{filename}失败: {e}")
        return None

    def train(self, epsilon_decay=0.995, min_epsilon=0.01, batch_size=24, num_workers=10):
        # 类型兜底，防止NoneType.append
        if self.average_rewards is None:
            self.average_rewards = []
        if self.end_stats_history is None:
            self.end_stats_history = []
        if self.episode_steps_history is None:
            self.episode_steps_history = []
        if self.illegal_action_count_history is None:
            self.illegal_action_count_history = []
        if self.queenbee_step_history is None:
            self.queenbee_step_history = []
        print(f"Starting AI training (并行采样worker={num_workers}，Ctrl+C终止)...")
        end_stats = {'win': 0, 'lose': 0, 'draw': 0, 'max_turns': 0, 'other': 0}
        episode = self.start_episode
        # 启动worker进程
        queue = mp.Queue(maxsize=100)
        workers = []
        from parallel_sampler import worker_process
        player_args = dict(name='AI_Parallel', is_first_player=True)
        env_args = dict(training_mode=True)
        for i in range(num_workers):
            w = mp.Process(target=worker_process, args=(queue, player_args, env_args))
            w.daemon = True
            w.start()
            workers.append(w)
        try:
            while True:
                # 从队列收集worker采样结果
                sample = queue.get()  # 阻塞等待
                # sample应包含: obs, action, reward, next_obs, terminated, episode_reward, episode_steps, illegal_count, queenbee_step, info
                # 你可根据parallel_sampler.worker_process实际返回结构调整
                (obs, action, reward, next_obs, terminated, episode_reward, episode_steps, illegal_action_count, queenbee_step, info) = sample
                # 经验回放
                self.player1_ai.add_experience(obs, action, reward, next_obs, terminated)
                loss = self.player1_ai.train_on_batch(batch_size)
                # ---loss历史自动保存---
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                if loss is not None:
                    self.loss_history.append(loss)
                    # 每100步保存一次loss历史
                    if len(self.loss_history) % 100 == 0:
                        np.save(os.path.join(self.model_dir, f"{self.run_prefix}_loss_history.npy"), np.array(self.loss_history))
                # 统计
                reason = info.get('reason', '') if isinstance(info, dict) else ''
                if reason == 'max_turns_reached':
                    end_stats['max_turns'] += 1
                elif reason == 'board_full_draw':
                    end_stats['draw'] += 1
                elif episode_reward == 20.0:
                    end_stats['win'] += 1
                elif episode_reward == -20.0:
                    end_stats['lose'] += 1
                elif reason == 'no_legal_action':
                    end_stats['other'] += 1
                else:
                    end_stats['other'] += 1
                # 自动衰减epsilon（推荐每1000局减半，最低0.05）
                # 修正：仅当episode>0且能整除decay_every时才衰减，避免第一轮直接减半
                if episode > 0:
                    self.player1_ai.update_epsilon(episode, decay_every=1000, decay_rate=0.5, min_epsilon=0.05)
                    self.player2_ai.update_epsilon(episode, decay_every=1000, decay_rate=0.5, min_epsilon=0.05)
                # ---reward异常检测与上限保护---
                REWARD_ABS_LIMIT = 200  # 正常Hive终局奖励绝不应超过±200
                if abs(episode_reward) > REWARD_ABS_LIMIT:
                    print(f"[WARN][REWARD] Episode {episode+1} reward异常: {episode_reward:.2f}，请检查reward shaping/终局奖励/循环。")
                    # reward上限保护，截断极端reward，防止AI刷分
                    episode_reward = max(min(episode_reward, REWARD_ABS_LIMIT), -REWARD_ABS_LIMIT)
                # ---自动统计reward分布---
                if not hasattr(self, '_reward_anomaly_counter'):
                    self._reward_anomaly_counter = {'>limit': 0, '<-limit': 0, 'normal': 0}
                if episode_reward >= REWARD_ABS_LIMIT:
                    self._reward_anomaly_counter['>limit'] += 1
                elif episode_reward <= -REWARD_ABS_LIMIT:
                    self._reward_anomaly_counter['<-limit'] += 1
                else:
                    self._reward_anomaly_counter['normal'] += 1
                if (episode+1) % 1000 == 0:
                    print(f"[REWARD-ANALYSIS] 近1000局 reward 超上限: {self._reward_anomaly_counter['>limit']}，超下限: {self._reward_anomaly_counter['<-limit']}，正常: {self._reward_anomaly_counter['normal']}")
                    self._reward_anomaly_counter = {'>limit': 0, '<-limit': 0, 'normal': 0}
                self.average_rewards.append(episode_reward)
                self.episode_steps_history.append(episode_steps)
                self.illegal_action_count_history.append(illegal_action_count)
                self.queenbee_step_history.append(queenbee_step)
                self.end_stats_history.append(end_stats.copy())
                end_stats = {k: 0 for k in end_stats}
                print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {self.player1_ai.epsilon:.4f}")
                episode += 1
        except KeyboardInterrupt:
            print("\n[INFO] 检测到Ctrl+C，正在保存模型和reward曲线...")
            self._save_checkpoint()
            print("[断点续训] 已保存断点，可下次继续训练。")
        finally:
            print("[INFO] 训练已结束，模型和统计已保存。")

    def _save_checkpoint(self):
        reward_file = os.path.join(self.model_dir, f"{self.run_prefix}_reward_history.npy")
        np.save(reward_file, np.array(self.average_rewards))
        end_stats_file = os.path.join(self.model_dir, f"{self.run_prefix}_end_stats_history.npy")
        np.save(end_stats_file, np.array(self.end_stats_history))
        np.save(os.path.join(self.model_dir, f"{self.run_prefix}_steps_history.npy"), np.array(self.episode_steps_history))
        np.save(os.path.join(self.model_dir, f"{self.run_prefix}_illegal_history.npy"), np.array(self.illegal_action_count_history))
        np.save(os.path.join(self.model_dir, f"{self.run_prefix}_queenbee_step_history.npy"), np.array(self.queenbee_step_history))
        final_model_file = os.path.join(self.model_dir, f"{self.run_prefix}_final.npz")
        self.player1_ai.neural_network.save_model(final_model_file)
        # 保存超参数meta到 JSON
        meta = {
            'player1_epsilon': self.player1_ai.epsilon,
            'player2_epsilon': self.player2_ai.epsilon,
            'player1_lr': getattr(self.player1_ai, 'learning_rate', None),
            'player2_lr': getattr(self.player2_ai, 'learning_rate', None),
            'player1_discount': getattr(self.player1_ai, 'discount_factor', None),
            'player2_discount': getattr(self.player2_ai, 'discount_factor', None),
        }
        meta_file = os.path.join(self.model_dir, f"{self.run_prefix}_meta.json")
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
        print(f"[断点续训] 模型、统计和超参数已保存到 {self.model_dir}")

    def _load_meta(self):
        # 从 JSON 加载超参数
        meta_path = os.path.join(self.model_dir, f"{self.run_prefix}_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if 'player1_epsilon' in meta:
                    self.player1_ai.epsilon = meta['player1_epsilon']
                if 'player2_epsilon' in meta:
                    self.player2_ai.epsilon = meta['player2_epsilon']
                if 'player1_lr' in meta and meta['player1_lr'] is not None:
                    self.player1_ai.learning_rate = meta['player1_lr']
                if 'player2_lr' in meta and meta['player2_lr'] is not None:
                    self.player2_ai.learning_rate = meta['player2_lr']
                if 'player1_discount' in meta and meta['player1_discount'] is not None:
                    self.player1_ai.discount_factor = meta['player1_discount']
                if 'player2_discount' in meta and meta['player2_discount'] is not None:
                    self.player2_ai.discount_factor = meta['player2_discount']
                print(f"[断点续训] 已恢复epsilon等超参数: {meta}")
            except Exception as e:
                print(f"[WARN] 加载meta超参数失败: {e}")

    def load_model(self):
        self.player1_ai.neural_network.load_model(self.model_path)
        self.player2_ai.neural_network.load_model(self.model_path) # Both players use the same model

    def self_play_train(self, num_episodes, batch_size=32, copy_every=100, train_batches_per_episode=3):
        """Self-play training: target network vs explorer network with periodic parameter copy."""
        # 初始化目标网络和探险网络
        target_agent = self.player1_ai
        explorer_agent = target_agent.clone()
        # 探险网络更高探索率
        explorer_agent.epsilon = min(1.0, target_agent.epsilon * 1.5)
        # 自我对弈循环
        for episode in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            terminated = False
            truncated = False
            # 每步采样
            while not terminated and not truncated:
                # 根据当前玩家索引选择网络：先手使用target，后手使用explorer
                if self.env.current_player_idx == 0:
                    current = target_agent
                else:
                    current = explorer_agent
                # 选择动作并执行
                action = current.select_action(self.env, self.env.game, self.env.board, self.env.current_player_idx)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                # 送入经验池
                target_agent.add_experience(obs, action, reward, next_obs, terminated)
                explorer_agent.add_experience(obs, action, reward, next_obs, terminated)
                obs = next_obs
            # 训练目标网络，每集后统一进行小批量训练，减少频繁调用开销
            for _ in range(train_batches_per_episode):
                target_agent.train_on_batch(batch_size)
            # 周期性复制目标网络到探险网络
            if episode % copy_every == 0:
                explorer_agent = target_agent.clone()
                explorer_agent.epsilon = min(1.0, target_agent.epsilon * 1.5)
                print(f"[Self-Play] 复制目标网络参数到探险网络 (Episode {episode})")
            print(f"[Self-Play] Episode {episode}/{num_episodes} 完成.")
        # 训练结束保存模型与统计
        self._save_checkpoint()

    def adversarial_train(self, num_episodes, batch_size=32, train_batches_per_episode=3):
        """Adversarial training: target network vs adversary picking lowest Q actions."""
        target_agent = self.player1_ai
        env = self.env
        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            terminated = False
            truncated = False
            # 每步对抗采样
            while not terminated and not truncated:
                if env.current_player_idx == 0:
                    action = target_agent.select_action(env, env.game, env.board, env.current_player_idx)
                else:
                    legal_actions = env.get_legal_actions()
                    if not legal_actions:
                        action = None
                    else:
                        # 使用目标网络评估，将Q最小的动作视为对抗动作
                        state_vec = target_agent._get_observation_from_game_state(env.game, env.board, env.current_player_idx)
                        action_encs = [target_agent._encode_action(a) for a in legal_actions]
                        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
                        action_tensor = torch.tensor(np.stack(action_encs), dtype=torch.float32)
                        state_batch = state_tensor.repeat(action_tensor.shape[0], 1)
                        sa_batch = torch.cat([state_batch, action_tensor], dim=1)
                        with torch.no_grad():
                            q_vals = target_agent.neural_network(sa_batch).squeeze(-1).cpu().numpy()
                        idx = int(np.argmin(q_vals))
                        action = legal_actions[idx]
                # 执行动作
                next_obs, reward, terminated, truncated, info = env.step(action)
                # 存储至经验池
                target_agent.add_experience(obs, action, reward, next_obs, terminated)
                obs = next_obs
            # 训练目标网络
            for _ in range(train_batches_per_episode):
                target_agent.train_on_batch(batch_size)
            if episode % 100 == 0:
                print(f"[Adversarial] Episode {episode}/{num_episodes} complete.")
        # 保存模型和统计
        self._save_checkpoint()

    def parallel_self_play_train(self, num_episodes, num_workers=4, batch_size=32, train_batches_per_episode=3):
        """Parallelized self-play training using multiprocessing with proper signal handling."""
        import multiprocessing as mp
        
        # 准备参数列表
        args_list = [(self.player1_ai, self.use_dlc) for _ in range(num_episodes)]
        
        # 使用带有初始化函数的进程池
        pool = mp.Pool(processes=num_workers, initializer=_init_worker)
        results = []
        
        try:
            # 分批提交任务，每批100个，便于中断
            batch_size_mp = 100
            for i in range(0, num_episodes, batch_size_mp):
                batch_end = min(i + batch_size_mp, num_episodes)
                batch_args = args_list[i:batch_end]
                
                print(f"[Parallel Self-Play] Submitting episodes {i+1}-{batch_end}/{num_episodes}")
                batch_results = pool.map(_parallel_self_play_worker, batch_args)
                
                # 立即处理结果
                for idx, eps in enumerate(batch_results, start=i+1):
                    for obs, action, reward, next_obs, terminated in eps:
                        self.player1_ai.add_experience(obs, action, reward, next_obs, terminated)
                    if idx % 10 == 0:
                        print(f"[Parallel Self-Play] Processed Episode {idx}/{num_episodes}")
                
                results.extend(batch_results)
                
        except KeyboardInterrupt:
            print("\n[Parallel Self-Play] Detected KeyboardInterrupt, terminating workers...")
            pool.terminate()
            pool.join()
            print(f"[Parallel Self-Play] Collected {len(results)} episodes before interruption")
            # Propagate interrupt to stop entire training pipeline
            raise
        else:
            pool.close()
            pool.join()
            
        # 批量训练
        print(f"[Parallel Self-Play] Training on {len(results)} episodes...")
        for _ in range(train_batches_per_episode):
            self.player1_ai.train_on_batch(batch_size)
        print(f"[Parallel Self-Play] Completed {num_episodes} episodes with {num_workers} workers.")
        self._save_checkpoint()

    def curriculum_train(self, episodes_per_level=5000):
        """Curriculum training: multi-stage self-play training."""
        levels = [
            {'id': 1, 'desc': '放置蜂后阶段'},
            {'id': 2, 'desc': '全动作阶段'}
        ]
        for lvl in levels:
            print(f"[Curriculum] Level {lvl['id']}: {lvl['desc']}，训练局数={episodes_per_level}")
            # 在此可根据 lvl 配置调整 env 规则，如仅允许放置蜂后等
            self.self_play_train(num_episodes=episodes_per_level)
        print("[Curriculum] 课程学习所有阶段训练完成，保存模型。")
        self._save_checkpoint()





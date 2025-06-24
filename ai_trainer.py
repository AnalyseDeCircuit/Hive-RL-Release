import random
import os
import glob
import datetime
import numpy as np
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard
import multiprocessing as mp

class AITrainer:
    def __init__(self, model_path=None, force_new=False, custom_dir=None, custom_prefix=None):
        self.base_model_dir = "./models"
        if not os.path.exists(self.base_model_dir):
            os.makedirs(self.base_model_dir)
        if force_new:
            # 强制新建唯一目录
            self.run_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
                self.run_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.model_dir = os.path.join(self.base_model_dir, self.run_prefix)
                os.makedirs(self.model_dir, exist_ok=True)
                print(f"[新训练] 新建模型目录: {self.model_dir}")
        self.model_path = model_path
        self.env = HiveEnv(training_mode=True)
        self.player1_ai = AIPlayer("AI_Player1", is_first_player=True, epsilon=1.0)
        self.player2_ai = AIPlayer("AI_Player2", is_first_player=False, epsilon=1.0)
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
        # 保存超参数meta
        meta = {
            'player1_epsilon': self.player1_ai.epsilon,
            'player2_epsilon': self.player2_ai.epsilon,
            'player1_lr': getattr(self.player1_ai, 'learning_rate', None),
            'player2_lr': getattr(self.player2_ai, 'learning_rate', None),
            'player1_discount': getattr(self.player1_ai, 'discount_factor', None),
            'player2_discount': getattr(self.player2_ai, 'discount_factor', None),
        }
        np.save(os.path.join(self.model_dir, f"{self.run_prefix}_meta.npy"), meta)
        print(f"[断点续训] 模型、统计和超参数已保存到 {self.model_dir}")

    def _load_meta(self):
        meta_path = os.path.join(self.model_dir, f"{self.run_prefix}_meta.npy")
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path, allow_pickle=True).item()
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





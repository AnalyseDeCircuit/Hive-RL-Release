import random
import os
import glob
import datetime
import numpy as np
import json
import torch
import time
import signal
import sys
import atexit
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard
import multiprocessing as mp

# 全局变量存储worker进程，用于退出时清理
_global_workers = []
_global_pools = []

# 模块级 reward shaping 函数，确保可被 multiprocessing pickle
def _shaping_transition(r, term):
    # 削弱微小中间奖励
    return 0 if not term and abs(r) < 0.1 else r

def _shaping_finetune(r, term):
    # 仅终局奖励
    return r if term else 0

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

# 强制清理所有后台进程
def _cleanup_all_processes():
    """强制清理所有后台进程和线程"""
    global _global_workers, _global_pools
    
    print("\n[Cleanup] 正在强制清理所有后台进程...")
    
    # 清理worker进程
    for worker in _global_workers:
        if worker.is_alive():
            print(f"[Cleanup] 终止worker进程 {worker.pid}")
            worker.terminate()
            worker.join(timeout=2)
            if worker.is_alive():
                print(f"[Cleanup] 强制杀死worker进程 {worker.pid}")
                try:
                    import psutil
                    process = psutil.Process(worker.pid)
                    process.kill()
                except:
                    pass
    
    # 清理进程池
    for pool in _global_pools:
        try:
            pool.terminate()
            pool.join(timeout=2)
        except:
            pass
    
    # 清空列表
    _global_workers.clear()
    _global_pools.clear()
    
    print("[Cleanup] 进程清理完成")

def _cleanup_workers(workers):
    """清理指定的worker进程列表"""
    print(f"\n[Cleanup] 正在清理 {len(workers)} 个worker进程...")
    
    for i, worker in enumerate(workers):
        if worker.is_alive():
            print(f"[Cleanup] 终止worker {i+1}/{len(workers)} (PID: {worker.pid})")
            worker.terminate()
            worker.join(timeout=3)
            
            if worker.is_alive():
                print(f"[Cleanup] 强制杀死worker {i+1}/{len(workers)} (PID: {worker.pid})")
                try:
                    # 尝试使用 psutil 强制杀死进程
                    import psutil
                    process = psutil.Process(worker.pid)
                    process.kill()
                    worker.join(timeout=1)
                except ImportError:
                    # 如果没有 psutil，使用系统级杀死
                    try:
                        import os
                        if os.name == 'nt':  # Windows
                            os.system(f"taskkill /F /PID {worker.pid}")
                        else:  # Unix/Linux
                            os.system(f"kill -9 {worker.pid}")
                    except:
                        pass
                except:
                    pass
    
    print("[Cleanup] Worker进程清理完成")

# 注册退出清理函数
atexit.register(_cleanup_all_processes)

# 信号处理函数
def _signal_handler(signum, frame):
    """信号处理函数，确保优雅退出"""
    print(f"\n[Signal] 收到信号 {signum}，开始清理...")
    _cleanup_all_processes()
    print("[Signal] 清理完成，退出程序")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

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

    def train(self, epsilon_decay=0.995, min_epsilon=0.01, batch_size=24, num_workers=10, max_episodes=10000, curriculum_epsilon_config=None):
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
        print(f"Starting AI training (并行采样worker={num_workers}，最大{max_episodes}局，Ctrl+C终止)...")
        end_stats = {'win': 0, 'lose': 0, 'draw': 0, 'max_turns': 0, 'other': 0}
        episode = self.start_episode
        
        # 启动worker进程 - 修复：正确传递reward_shaper配置和epsilon同步
        queue = mp.Queue(maxsize=100)
        epsilon_sync_queues = []  # 为每个worker创建独立的epsilon同步队列
        workers = []
        from parallel_sampler import worker_process
        
        # 传递当前的epsilon值给worker
        player_args = dict(name='AI_Parallel', is_first_player=True, use_dlc=self.use_dlc, epsilon=self.player1_ai.epsilon)
        
        # 修复：正确传递reward_shaper给worker环境
        current_reward_shaper = getattr(self.env, 'reward_shaper', None)
        reward_shaper_config = None
        if current_reward_shaper:
            # 提取reward_shaper的配置信息
            reward_shaper_config = {
                'phase': getattr(current_reward_shaper, 'phase', 'foundation')
            }
            print(f"[Trainer] 传递奖励整形配置给workers: {reward_shaper_config}")
        
        env_args = dict(training_mode=True, use_dlc=self.use_dlc)
        
        episodes_per_worker = max(10, max_episodes // num_workers)
        for i in range(num_workers):
            # 为每个worker创建独立的epsilon同步队列
            epsilon_sync_queue = mp.Queue(maxsize=10)
            epsilon_sync_queues.append(epsilon_sync_queue)
            
            w = mp.Process(target=worker_process, args=(queue, player_args, env_args, episodes_per_worker, reward_shaper_config, epsilon_sync_queue))
            w.daemon = True
            w.start()
            workers.append(w)
            # 添加到全局列表用于清理
            _global_workers.append(w)
        
        print(f"[Trainer] 启动 {num_workers} 个worker，初始epsilon: {self.player1_ai.epsilon:.4f}")
        
        # 添加worker存活检查
        workers_alive = num_workers
        samples_received = 0
        
        # 新增：跟踪当前episode的状态
        current_episode_data = None
        episode_transition_count = 0
        
        try:
            while episode < max_episodes and workers_alive > 0:
                try:
                    # 从队列收集worker采样结果 - 添加超时避免无限等待
                    sample = queue.get(timeout=30)  # 30秒超时
                    samples_received += 1
                except:
                    # 检查worker状态
                    alive_count = sum(1 for w in workers if w.is_alive())
                    if alive_count == 0:
                        print("所有worker已退出，结束训练")
                        break
                    else:
                        print(f"队列超时，当前存活worker: {alive_count}")
                        continue
                
                # sample应包含: obs, action, reward, next_obs, terminated, episode_reward, episode_steps, illegal_count, queenbee_step, info
                try:
                    (obs, action, reward, next_obs, terminated, episode_reward, episode_steps, illegal_action_count, queenbee_step, info) = sample
                except ValueError as e:
                    print(f"样本解析错误: {e}, sample: {sample}")
                    continue
                
                # 只处理terminated=True的样本（episode结束）
                if not terminated:
                    print("[DEBUG] 收到未终止的样本，跳过统计但仍进行训练")
                
                # 经验回放 - 所有样本都用于训练
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
                
                # 更新当前episode数据
                current_episode_data = {
                    'episode_reward': episode_reward,
                    'episode_steps': episode_steps,
                    'illegal_action_count': illegal_action_count,
                    'queenbee_step': queenbee_step,
                    'info': info,
                    'terminated': terminated
                }
                episode_transition_count += 1
                
                # 只有当这一步确实结束了episode时，才进行终局统计和记录
                if terminated:
                    reason = info.get('reason', '') if isinstance(info, dict) else ''
                    
                    # 修复：严格基于游戏逻辑的reason进行分类，不依赖reward
                    # 这确保分类反映真实的游戏结果，而不是reward shaping的副作用
                    
                    if reason == 'max_turns_reached':
                        end_stats['max_turns'] += 1
                    elif reason in ['board_full_draw', 'draw']:
                        end_stats['draw'] += 1
                    elif reason == 'player1_win':
                        # 玩家1获胜：对于当前玩家来说是胜利还是失败？
                        # 这里需要知道当前是谁的回合
                        # 但是由于parallel_sampler的设计，我们总是从玩家1视角统计
                        end_stats['win'] += 1
                    elif reason == 'player2_win':
                        # 玩家2获胜：对于玩家1来说是失败
                        end_stats['lose'] += 1
                    elif reason == 'queen_surrounded':
                        # 蜂后被围：失败
                        end_stats['lose'] += 1
                    elif reason == 'no_legal_action':
                        end_stats['other'] += 1
                    elif reason == 'must_place_queen_violation':
                        end_stats['other'] += 1
                    elif reason == 'illegal_action':
                        end_stats['other'] += 1
                    elif reason == 'unknown_termination':
                        # 异常终止，原因未知
                        end_stats['other'] += 1
                    elif reason == '':
                        # reason为空的情况：这是最大的问题！
                        # 在这种情况下，我们需要依赖其他信息来判断
                        # 临时使用reward作为最后手段，但这不是最佳解决方案
                        if abs(episode_reward) < 0.01:
                            end_stats['draw'] += 1  # 接近0奖励可能是平局或超时
                        elif episode_reward > 0:
                            end_stats['win'] += 1   # 暂时处理
                        else:
                            end_stats['lose'] += 1  # 暂时处理
                        
                        # 记录reason为空的情况，用于调试
                        if not hasattr(self, '_empty_reason_count'):
                            self._empty_reason_count = 0
                        self._empty_reason_count += 1
                    else:
                        # 未知的reason
                        end_stats['other'] += 1
                        if not hasattr(self, '_unknown_reasons'):
                            self._unknown_reasons = set()
                        self._unknown_reasons.add(reason)
                    
                    # 调试信息：记录reason分布和空reason情况
                    if not hasattr(self, '_reason_debug_count'):
                        self._reason_debug_count = {}
                    if reason not in self._reason_debug_count:
                        self._reason_debug_count[reason] = 0
                    self._reason_debug_count[reason] += 1
                    
                    # 每1000个episode打印一次reason统计和空reason数量
                    if episode % 1000 == 0 and episode > 0:
                        print(f"[REASON-DEBUG] Episode {episode} - Reason统计:")
                        for r, count in self._reason_debug_count.items():
                            print(f"  '{r}': {count}")
                        
                        if hasattr(self, '_empty_reason_count'):
                            print(f"  空reason数量: {self._empty_reason_count}")
                            
                        if hasattr(self, '_unknown_reasons'):
                            print(f"  未知reasons: {self._unknown_reasons}")
                            
                        self._reason_debug_count = {}  # 重置计数器
                    
                    # 记录episode级别的统计
                    self.average_rewards.append(episode_reward)
                    self.episode_steps_history.append(episode_steps)
                    self.illegal_action_count_history.append(illegal_action_count)
                    self.queenbee_step_history.append(queenbee_step)
                    self.end_stats_history.append(end_stats.copy())
                    
                    # 实时保存奖励数据，供监控器读取
                    if episode % 10 == 0:  # 每10个episode保存一次，减少IO开销
                        reward_file = os.path.join(self.model_dir, f"{self.run_prefix}_reward_history.npy")
                        np.save(reward_file, np.array(self.average_rewards))
                    
                    # Epsilon管理：课程学习优先，否则使用简单衰减
                    old_epsilon = self.player1_ai.epsilon
                    if curriculum_epsilon_config is not None:
                        # 课程学习的线性epsilon衰减 - 修复：使用阶段内的episode数
                        total_episodes_before = curriculum_epsilon_config.get('total_episodes_before', 0)
                        current_total_episodes = len(self.average_rewards) if self.average_rewards else 0
                        episodes_in_current_phase = current_total_episodes - total_episodes_before
                        
                        if episodes_in_current_phase < curriculum_epsilon_config['decay_episodes']:
                            progress = episodes_in_current_phase / curriculum_epsilon_config['decay_episodes']
                            current_epsilon = (curriculum_epsilon_config['start'] * (1 - progress) + 
                                             curriculum_epsilon_config['end'] * progress)
                            self.player1_ai.epsilon = max(current_epsilon, curriculum_epsilon_config['end'])
                            self.player2_ai.epsilon = max(current_epsilon, curriculum_epsilon_config['end'])
                        else:
                            self.player1_ai.epsilon = curriculum_epsilon_config['end']
                            self.player2_ai.epsilon = curriculum_epsilon_config['end']
                    elif episode > 0 and epsilon_decay < 1.0 and episode % 1000 == 0:
                        # 独立训练时的简单epsilon衰减
                        self.player1_ai.epsilon = max(self.player1_ai.epsilon * epsilon_decay, min_epsilon)
                        self.player2_ai.epsilon = max(self.player2_ai.epsilon * epsilon_decay, min_epsilon)
                    
                    # 关键修复：同步epsilon到所有worker进程
                    if abs(self.player1_ai.epsilon - old_epsilon) > 1e-6:  # epsilon发生了变化
                        for epsilon_sync_queue in epsilon_sync_queues:
                            try:
                                # 非阻塞发送新的epsilon值
                                epsilon_sync_queue.put_nowait(self.player1_ai.epsilon)
                            except:
                                # 队列满了，跳过这个worker（它会在下次episode时更新）
                                pass
                    
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
                    
                    # 重置统计计数器
                    end_stats = {k: 0 for k in end_stats}
                    episode_transition_count = 0
                    
                    # 打印episode信息
                    print(f"Episode {episode+1}, Cumulative: {episode_reward:.2f}, Steps: {episode_steps}, Reason: {reason}, Epsilon: {self.player1_ai.epsilon:.4f}")
                    episode += 1
        except KeyboardInterrupt:
            print("\n[INFO] 检测到Ctrl+C，正在保存模型和reward曲线...")
            self._save_checkpoint()
            print("[断点续训] 已保存断点，可下次继续训练。")
            # 立即强制终止所有worker进程
            _cleanup_workers(workers)
            raise  # 重新抛出异常以便上层处理
        finally:
            # 确保所有worker都被正确清理
            _cleanup_workers(workers)
            # 从全局列表中移除已清理的worker
            for w in workers:
                try:
                    if w in _global_workers:
                        _global_workers.remove(w)
                except Exception:
                    pass
            print("[INFO] 所有worker已终止，训练结束。")

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
        _global_pools.append(pool)  # 添加到全局列表用于清理
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
            # 从全局列表中移除
            if pool in _global_pools:
                _global_pools.remove(pool)
            # Propagate interrupt to stop entire training pipeline
            raise
        finally:
            try:
                pool.close()
                pool.join()
            except:
                pool.terminate()
                pool.join()
            # 从全局列表中移除
            if pool in _global_pools:
                _global_pools.remove(pool)
            
        # 批量训练
        print(f"[Parallel Self-Play] Training on {len(results)} episodes...")
        for _ in range(train_batches_per_episode):
            self.player1_ai.train_on_batch(batch_size)
        print(f"[Parallel Self-Play] Completed {num_episodes} episodes with {num_workers} workers.")
        self._save_checkpoint()

    def curriculum_train(self):
        """修复后的课程学习: 防止训练退化"""
        # 保存原始环境设置
        original_reward_shaper = getattr(self.env, 'reward_shaper', None)
        
        # 初始化防退化机制
        self.phase_models = {}  # 保存每个阶段的最佳模型
        self.baseline_performance = []  # 记录每个阶段的基线性能
        
        # 导入新的奖励整形系统
        try:
            from improved_reward_shaping import create_curriculum_phases, HiveRewardShaper
            phases = self._create_fixed_curriculum_phases()  # 使用修复后的配置
            print("[Curriculum-Fixed] 使用修复后的奖励整形系统")
        except ImportError:
            print("[Curriculum] 奖励整形系统未找到，使用原有配置")
            phases = self._create_fixed_curriculum_phases()
        
        total_episodes_before = len(self.average_rewards) if self.average_rewards else 0
        total_episodes_target = sum(phase['episodes'] for phase in phases)
        
        print(f"[Curriculum] 总训练目标: {total_episodes_target:,} episodes")
        print(f"[Curriculum] 预计时间: 3-4小时 (10并发)")
        
        for phase_idx, phase in enumerate(phases):
            print(f"\n[Curriculum] 阶段 {phase_idx+1}/3: {phase['name']}")
            print(f"[Curriculum] 描述: {phase['description']}")
            print(f"[Curriculum] 本阶段episodes: {phase['episodes']:,}")
            print(f"[Curriculum] Epsilon: {phase['epsilon_start']} -> {phase['epsilon_end']}")
            print(f"[Curriculum] 按Ctrl+C进入下一阶段，连按两次完全退出")
            
            # 设置奖励整形器
            reward_shaper = phase.get('reward_shaper')
            if reward_shaper:
                print(f"[Curriculum] 使用奖励整形: {phase['name']} 阶段")
                self.env.reward_shaper = reward_shaper
            else:
                print(f"[Curriculum] 阶段 {phase['name']} 未配置奖励整形器，使用原始奖励系统")
            
            # 设置epsilon参数
            self.player1_ai.epsilon = phase['epsilon_start']
            self.player2_ai.epsilon = phase['epsilon_start']
            
            # 检查训练退化
            degradation_detected = self._check_training_degradation(phase_idx) if phase_idx > 0 else False
            if degradation_detected:
                self._apply_degradation_recovery(phase_idx)
            
            # 配置epsilon线性衰减 - 直接在训练循环中管理，不再需要update_epsilon方法
            epsilon_decay_episodes = phase.get('epsilon_decay_episodes', 5000)
            
            # 记录这个阶段的参数，供训练循环使用
            phase_epsilon_config = {
                'start': phase['epsilon_start'],
                'end': phase['epsilon_end'],
                'decay_episodes': epsilon_decay_episodes,
                'total_episodes_before': total_episodes_before
            }
            
            # 连续训练整个阶段 - 不再分批
            episodes_before = len(self.average_rewards) if self.average_rewards else 0
            
            try:
                print(f"[Curriculum] 开始连续训练 {phase['episodes']:,} episodes...")
                
                # 关键改进：连续训练，禁用train()内部的epsilon衰减
                self.train(
                    max_episodes=phase['episodes'], 
                    num_workers=10, 
                    epsilon_decay=1.0,  # 禁用内部epsilon衰减
                    min_epsilon=0.0,    # 不限制最小值
                    curriculum_epsilon_config=phase_epsilon_config  # 传递epsilon配置
                )
                
                episodes_after = len(self.average_rewards) if self.average_rewards else 0
                actual_trained = episodes_after - episodes_before
                
                # 保存阶段检查点
                self._save_phase_checkpoint(phase_idx, phase['name'])
                
                print(f"[Curriculum] 阶段 {phase['name']} 完成，实际训练: {actual_trained:,} episodes")
                print(f"[Curriculum] 最终Epsilon: P1={self.player1_ai.epsilon:.3f}, P2={self.player2_ai.epsilon:.3f}")
                
            except KeyboardInterrupt:
                # 强制清理所有进程
                print(f"\n[Curriculum] 阶段 {phase['name']} 被中断，正在清理进程...")
                _cleanup_all_processes()
                
                # 确保保存当前进度
                print("[Curriculum] 正在保存进度...")
                self._save_checkpoint()
                print("[Curriculum] 进度已保存")
                
                # 检查是否是连续的Ctrl+C
                print("\n[Curriculum] 训练已中断")
                print("选项:")
                print("  1. 按 Enter 继续当前阶段")
                print("  2. 输入 'next' 跳到下一阶段")
                print("  3. 输入 'exit' 完全退出")
                print("  4. 再次按 Ctrl+C 强制退出")
                
                try:
                    user_choice = input("请选择: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    # 处理连续Ctrl+C或输入中断
                    print("\n[Curriculum] 检测到强制退出信号，立即退出")
                    _cleanup_all_processes()
                    sys.exit(0)
                
                if user_choice == 'exit':
                    print("[Curriculum] 用户选择退出课程学习")
                    _cleanup_all_processes()
                    break
                elif user_choice == 'next':
                    print(f"[Curriculum] 跳过阶段 {phase['name']}，进入下一阶段")
                    continue
                else:
                    print(f"[Curriculum] 继续阶段 {phase['name']}")
                    # 计算剩余episodes继续训练
                    episodes_after = len(self.average_rewards) if self.average_rewards else 0
                    remaining = phase['episodes'] - (episodes_after - episodes_before)
                    if remaining > 0:
                        try:
                            self.train(
                                max_episodes=remaining, 
                                num_workers=10,
                                epsilon_decay=1.0,  # 禁用内部epsilon衰减
                                min_epsilon=0.0,
                                curriculum_epsilon_config=phase_epsilon_config  # 传递epsilon配置
                            )
                        except KeyboardInterrupt:
                            print(f"\n[Curriculum] 阶段 {phase['name']} 二次中断，强制退出")
                            _cleanup_all_processes()
                            self._save_checkpoint()
                            sys.exit(0)
            
            total_episodes_before += phase['episodes']
        
        print("\n[Curriculum] 课程学习训练完成！")
        
        # 恢复原始设置并最终保存
        self.env.reward_shaper = original_reward_shaper
        self._save_checkpoint()
        print("[Curriculum] 最终模型已保存")
        
        total_episodes_after = len(self.average_rewards) if self.average_rewards else 0
        print(f"\n[Curriculum] 课程学习完成!")
        print(f"[Curriculum] 总训练episodes: {total_episodes_after:,}")
        print(f"[Curriculum] 新增episodes: {total_episodes_after - (total_episodes_after - total_episodes_target):,}")
        
        # 确保最终保存
        try:
            self._save_checkpoint()
            print("[Curriculum] 最终模型已保存")
        except Exception as e:
            print(f"[Curriculum] 保存模型时出错: {e}")

    def curriculum_train_with_signal_handling(self):
        """带信号处理的课程训练 - 确保Ctrl+C时正确保存"""
        import signal
        import sys
        
        def signal_handler(signum, frame):
            print(f"\n[Curriculum] 接收到退出信号 {signum}")
            print("[Curriculum] 正在保存当前进度...")
            try:
                self._save_checkpoint()
                print("[Curriculum] 进度保存完成")
            except Exception as e:
                print(f"[Curriculum] 保存失败: {e}")
            finally:
                print("[Curriculum] 强制退出")
                sys.exit(0)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            self.curriculum_train()
        except Exception as e:
            print(f"[Curriculum] 训练过程中出现异常: {e}")
            self._save_checkpoint()
            raise
    
    def _create_fixed_curriculum_phases(self):
        """
        创建修复后的课程学习阶段配置 - 防止训练退化
        
        主要修复：
        1. 渐进式奖励权重，避免跳跃变化
        2. 更保守的epsilon衰减
        3. 增加各阶段的训练量
        """
        try:
            from improved_reward_shaping import HiveRewardShaper
        except ImportError:
            HiveRewardShaper = None
            
        return [
            {
                'name': 'foundation',
                'episodes': 35000,  # 增加基础训练量
                'description': '基础规则学习 - 重点掌握合法动作和基本生存策略',
                'reward_shaper': HiveRewardShaper('foundation') if HiveRewardShaper else None,
                'epsilon_start': 0.9,
                'epsilon_end': 0.75,  # 更保守的衰减，避免过早收敛
                'epsilon_decay_episodes': 10000,  # 延长衰减时间
            },
            {
                'name': 'strategy',
                'episodes': 45000,  # 增加策略训练量
                'description': '战略发展阶段 - 学习攻防平衡和棋子协调',
                'reward_shaper': HiveRewardShaper('strategy') if HiveRewardShaper else None,
                'epsilon_start': 0.75,  # 承接上一阶段
                'epsilon_end': 0.35,   # 更保守的衰减
                'epsilon_decay_episodes': 15000,
            },
            {
                'name': 'mastery',
                'episodes': 40000,  # 精通阶段适度训练量
                'description': '高级策略精通 - 复杂局面处理和深度计算',
                'reward_shaper': HiveRewardShaper('mastery') if HiveRewardShaper else None,
                'epsilon_start': 0.35,  # 承接上一阶段
                'epsilon_end': 0.05,    # 最终保持小量探索
                'epsilon_decay_episodes': 12000,
            }
        ]
    
    def _save_phase_checkpoint(self, phase_idx, phase_name):
        """保存阶段检查点，防止训练退化"""
        checkpoint_dir = os.path.join(self.model_dir, "phase_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        phase_model_path = os.path.join(checkpoint_dir, f"phase_{phase_idx}_{phase_name}.npz")
        self.player1_ai.neural_network.save_model(phase_model_path)
        
        # 保存性能指标 - 修复类型错误
        if self.average_rewards and len(self.average_rewards) > 0:
            avg_reward = np.mean(self.average_rewards[-1000:]) if len(self.average_rewards) >= 1000 else np.mean(self.average_rewards)
            episode_count = len(self.average_rewards)
        else:
            avg_reward = 0.0
            episode_count = 0
            
        phase_performance = {
            'average_reward': float(avg_reward),
            'episode_count': episode_count,
            'epsilon': self.player1_ai.epsilon,
            'phase_name': phase_name
        }
        
        performance_path = os.path.join(checkpoint_dir, f"phase_{phase_idx}_{phase_name}_performance.json")
        with open(performance_path, 'w') as f:
            json.dump(phase_performance, f)
        
        if not hasattr(self, 'phase_models'):
            self.phase_models = {}
        
        self.phase_models[phase_idx] = {
            'model_path': phase_model_path,
            'performance': phase_performance
        }
        
        print(f"[Phase-Checkpoint] 阶段 {phase_name} 检查点已保存，性能: {phase_performance['average_reward']:.3f}")
    
    def _check_training_degradation(self, phase_idx):
        """检查是否发生训练退化"""
        if phase_idx == 0 or not self.average_rewards or len(self.average_rewards) < 1000:
            return False
            
        # 获取最近1000个episode的平均性能
        recent_performance = np.mean(self.average_rewards[-1000:])
        
        # 与上一阶段的基线比较
        if hasattr(self, 'phase_models') and phase_idx - 1 in self.phase_models:
            baseline_performance = self.phase_models[phase_idx - 1]['performance']['average_reward']
            
            # 如果性能下降超过30%，认为发生了退化
            degradation_threshold = 0.7
            if recent_performance < baseline_performance * degradation_threshold:
                print(f"[Degradation-Warning] 检测到训练退化！")
                print(f"  当前性能: {recent_performance:.3f}")
                print(f"  基线性能: {baseline_performance:.3f}")
                print(f"  性能比率: {recent_performance/baseline_performance:.3f}")
                return True
        
        return False
    
    def _apply_degradation_recovery(self, phase_idx):
        """应用训练退化恢复策略"""
        if not hasattr(self, 'phase_models') or phase_idx - 1 not in self.phase_models:
            print("[Recovery] 无前一阶段模型，无法恢复")
            return
            
        print("[Recovery] 开始应用训练退化恢复...")
        
        # 策略1: 降低学习率
        current_lr = self.player1_ai.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.5
        for param_group in self.player1_ai.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"[Recovery] 学习率从 {current_lr} 降低到 {new_lr}")
        
        # 策略2: 增加epsilon，重新探索
        self.player1_ai.epsilon = min(self.player1_ai.epsilon * 1.5, 0.6)
        self.player2_ai.epsilon = self.player1_ai.epsilon
        print(f"[Recovery] Epsilon 提升到 {self.player1_ai.epsilon:.3f}")
        
        # 策略3: 清理部分经验池，减少负样本影响
        if hasattr(self.player1_ai, 'replay_buffer') and len(self.player1_ai.replay_buffer) > 5000:
            # 保留最近30%的经验
            keep_size = len(self.player1_ai.replay_buffer) * 3 // 10
            self.player1_ai.replay_buffer = self.player1_ai.replay_buffer[-keep_size:]
            print(f"[Recovery] 经验池大小缩减到 {keep_size}")
        
        print("[Recovery] 恢复策略已应用")





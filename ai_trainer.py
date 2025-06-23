import random
import numpy as np
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard
import os
import datetime
import multiprocessing as mp
from parallel_sampler import worker_process

class AITrainer:
    def __init__(self, model_path=None):
        # 默认模型保存目录
        self.base_model_dir = "./models"
        if not os.path.exists(self.base_model_dir):
            os.makedirs(self.base_model_dir)
        # 训练唯一前缀：年月日_时分秒
        self.run_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 每次训练单独子目录
        self.model_dir = os.path.join(self.base_model_dir, self.run_prefix)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_path = model_path # 仅用于兼容接口，实际训练时动态生成
        self.env = HiveEnv(training_mode=True)  # 明确指定训练模式
        self.player1_ai = AIPlayer("AI_Player1", is_first_player=True, epsilon=1.0)
        self.player2_ai = AIPlayer("AI_Player2", is_first_player=False, epsilon=1.0)

    def train(self, num_episodes=1000, epsilon_decay=0.995, min_epsilon=0.01, batch_size=24, num_workers=8):
        """
        num_workers: 采样进程数，推荐高性能CPU+GPU可设为8-12（如13900HX+4070laptop建议8或更高），低配建议2-4。
        batch_size: 可根据显存适当提升（如32/64）。
        """
        print("Starting AI training...")
        average_rewards = []  # 新增：用于统计每局reward
        # 终局统计
        end_stats = {'win': 0, 'lose': 0, 'draw': 0, 'max_turns': 0, 'other': 0}
        end_stats_history = []
        # 新增：每局步数、非法动作数、主动落蜂后步数统计
        episode_steps_history = []
        illegal_action_count_history = []
        queenbee_step_history = []
        # 多进程采样worker支持
        workers = []
        queue = None
        if num_workers and num_workers > 0:
            import multiprocessing as mp
            from parallel_sampler import worker_process
            queue = mp.Queue(maxsize=32)
            player_args = dict(name='AI_Parallel', is_first_player=True)
            env_args = dict(training_mode=True)
            workers = [mp.Process(target=worker_process, args=(queue, player_args, env_args)) for _ in range(num_workers)]
            for w in workers:
                w.daemon = True
                w.start()
        try:
            for episode in range(num_episodes):
                observation, info = self.env.reset()
                terminated = False
                truncated = False
                episode_reward = 0
                episode_steps = 0
                illegal_action_count = 0
                queenbee_step = -1  # -1表示未主动落蜂后

                while not terminated and not truncated:
                    current_player_idx = self.env.current_player_idx
                    current_player_obj = self.player1_ai if current_player_idx == 0 else self.player2_ai
                    # Select action using the AI player
                    action = current_player_obj.select_action(self.env, self.env.game, self.env.board, current_player_idx)
                    if action is None:
                        # ---保险增强：详细调试输出---
                        print(f"\033[91m[DEBUG] 无合法动作！\033[0m 当前玩家: {current_player_obj.name} (idx={current_player_idx})，回合: {getattr(self.env, 'turn_count', '?')}")
                        print(f"[DEBUG] 玩家手牌: {getattr(current_player_obj, 'piece_count', {})}")
                        print(f"[DEBUG] 棋盘状态:")
                        self.env.board.display_board()
                        legal_actions = self.env.get_legal_actions() if hasattr(self.env, 'get_legal_actions') else []
                        print(f"[DEBUG] 环境get_legal_actions()返回: {legal_actions}")
                        # 若蜂后未落，尝试强制生成一个放蜂后动作
                        if hasattr(current_player_obj, 'is_queen_bee_placed') and not current_player_obj.is_queen_bee_placed:
                            from hive_env import Action
                            from utils import PieceType, BOARD_SIZE
                            placed = False
                            for x in range(BOARD_SIZE):
                                for y in range(BOARD_SIZE):
                                    if self.env.board.get_piece_at(x, y) is None:
                                        action = Action.encode_place_action(x, y, int(PieceType.QUEEN_BEE))
                                        print(f"[DEBUG] 保险兜底：强制生成放蜂后动作 action={action} at ({x},{y})")
                                        placed = True
                                        break
                                if placed:
                                    break
                            if not placed:
                                print("[DEBUG] 保险兜底失败：找不到可放蜂后的位置。")
                                reward = -5.0
                                terminated = True
                                truncated = False
                                next_observation = observation
                                info = {'reason': 'no_legal_action'}
                                current_player_obj.add_experience(observation, action, reward, next_observation, terminated)
                                break
                        else:
                            reward = -5.0
                            terminated = True
                            truncated = False
                            next_observation = observation
                            info = {'reason': 'no_legal_action'}
                            current_player_obj.add_experience(observation, action, reward, next_observation, terminated)
                            break
                    try:
                        next_observation, reward, terminated, truncated, info = self.env.step(action)
                    except Exception as e:
                        print(f"[ERROR] step异常: {e}")
                        reward = -2.0
                        terminated = True
                        truncated = False
                        next_observation = observation
                        info = {'reason': str(e)}
                    # 统计步数
                    episode_steps += 1
                    # 统计非法动作
                    if reward <= -2.0:
                        illegal_action_count += 1
                    # 统计主动落蜂后步数（首次place QueenBee）
                    if queenbee_step == -1:
                        action_type, _, _, _, _, piece_type_id = self.env._decode_action(action)
                        if action_type == 'place' and piece_type_id == 0:
                            queenbee_step = episode_steps
                    episode_reward += reward
                    current_player_obj.add_experience(observation, action, reward, next_observation, terminated)
                    current_player_obj.train_on_batch(batch_size)
                    observation = next_observation
                # 终局统计
                reason = info.get('reason', '')
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
                # epsilon衰减：前200局不变，之后每局*0.999（更平滑）
                if episode >= 200:
                    self.player1_ai.epsilon = max(min_epsilon, self.player1_ai.epsilon * 0.999)
                    self.player2_ai.epsilon = max(min_epsilon, self.player2_ai.epsilon * 0.999)
                average_rewards.append(episode_reward)
                episode_steps_history.append(episode_steps)
                illegal_action_count_history.append(illegal_action_count)
                queenbee_step_history.append(queenbee_step)
                # 每集采集一次终局统计
                end_stats_history.append(end_stats.copy())
                end_stats = {k: 0 for k in end_stats}
                print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.player1_ai.epsilon:.4f}")
                # 每100局输出最近100局平均reward和终局统计
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(average_rewards[-100:])
                    print(f"\033[93m[统计] 最近100局平均Reward: {avg_reward:.2f}\033[0m")
                    print(f"[统计] 最近100局终局 win={sum(d['win'] for d in end_stats_history[-100:])} lose={sum(d['lose'] for d in end_stats_history[-100:])} draw={sum(d['draw'] for d in end_stats_history[-100:])} max_turns={sum(d['max_turns'] for d in end_stats_history[-100:])} other={sum(d['other'] for d in end_stats_history[-100:])}")
                    avg_steps = np.mean(episode_steps_history[-100:])
                    avg_illegal = np.mean(illegal_action_count_history[-100:])
                    avg_queenbee = np.mean([q for q in queenbee_step_history[-100:] if q > 0]) if any(q > 0 for q in queenbee_step_history[-100:]) else -1
                    print(f"[统计] 最近100局平均步数: {avg_steps:.2f}，平均非法动作数: {avg_illegal:.2f}，平均主动落蜂后步数: {avg_queenbee if avg_queenbee>0 else 'N/A'}")
                    end_stats_history.append(end_stats.copy())
                    end_stats = {k: 0 for k in end_stats}
        except KeyboardInterrupt:
            print("\n[INFO] 检测到Ctrl+C，正在保存模型和reward曲线...")
        finally:
            if workers:
                for w in workers:
                    w.terminate()
                for w in workers:
                    w.join()
            reward_file = os.path.join(self.model_dir, f"{self.run_prefix}_reward_history.npy")
            np.save(reward_file, np.array(average_rewards))
            print(f"Reward history saved to {reward_file}")
            # 保存终局统计
            end_stats_file = os.path.join(self.model_dir, f"{self.run_prefix}_end_stats_history.npy")
            np.save(end_stats_file, np.array(end_stats_history))
            print(f"End stats history saved to {end_stats_file}")
            # 保存步数、非法动作、主动落蜂后步数
            np.save(os.path.join(self.model_dir, f"{self.run_prefix}_steps_history.npy"), np.array(episode_steps_history))
            np.save(os.path.join(self.model_dir, f"{self.run_prefix}_illegal_history.npy"), np.array(illegal_action_count_history))
            np.save(os.path.join(self.model_dir, f"{self.run_prefix}_queenbee_step_history.npy"), np.array(queenbee_step_history))
            print("[统计] 步数、非法动作、主动落蜂后步数历史已保存。")
            final_model_file = os.path.join(self.model_dir, f"{self.run_prefix}_final.npz")
            self.player1_ai.neural_network.save_model(final_model_file)
            print(f"Final model saved to {final_model_file}")
            print("AI training finished (正常结束或中断)。模型和reward曲线已保存。")

    def load_model(self):
        self.player1_ai.neural_network.load_model(self.model_path)
        self.player2_ai.neural_network.load_model(self.model_path) # Both players use the same model





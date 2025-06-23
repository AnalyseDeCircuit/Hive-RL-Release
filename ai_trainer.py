import random
import numpy as np
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard
import os
import datetime

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

    def train(self, num_episodes=1000, epsilon_decay=0.995, min_epsilon=0.01, batch_size=32):
        print("Starting AI training...")
        average_rewards = []  # 新增：用于统计每局reward
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            terminated = False
            truncated = False
            episode_reward = 0

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
                            reward = -2.5
                            terminated = True
                            truncated = False
                            next_observation = observation
                            info = {'reason': 'no_legal_action'}
                            current_player_obj.add_experience(observation, action, reward, next_observation, terminated)
                            break
                    else:
                        reward = -2.5
                        terminated = True
                        truncated = False
                        next_observation = observation
                        info = {'reason': 'no_legal_action'}
                        current_player_obj.add_experience(observation, action, reward, next_observation, terminated)
                        break

                # step保险：捕获非法动作异常，直接惩罚并终止，不自动兜底
                try:
                    next_observation, reward, terminated, truncated, info = self.env.step(action)
                except Exception as e:
                    print(f"Error during step: {e}")
                    reward = -1.0
                    terminated = True
                    truncated = False
                    next_observation = observation
                    info = {'reason': str(e)}
                episode_reward += reward

                # 调试输出每步 reward
                print(f"Step: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}")

                # Add experience to replay buffer for both players (if applicable, for self-play)
                # For self-play, both agents learn from the same experiences, but from their own perspective.
                # The report implies a single agent learning, so we'll add experience to the current player's buffer.
                current_player_obj.add_experience(observation, action, reward, next_observation, terminated)

                # Train the AI player
                current_player_obj.train_on_batch(batch_size)

                observation = next_observation

            # Decay epsilon
            self.player1_ai.epsilon = max(min_epsilon, self.player1_ai.epsilon * epsilon_decay)
            self.player2_ai.epsilon = max(min_epsilon, self.player2_ai.epsilon * epsilon_decay)

            # 动态调整epsilon，训练后期逐步降低探索率
            if episode % 100 == 0 and self.player1_ai.epsilon > 0.1:
                self.player1_ai.epsilon *= 0.95
            if episode % 100 == 0 and self.player2_ai.epsilon > 0.1:
                self.player2_ai.epsilon *= 0.95

            average_rewards.append(episode_reward)
            # 每100局输出最近100局平均reward
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(average_rewards[-100:])
                print(f"\033[93m[统计] 最近100局平均Reward: {avg_reward:.2f}\033[0m")

            print(f"\033[96mEpisode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {self.player1_ai.epsilon:.2f}\033[0m")

            # Save model periodically
            if (episode + 1) % 100 == 0:
                model_file = os.path.join(self.model_dir, f"{self.run_prefix}_ep{episode+1}.npz")
                self.player1_ai.neural_network.save_model(model_file)
                print(f"Model saved after {episode + 1} episodes: {model_file}")
        # 训练结束后保存reward曲线
        reward_file = os.path.join(self.model_dir, f"{self.run_prefix}_reward_history.npy")
        np.save(reward_file, np.array(average_rewards))
        print(f"AI training finished. Reward history saved to {reward_file}")
        # 最终模型
        final_model_file = os.path.join(self.model_dir, f"{self.run_prefix}_final.npz")
        self.player1_ai.neural_network.save_model(final_model_file)
        print(f"Final model saved to {final_model_file}")

    def load_model(self):
        self.player1_ai.neural_network.load_model(self.model_path)
        self.player2_ai.neural_network.load_model(self.model_path) # Both players use the same model





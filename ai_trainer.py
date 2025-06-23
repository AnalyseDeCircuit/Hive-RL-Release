import random
import numpy as np
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard

class AITrainer:
    def __init__(self, model_path="./ai_model.npz"):
        self.env = HiveEnv(training_mode=True)  # 明确指定训练模式
        self.player1_ai = AIPlayer("AI_Player1", is_first_player=True, epsilon=1.0) # Start with high exploration
        self.player2_ai = AIPlayer("AI_Player2", is_first_player=False, epsilon=1.0) # Start with high exploration
        self.model_path = model_path

    def train(self, num_episodes=1000, epsilon_decay=0.995, min_epsilon=0.01, batch_size=32):
        print("Starting AI training...")
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
                if action is None: # No legal actions, game might be stuck
                    # 无合法动作，给予更强惩罚
                    reward = -2.5
                    terminated = True
                    truncated = False
                    next_observation = observation
                    info = {'reason': 'no_legal_action'}
                    print(f"No legal actions for player {current_player_idx}, terminating episode with penalty.")
                    # 记录经验
                    current_player_obj.add_experience(observation, action, reward, next_observation, terminated)
                    break

                next_observation, reward, terminated, truncated, info = self.env.step(action)
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

            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {self.player1_ai.epsilon:.2f}")

            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.player1_ai.neural_network.save_model(self.model_path)
                print(f"Model saved after {episode + 1} episodes.")

        print("AI training finished.")
        self.player1_ai.neural_network.save_model(self.model_path) # Save final model

    def load_model(self):
        self.player1_ai.neural_network.load_model(self.model_path)
        self.player2_ai.neural_network.load_model(self.model_path) # Both players use the same model





import random
from hive_env import HiveEnv
from ai_player import AIPlayer
from game import Game
from board import ChessBoard
from player import Player

class RandomPlayer(Player):
    def select_action(self, env, game_state, board, current_player_idx):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)

class AIEvaluator:
    def __init__(self, use_dlc=False, model_path="./ai_model.npz"):
        self.use_dlc = use_dlc  # 是否启用DLC棋子
        # 初始化环境和玩家
        self.env = HiveEnv(use_dlc=self.use_dlc)
        self.model_path = model_path
        self.ai_player = AIPlayer("Evaluated_AI", is_first_player=True, epsilon=0.0, use_dlc=self.use_dlc)  # No exploration during evaluation
        self.random_player = RandomPlayer("Random_AI", is_first_player=False, use_dlc=self.use_dlc)  # Always explore (random actions)

    def evaluate(self, num_games=100):
        print(f"Starting AI evaluation for {num_games} games...")
        try:
            self.ai_player.neural_network.load_model(self.model_path)
        except FileNotFoundError:
            print(f"Error: Model not found at {self.model_path}. Please train the AI first.")
            return

        ai_wins = 0
        random_wins = 0
        draws = 0

        for game_num in range(num_games):
            observation, info = self.env.reset()
            terminated = False
            truncated = False
            
            # Randomly decide who goes first for fair evaluation
            if random.random() < 0.5:
                self.env.current_player_idx = 0 # AI goes first
                current_ai = self.ai_player
                current_opponent = self.random_player
            else:
                self.env.current_player_idx = 1 # Random goes first
                current_ai = self.random_player
                current_opponent = self.ai_player

            self.env.game.player1 = self.ai_player if self.env.current_player_idx == 0 else self.random_player
            self.env.game.player2 = self.random_player if self.env.current_player_idx == 0 else self.ai_player

            while not terminated and not truncated:
                player_to_move = self.env.game.current_player
                
                if player_to_move == self.ai_player:
                    action = self.ai_player.select_action(None, self.env.game, self.env.board, self.env.current_player_idx)
                else:
                    # 修正：补齐AIPlayer.select_action参数
                    action = self.random_player.select_action(None, self.env.game, self.env.board, self.env.current_player_idx)

                if action is None: # No legal actions, game might be stuck
                    terminated = True
                    break

                next_observation, reward, terminated, truncated, info = self.env.step(action)
                observation = next_observation

            game_status = self.env.game.check_game_over()
            if game_status == 1: # Player 1 wins
                if self.env.game.player1 == self.ai_player:
                    ai_wins += 1
                else:
                    random_wins += 1
            elif game_status == 2: # Player 2 wins
                if self.env.game.player2 == self.ai_player:
                    ai_wins += 1
                else:
                    random_wins += 1
            elif game_status == 3: # Draw
                draws += 1
            
            print(f"Game {game_num + 1}/{num_games} finished. AI Wins: {ai_wins}, Random Wins: {random_wins}, Draws: {draws}")

        total_games = ai_wins + random_wins + draws
        if total_games > 0:
            ai_win_rate = (ai_wins / total_games) * 100
            print(f"\nEvaluation Complete: ")
            print(f"Total Games: {total_games}")
            print(f"AI Wins: {ai_wins} ({ai_win_rate:.2f}%)")
            print(f"Random Wins: {random_wins}")
            print(f"Draws: {draws}")
        else:
            print("No games were played during evaluation.")



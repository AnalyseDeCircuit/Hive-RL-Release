# 问题：环境 get_legal_actions 生成 place 动作时未跳过已被占用格子，导致AI可能生成重复落子动作，训练时会出现“Cannot place piece here. Position is occupied.”报错。
# 修复：已在 get_legal_actions 生成 place 动作时增加 if self.board.get_piece_at(x, y) is not None: continue，彻底杜绝重复落子。
# 问题：环境 get_legal_actions 在 must_place_queen（第4回合强制落蜂）时依然允许生成 move 动作，导致AI可能选到非法move，训练时会出现“QueenBee must be placed before moving other pieces.”报错。
# 修复：已同步AIPlayer逻辑，只有蜂后已落且不是must_place_queen时才生成move动作。
# 影响：环境与AI动作生成规则完全一致，训练时不会再出现非法move或重复落子问题。
# 日期：2025-06-23
# 作者：Copilot

import random
import numpy as np
from player import Player
from game import Game
from board import ChessBoard
from piece import PieceType
from hive_env import Action # Re-use the Action helper from hive_env
from neural_network import NeuralNetwork # Import the NeuralNetwork class
from utils import BOARD_SIZE, PIECE_TYPE_LIST, PIECE_TYPE_ID_MAP, PIECE_TYPE_NAME_LIST

class AIPlayer(Player):
    def __init__(self, name, is_first_player, is_ai=True, epsilon=0.6, learning_rate=0.01, discount_factor=0.99, use_dlc=False):
        super().__init__(name, is_first_player, is_ai, use_dlc)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initialize Neural Network (input_dim=814, hidden_dim=256, output_dim=1 for state value estimation)
        self.neural_network = NeuralNetwork(input_dim=814, hidden_dim=256, output_dim=1)
        
        # Experience Replay Buffer
        self.replay_buffer = []
        self.max_buffer_size = 10000 # As per report

    def select_action(self, env, game_state: Game, board: ChessBoard, current_player_idx: int):
        current_state_vector = self._get_observation_from_game_state(game_state, board, current_player_idx)
        if env is not None:
            legal_actions = env.get_legal_actions()
        else:
            legal_actions = self._get_legal_actions_from_game_state(game_state, board, current_player_idx)

        if not legal_actions:
            return None # No legal actions, game might be over or stuck

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            # Explore: choose a random legal action
            action = random.choice(legal_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            best_actions = []
            max_q_value = -float("inf")

            for action_candidate in legal_actions:
                # 用局部变量模拟，不污染全局Game单例
                sim_player1 = game_state.player1.clone()
                sim_player2 = game_state.player2.clone()
                sim_board = board.clone()
                sim_turn_count = game_state.turn_count

                simulated_reward = 0.0 # Immediate reward for simulation
                try:
                    action_type, from_x, from_y, to_x, to_y, piece_type_id = Action.decode_action(action_candidate)
                    if current_player_idx == 0:
                        sim_current = sim_player1
                    else:
                        sim_current = sim_player2
                    # Apply action to sim_player
                    if action_type == 'place':
                        piece_type_name = PIECE_TYPE_NAME_LIST[piece_type_id] if isinstance(piece_type_id, int) and 0 <= piece_type_id < len(PIECE_TYPE_NAME_LIST) else str(piece_type_id)
                        piece_type = getattr(PieceType, piece_type_name)
                        if sim_current.piece_count.get(piece_type, 0) > 0 and \
                           sim_board.is_valid_placement(to_x, to_y, sim_current.is_first_player, sim_turn_count):
                            sim_current.place_piece(sim_board, to_x, to_y, piece_type, sim_turn_count)
                            simulated_reward = 0.1
                        else:
                            simulated_reward = -0.5
                    elif action_type == 'move':
                        piece_to_move = sim_board.get_piece_at(from_x, from_y)
                        if piece_to_move and piece_to_move.owner == sim_current and \
                           piece_to_move.is_valid_move(sim_board, to_x, to_y):
                            sim_current.move_piece(sim_board, from_x, from_y, to_x, to_y, piece_to_move.piece_type)
                            simulated_reward = 0.1
                        else:
                            simulated_reward = -0.5
                    # 检查胜负
                    # 这里只能简单模拟，若需完整胜负判定可补充
                except Exception:
                    simulated_reward = -1.0

                # 生成下一个状态特征
                next_state_vector = self._get_observation_from_game_state(game_state, sim_board, 1 - current_player_idx)
                q_value = simulated_reward + self.discount_factor * self.neural_network.forward(next_state_vector)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_actions = [action_candidate]
                elif q_value == max_q_value:
                    best_actions.append(action_candidate)
            if not best_actions:
                return None  # 没有可选动作，返回None，防止崩溃
            action = random.choice(best_actions)
        # ---保险：蜂后未落时禁止move动作（无论env返回什么）---
        current_player = game_state.player1 if current_player_idx == 0 else game_state.player2
        if not getattr(current_player, 'is_queen_bee_placed', False):
            # 只保留place动作
            legal_actions = [a for a in legal_actions if Action.decode_action(a)[0] == 'place']
            if not legal_actions:
                return None
            # 如果action不是place，强制随机选一个place
            if action is not None:
                action_type, *_ = Action.decode_action(action)
                if action_type != 'place':
                    action = random.choice(legal_actions)
        return action

    def _get_observation_from_game_state(self, game_state: Game, board: ChessBoard, current_player_idx: int):
        # This method duplicates logic from HiveEnv._get_observation
        # It's a temporary solution until GameState class is fully integrated.
        
        # 1. Board state (800 dimensions)
        board_encoding = np.zeros((BOARD_SIZE, BOARD_SIZE, len(PIECE_TYPE_LIST)), dtype=np.float32)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                pieces_at_pos = board.get_pieces_at(x, y)
                if pieces_at_pos:
                    top_piece = pieces_at_pos[-1] # Get the top piece
                    piece_type_val = top_piece.piece_type
                    if not isinstance(piece_type_val, int):
                        piece_type_val = int(getattr(PieceType, str(piece_type_val), 0))
                    piece_type_id = piece_type_val if piece_type_val in PIECE_TYPE_LIST else 0
                    board_encoding[x, y, piece_type_id] = 1.0
        board_encoding = board_encoding.flatten()

        # 2. Player hand information (10 dimensions)
        player1_hand_encoding = np.zeros(5, dtype=np.float32)
        player2_hand_encoding = np.zeros(5, dtype=np.float32)

        piece_type_map = {
            0: 0, # QUEEN_BEE
            1: 1, # BEETLE
            2: 2, # SPIDER
            3: 3, # ANT
            4: 4  # GRASSHOPPER
        }
        max_counts = {
            0: 1, # QUEEN_BEE
            1: 2, # BEETLE
            2: 2, # SPIDER
            3: 3, # ANT
            4: 3  # GRASSHOPPER
        }
        # 防御性检查，防止player1/player2为None
        if game_state.player1 is not None:
            for piece_type, count in game_state.player1.piece_count.items():
                piece_type_id = piece_type if isinstance(piece_type, int) else int(getattr(PieceType, str(piece_type), 0))
                if piece_type_id in piece_type_map:
                    player1_hand_encoding[piece_type_map[piece_type_id]] = count / max_counts[piece_type_id]
        if game_state.player2 is not None:
            for piece_type, count in game_state.player2.piece_count.items():
                piece_type_id = piece_type if isinstance(piece_type, int) else int(getattr(PieceType, str(piece_type), 0))
                if piece_type_id in piece_type_map:
                    player2_hand_encoding[piece_type_map[piece_type_id]] = count / max_counts[piece_type_id]

        # 3. Game state information (4 dimensions)
        current_player_encoding = np.array([current_player_idx], dtype=np.float32)
        turn_count_encoding = np.array([game_state.turn_count / 50.0], dtype=np.float32) # Normalized by 50 as per report
        player1_queen_placed_encoding = np.array([1.0 if (game_state.player1 is not None and getattr(game_state.player1, 'is_queen_bee_placed', False)) else 0.0], dtype=np.float32)
        player2_queen_placed_encoding = np.array([1.0 if (game_state.player2 is not None and getattr(game_state.player2, 'is_queen_bee_placed', False)) else 0.0], dtype=np.float32)

        observation = np.concatenate([
            board_encoding,
            player1_hand_encoding,
            player2_hand_encoding,
            current_player_encoding,
            turn_count_encoding,
            player1_queen_placed_encoding,
            player2_queen_placed_encoding
        ])
        return observation

    def _get_legal_actions_from_game_state(self, game_state: Game, board: ChessBoard, current_player_idx: int):
        # This method duplicates logic from HiveEnv.get_legal_actions
        # It's a temporary solution until GameState class is fully integrated.
        legal_actions = []
        current_player = game_state.player1 if current_player_idx == 0 else game_state.player2
        if current_player is None:
            return legal_actions
        # 判断是否必须本回合放蜂后
        must_place_queen = (game_state.turn_count == 3 and not getattr(current_player, 'is_queen_bee_placed', False))
        # Generate place actions
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board.get_piece_at(x, y) is not None:
                    continue
                for piece_type_id in PIECE_TYPE_LIST:
                    count = current_player.piece_count.get(piece_type_id, 0)
                    if count == 0:
                        # 兼容key为PieceType对象的情况
                        from utils import PieceType, PIECE_TYPE_NAME_LIST
                        piece_type_obj = getattr(PieceType, PIECE_TYPE_NAME_LIST[piece_type_id])
                        count = current_player.piece_count.get(piece_type_obj, 0)
                    if count > 0:
                        if board.is_valid_placement(x, y, current_player.is_first_player, game_state.turn_count):
                            if must_place_queen and piece_type_id != 0:
                                continue
                            action_int = Action.encode_place_action(x, y, piece_type_id)
                            legal_actions.append(action_int)
        # 只有蜂后已落时才允许生成移动动作（无论回合数）
        if getattr(current_player, 'is_queen_bee_placed', False):
            # 但第4回合强制放蜂后时依然不能移动
            if not must_place_queen:
                for from_x in range(BOARD_SIZE):
                    for from_y in range(BOARD_SIZE):
                        piece_to_move = board.get_piece_at(from_x, from_y)
                        if piece_to_move and piece_to_move.owner == current_player:
                            for to_x in range(BOARD_SIZE):
                                for to_y in range(BOARD_SIZE):
                                    if piece_to_move.is_valid_move(board, to_x, to_y):
                                        action_int = Action.encode_move_action(from_x, from_y, to_x, to_y)
                                        legal_actions.append(action_int)
        # debug输出所有生成的落子动作
        # print(f"[DEBUG] legal_actions: {legal_actions}")
        return legal_actions

    def add_experience(self, state, action, reward, next_state, terminated):
        # Add experience to replay buffer
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0) # Remove oldest experience
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    def train_on_batch(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return # Not enough experiences to train

        batch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, next_state, terminated in batch:
            target_value = reward
            if not terminated:
                # Predict value of next state
                next_state_value = self.neural_network.forward(next_state)
                target_value += self.discount_factor * next_state_value
            
            # Train the neural network
            self.neural_network.train(state, target_value, self.learning_rate)

    def clone(self):
        # 返回AIPlayer实例，保留所有AI参数和piece_count，深拷贝 queen_bee_position，避免引用污染
        cloned_player = AIPlayer(
            self.name,
            self.is_first_player,
            is_ai=self.is_ai,
            epsilon=self.epsilon,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            use_dlc=(len(self.piece_count) > 5)
        )
        cloned_player.piece_count = self.piece_count.copy()
        cloned_player.is_queen_bee_placed = self.is_queen_bee_placed
        # 深拷贝 queen_bee_position，避免引用污染
        if self.queen_bee_position is not None:
            if isinstance(self.queen_bee_position, (tuple, list)):
                cloned_player.queen_bee_position = tuple(self.queen_bee_position)
            else:
                cloned_player.queen_bee_position = self.queen_bee_position
        else:
            cloned_player.queen_bee_position = None
        # 神经网络等可选属性如需深拷贝可补充
        return cloned_player




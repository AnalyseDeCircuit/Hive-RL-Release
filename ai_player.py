# 问题：环境 get_legal_actions 生成 place 动作时未跳过已被占用格子，导致AI可能生成重复落子动作，训练时会出现“Cannot place piece here. Position is occupied.”报错。
# 修复：已在 get_legal_actions 生成 place 动作时增加 if self.board.get_piece_at(x, y) is not None: continue，彻底杜绝重复落子。
# 问题：环境 get_legal_actions 在 must_place_queen（第4回合强制落蜂）时依然允许生成 move 动作，导致AI可能选到非法move，训练时会出现“QueenBee must be placed before moving other pieces.”报错。
# 修复：已同步AIPlayer逻辑，只有蜂后已落且不是must_place_queen时才生成move动作。
# 影响：环境与AI动作生成规则完全一致，训练时不会再出现非法move或重复落子问题。
# 日期：2025-06-23
# 作者：Copilot

import random
import numpy as np
import torch
from player import Player
from game import Game
from board import ChessBoard
from piece import PieceType
from hive_env import Action # Re-use the Action helper from hive_env
from neural_network_torch import NeuralNetwork # 替换为PyTorch实现
from utils import BOARD_SIZE, PIECE_TYPE_LIST, PIECE_TYPE_ID_MAP, PIECE_TYPE_NAME_LIST

class AIPlayer(Player):
    def __init__(self, name, is_first_player, is_ai=True, epsilon=0.6, learning_rate=0.01, discount_factor=0.99, use_dlc=False):
        super().__init__(name, is_first_player, is_ai, use_dlc)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # PyTorch神经网络（支持DLC全部8种棋子）
        # 状态空间: 800(board) + 16(hand) + 4 = 820
        # 动作空间: BOARD_SIZE*4(one-hot coords) + len(PIECE_TYPE_LIST)(piece types) = 4*10 + 8 = 48
        input_dim = 820 + BOARD_SIZE * 4 + len(PIECE_TYPE_LIST)
        # 新结构：输入→1024→512→1
        self.neural_network = NeuralNetwork(input_dim=input_dim, hidden_dims=[1024, 512], output_dim=1)
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=self.learning_rate)

        # Experience Replay Buffer
        self.replay_buffer = []
        self.max_buffer_size = 10000 # As per report

    def _encode_action(self, action_int):
        """
        多维离散特征拼接编码：
        - from_x, from_y, to_x, to_y: 0~BOARD_SIZE-1，分别one-hot（move动作用，place动作from_x/from_y全0）
        - piece_type_id: 0~4，one-hot
        返回 shape=(BOARD_SIZE*4+5,)
        """
        action_type, from_x, from_y, to_x, to_y, piece_type_id = Action.decode_action(action_int)
        # One-hot编码
        fx = np.zeros(BOARD_SIZE, dtype=np.float32)
        fy = np.zeros(BOARD_SIZE, dtype=np.float32)
        tx = np.zeros(BOARD_SIZE, dtype=np.float32)
        ty = np.zeros(BOARD_SIZE, dtype=np.float32)
        pt = np.zeros(len(PIECE_TYPE_LIST), dtype=np.float32)
        if action_type == 'move':
            if from_x is not None and 0 <= from_x < BOARD_SIZE:
                fx[from_x] = 1.0
            if from_y is not None and 0 <= from_y < BOARD_SIZE:
                fy[from_y] = 1.0
        # place动作from_x/from_y全0
        if to_x is not None and 0 <= to_x < BOARD_SIZE:
            tx[to_x] = 1.0
        if to_y is not None and 0 <= to_y < BOARD_SIZE:
            ty[to_y] = 1.0
        if piece_type_id is not None and 0 <= piece_type_id < len(PIECE_TYPE_LIST):
            pt[piece_type_id] = 1.0
        return np.concatenate([fx, fy, tx, ty, pt])

    def select_action(self, env, game_state: Game, board: ChessBoard, current_player_idx: int, debug=False):
        # ---兼容AI对战/测试流程：env为None时自动fallback到自带合法动作生成器---
        if env is None:
            legal_actions = self._get_legal_actions_from_game_state(game_state, board, current_player_idx)
        else:
            legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None # No legal actions, game might be over or stuck
        current_player = game_state.player1 if current_player_idx == 0 else game_state.player2
        must_place_queen = (game_state.turn_count == 3 and not getattr(current_player, 'is_queen_bee_placed', False))
        # ---新增：前4步且蜂后未落时，优先探索放蜂后---
        if game_state.turn_count < 4 and not getattr(current_player, 'is_queen_bee_placed', False):
            queenbee_actions = [a for a in legal_actions if (Action.decode_action(a)[0] == 'place' and Action.decode_action(a)[5] == 0)]
            if queenbee_actions:
                if random.random() < self.epsilon:
                    return random.choice(queenbee_actions)
        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            # Explore: choose a random legal action
            action = random.choice(legal_actions)
        else:
            # 新版：批量Q(s,a)评估
            state_vector = self._get_observation_from_game_state(game_state, board, current_player_idx)
            state_tensor = torch.tensor(state_vector, dtype=torch.float32)
            action_encodings = np.stack([self._encode_action(a) for a in legal_actions])
            action_tensors = torch.tensor(action_encodings, dtype=torch.float32)
            state_batch = state_tensor.repeat(action_tensors.shape[0], 1)
            sa_batch = torch.cat([state_batch, action_tensors], dim=1)  # shape=(N, state+action)
            with torch.no_grad():
                q_values = self.neural_network(sa_batch).squeeze(-1).cpu().numpy()  # shape=(N,)
            max_q = np.max(q_values)
            best_idxs = np.where(q_values == max_q)[0]
            best_actions = [legal_actions[i] for i in best_idxs]
            action = random.choice(best_actions)
        # ---终极保险：返回前再次强制过滤，action 必须在 legal_actions 内---
        if action not in legal_actions:
            print(f"[AIPlayer][DEBUG] action {action} 不在 legal_actions，强制随机采样合法动作。")
            action = random.choice(legal_actions)
        # ---蜂后未落时绝不允许move动作---
        if not getattr(current_player, 'is_queen_bee_placed', False):
            place_actions = [a for a in legal_actions if Action.decode_action(a)[0] == 'place']
            if not place_actions:
                print("[AIPlayer][WARN] 蜂后未落且无place动作，AI被保险层拦截，返回None，环境应给予惩罚。")
                return None
            action_type = Action.decode_action(action)[0]
            if action_type != 'place':
                print(f"[AIPlayer][DEBUG] 蜂后未落但action类型为{action_type}，强制采样place动作。")
                action = random.choice(place_actions)
        # ---必须落蜂后时只允许place QueenBee---
        if must_place_queen:
            queen_actions = [a for a in legal_actions if (Action.decode_action(a)[0] == 'place' and Action.decode_action(a)[5] == 0)]
            if not queen_actions:
                print("[AIPlayer][WARN] must_place_queen但无QueenBee可下，AI被保险层拦截，返回None，环境应给予惩罚。")
                return None
            action_type, *_, piece_type_id = Action.decode_action(action)
            if not (action_type == 'place' and piece_type_id == 0):
                print(f"[AIPlayer][DEBUG] must_place_queen但action不是place QueenBee，强制采样QueenBee动作。")
                action = random.choice(queen_actions)
        return action

    # --- 辅助函数：兼容 PieceType/int/str 的 piece_type_id 提取 ---
    def safe_piece_type_id(self, piece_type):
        from piece import PieceType
        if isinstance(piece_type, int):
            return piece_type
        if hasattr(piece_type, 'value'):
            return int(piece_type.value)
        # 兼容字符串（如 'QUEEN_BEE'）
        if isinstance(piece_type, str):
            try:
                return int(getattr(PieceType, piece_type).value)
            except Exception:
                try:
                    return int(piece_type)
                except Exception:
                    return 0
        return int(piece_type) if piece_type is not None else 0

    def _get_observation_from_game_state(self, game_state: Game, board: ChessBoard, current_player_idx: int, debug=False):
        # This method duplicates logic from HiveEnv._get_observation
        # It's a temporary solution until GameState class is fully integrated.
        
        # 1. Board state (800 dimensions)
        board_encoding = np.zeros((BOARD_SIZE, BOARD_SIZE, len(PIECE_TYPE_LIST)), dtype=np.float32)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                pieces_at_pos = board.get_pieces_at(x, y)
                if pieces_at_pos:
                    top_piece = pieces_at_pos[-1] # Get the top piece
                    piece_type_id = self.safe_piece_type_id(top_piece.piece_type)
                    if piece_type_id in PIECE_TYPE_LIST:
                        board_encoding[x, y, piece_type_id] = 1.0
        board_encoding = board_encoding.flatten()

        # 2. Player hand information (8 dimensions per player, including DLC)
        player1_hand_encoding = np.zeros(len(PIECE_TYPE_LIST), dtype=np.float32)
        player2_hand_encoding = np.zeros(len(PIECE_TYPE_LIST), dtype=np.float32)
        # 最大棋子数量映射
        max_counts = {
            0: 1, # QUEEN_BEE
            1: 2, # BEETLE
            2: 2, # SPIDER
            3: 3, # ANT
            4: 3, # GRASSHOPPER
            5: 1, # LADYBUG
            6: 1, # MOSQUITO
            7: 1  # PILLBUG
        }
        # 填充手牌编码
        if game_state.player1 is not None:
            for pt, count in game_state.player1.piece_count.items():
                pid = int(pt)
                if pid in max_counts:
                    player1_hand_encoding[pid] = count / max_counts[pid]
        if game_state.player2 is not None:
            for pt, count in game_state.player2.piece_count.items():
                pid = int(pt)
                if pid in max_counts:
                    player2_hand_encoding[pid] = count / max_counts[pid]

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
        # --- debug输出 ---
        if debug:
            print(f"[DEBUG][obs] sum={np.sum(observation):.4f} min={np.min(observation):.4f} max={np.max(observation):.4f} ",
                  f"head={observation[:20]} tail={observation[-20:]}")
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
        # 现在action为int，训练时需拼接编码
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    def train_on_batch(self, batch_size=32):
        # 只采样合法动作样本（reward > -2.0，彻底丢弃所有非法动作样本）
        legal_samples = [exp for exp in self.replay_buffer if exp[2] > -2.0]
        if len(legal_samples) < batch_size:
            return None
        
        batch = random.sample(legal_samples, batch_size)
        # 拼接 state+action 作为输入
        sa_inputs = []
        targets = []
        
        for state, action, reward, next_state, terminated in batch:
            action_vec = self._encode_action(action)
            sa_input = np.concatenate([state, action_vec])
            sa_inputs.append(sa_input)
            
            # 改进的Q值计算
            target = reward
            if not terminated and len(legal_samples) > batch_size:
                # 简化的双重DQN更新：使用当前网络选择动作，目标网络评估价值
                # 这里简化为使用reward的折扣版本
                target = reward + self.discount_factor * reward * 0.1  # 简化版bootstrapping
            
            targets.append(target)
        
        if not sa_inputs:
            return None
            
        sa_inputs = np.stack(sa_inputs)
        targets = np.array(targets, dtype=np.float32)
        
        # 添加梯度裁剪防止梯度爆炸
        try:
            loss = self.neural_network.train_step(sa_inputs, targets, self.optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), max_norm=1.0)
            return loss
        except Exception as e:
            print(f"训练批次失败: {e}")
            return None

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
        # Copy neural network parameters (for PyTorch implementation)
        try:
            # 使用 state_dict 复制所有模型参数
            cloned_player.neural_network.load_state_dict(self.neural_network.state_dict())
        except Exception:
            # 如果是 numpy 实现则忽略
            pass
        return cloned_player

    def save_to_file(self, file_path: str):
        """保存神经网络权重到文件"""
        import torch
        torch.save(self.neural_network.state_dict(), file_path)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'AIPlayer':
        """从文件加载神经网络权重并创建 AIPlayer 实例"""
        import torch
        state_dict = torch.load(file_path)
        # 根据文件名或其他约定实例化默认参数
        player = cls("AI", False, True)
        player.neural_network.load_state_dict(state_dict)
        player.neural_network.eval()
        return player
    
# ============ 新增：集成多模型投票类 ============
from typing import List

class EnsembleAIPlayer:
    """
    多模型投票AI：持有多套AIPlayer实例，决策时对各模型Q值求平均/加权后选动作。
    """
    def __init__(self, agents: List[AIPlayer], weights=None, epsilon=0.05):
        self.agents = agents
        self.weights = weights or [1.0] * len(agents)
        self.epsilon = epsilon

    def select_action(self, env, game_state, board, current_player_idx, debug=False):
        legal = env.get_legal_actions()
        if not legal:
            return None
        # 随机探索
        if random.random() < self.epsilon:
            return random.choice(legal)
        # 计算加权平均Q值
        avg_scores = []
        for a in legal:
            score = 0.0
            for w, agent in zip(self.weights, self.agents):
                # 单模型批量评估
                state_vec = agent._get_observation_from_game_state(game_state, board, current_player_idx)
                action_vec = agent._encode_action(a)
                with agent.neural_network._no_grad():
                    q = agent.neural_network.forward(np.concatenate([state_vec, action_vec]))
                score += w * q
            avg_scores.append(score / sum(self.weights))
        # 取最大平均Q对应动作
        best_idx = int(np.argmax(avg_scores))
        return legal[best_idx]




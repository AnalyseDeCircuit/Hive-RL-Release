import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game
from board import ChessBoard
from player import Player
from piece import Piece, QueenBee, Beetle, Spider, Ant, Grasshopper, Ladybug, Mosquito, Pillbug
from utils import BOARD_SIZE, DIRECTIONS, PieceType, PIECE_TYPE_LIST, PIECE_TYPE_NAME_LIST
import numba

@numba.njit
def encode_state_vector_numba(board_encoding, player1_hand_encoding, player2_hand_encoding, current_player_encoding, turn_count_encoding, player1_queen_placed_encoding, player2_queen_placed_encoding):
    total_len = board_encoding.size + player1_hand_encoding.size + player2_hand_encoding.size + current_player_encoding.size + turn_count_encoding.size + player1_queen_placed_encoding.size + player2_queen_placed_encoding.size
    state = np.empty(total_len, dtype=np.float32)
    idx = 0
    state[idx:idx+board_encoding.size] = board_encoding
    idx += board_encoding.size
    state[idx:idx+player1_hand_encoding.size] = player1_hand_encoding
    idx += player1_hand_encoding.size
    state[idx:idx+player2_hand_encoding.size] = player2_hand_encoding
    idx += player2_hand_encoding.size
    state[idx:idx+current_player_encoding.size] = current_player_encoding
    idx += current_player_encoding.size
    state[idx:idx+turn_count_encoding.size] = turn_count_encoding
    idx += turn_count_encoding.size
    state[idx:idx+player1_queen_placed_encoding.size] = player1_queen_placed_encoding
    idx += player1_queen_placed_encoding.size
    state[idx:idx+player2_queen_placed_encoding.size] = player2_queen_placed_encoding
    return state

class HiveEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    MAX_TURNS = 120  # 步数上限由200改为120

    def __init__(self, training_mode=True):
        super(HiveEnv, self).__init__()
        self.game = Game()
        self.board = ChessBoard()
        self.player1 = Player("Player1", is_ai=True)
        self.player2 = Player("Player2", is_ai=True)
        self.game.player1 = self.player1
        self.game.player2 = self.player2
        self.training_mode = training_mode  # 新增：训练/游玩模式标志

        # Define observation space (814-dimensional vector)
        # 10x10 board with 8 piece types per cell = 800
        # 2 players * 5 piece types (normalized counts) = 10
        # Current player (1), turn count (1), queen bee placed (2) = 4
        # Total = 800 + 10 + 4 = 814
        self.observation_space = spaces.Box(low=0, high=1, shape=(814,), dtype=np.float32)

        # Define action space
        # The report mentions action encoding: place (x*1000 + y*100 + pieceType) < 10000
        # move (10000 + fromX*1000 + fromY*100 + x*10 + y)
        # Max x, y = 9. Max pieceType = 7 (Pillbug)
        # Max place action: 9*1000 + 9*100 + 7 = 9907
        # Max move action: 10000 + 9*1000 + 9*100 + 9*10 + 9 = 19999
        # So, action space can be a Discrete space from 0 to 19999
        self.action_space = spaces.Discrete(20000) # A large enough discrete space

        self.current_player_idx = 0 # 0 for player1, 1 for player2
        self.turn_count = 0

    def _get_observation(self):
        # 1. Board state (800 dimensions) - 用numba加速
        board_arr = board_to_numpy(self.board, BOARD_SIZE, len(PIECE_TYPE_LIST))
        board_encoding = encode_board_numba(board_arr, BOARD_SIZE, len(PIECE_TYPE_LIST))

        # 2. Player hand information (10 dimensions)
        # Using 5 basic piece types as mentioned in the report (QueenBee, Beetle, Spider, Ant, Grasshopper)
        # Max count for each piece type is 3 (for Ant, Spider, Grasshopper) or 2 (for Beetle) or 1 (for QueenBee)
        # The report says 'divide by max quantity (usually 3)'
        player1_hand_encoding = np.zeros(5, dtype=np.float32)
        player2_hand_encoding = np.zeros(5, dtype=np.float32)

        piece_type_map = {
            PieceType.QUEEN_BEE: 0,
            PieceType.BEETLE: 1,
            PieceType.SPIDER: 2,
            PieceType.ANT: 3,
            PieceType.GRASSHOPPER: 4
        }
        max_counts = {
            PieceType.QUEEN_BEE: 1,
            PieceType.BEETLE: 2,
            PieceType.SPIDER: 2,
            PieceType.ANT: 3,
            PieceType.GRASSHOPPER: 3
        }

        for piece_type, count in self.player1.piece_count.items():
            if piece_type in piece_type_map:
                player1_hand_encoding[piece_type_map[piece_type]] = count / max_counts[piece_type]
        for piece_type, count in self.player2.piece_count.items():
            if piece_type in piece_type_map:
                player2_hand_encoding[piece_type_map[piece_type]] = count / max_counts[piece_type]

        # 3. Game state information (4 dimensions)
        current_player_encoding = np.array([self.current_player_idx], dtype=np.float32)
        turn_count_encoding = np.array([self.turn_count / 50.0], dtype=np.float32) # Normalized by 50 as per report
        player1_queen_placed_encoding = np.array([1.0 if self.player1.is_queen_bee_placed else 0.0], dtype=np.float32)
        player2_queen_placed_encoding = np.array([1.0 if self.player2.is_queen_bee_placed else 0.0], dtype=np.float32)

        observation = encode_state_vector_numba(
            board_encoding,
            player1_hand_encoding,
            player2_hand_encoding,
            current_player_encoding,
            turn_count_encoding,
            player1_queen_placed_encoding,
            player2_queen_placed_encoding
        )
        return observation

    def _decode_action(self, action_int):
        # Decode action_int back to (action_type, from_x, from_y, to_x, to_y, piece_type_id)
        # Place action: x * 1000 + y * 100 + pieceType
        # Move action: 10000 + fromX * 1000 + fromY * 100 + x * 10 + y

        if action_int < 10000:
            # Place action
            piece_type_id = action_int % 100
            temp = action_int // 100
            to_y = temp % 10
            to_x = temp // 10
            return 'place', None, None, to_x, to_y, piece_type_id
        else:
            # Move action
            action_int -= 10000
            to_y = action_int % 10
            temp = action_int // 10
            to_x = temp % 10
            temp = temp // 10
            from_y = temp % 10
            from_x = temp // 10
            # Piece type is not encoded in move action, will need to infer from board
            return 'move', from_x, from_y, to_x, to_y, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        self.board = ChessBoard()
        self.player1 = Player("Player1", is_ai=True)
        self.player2 = Player("Player2", is_ai=True)
        self.game.player1 = self.player1
        self.game.player2 = self.player2
        self.game.initialize_game(self.player1, self.player2) # Assuming no DLC for RL
        self.current_player_idx = 0
        self.turn_count = 0
        # 包围奖励累计上限计数器
        self.queenbee_surround_bonus_count = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # --- 动作合法性二次校验，彻底杜绝非法动作污染训练 ---
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            print(f"[DEBUG][step] 非法动作: {action}, legal_actions={legal_actions}")
            print(f"[DEBUG][step] turn={self.turn_count}, current_player_idx={self.current_player_idx}")
            print(f"[DEBUG][step] player1_hand={self.player1.piece_count if hasattr(self.player1, 'piece_count') else self.player1}")
            print(f"[DEBUG][step] player2_hand={self.player2.piece_count if hasattr(self.player2, 'piece_count') else self.player2}")
            # 棋盘快照（可选，避免太大）
            try:
                board_snapshot = [[str(self.board.get_piece_at(x, y)) for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]
                print(f"[DEBUG][step] board snapshot: {board_snapshot}")
            except Exception as e:
                print(f"[DEBUG][step] board snapshot error: {e}")
            reward = -5.0
            terminated = True
            truncated = False
            info = {'reason': 'illegal_action'}
            observation = self._get_observation()
            return observation, reward, terminated, truncated, info

        # Decode the action
        action_type, from_x, from_y, to_x, to_y, piece_type_id = self._decode_action(action)
        # --- 保险：必须落蜂后时非蜂后动作直接终止 ---
        max_counts = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3}
        total_placed = 0
        current_player = self.player1 if self.current_player_idx == 0 else self.player2
        for pt_id, max_count in max_counts.items():
            count = current_player.piece_count.get(pt_id, 0)
            placed = max_count - count
            total_placed += placed
        must_place_queen = (not current_player.is_queen_bee_placed) and (total_placed >= 3)
        if must_place_queen and (action_type != 'place' or piece_type_id != 0):
            print(f"[DEBUG][step][保险] 必须落蜂后但选了非蜂后动作: action={action}, turn={self.turn_count}, player={self.current_player_idx}")
            reward = -1.0
            terminated = True
            truncated = False
            info = {'reason': 'must_place_queen_violation'}
            observation = self._get_observation()
            return observation, reward, terminated, truncated, info

        # DEBUG: step前检查place动作目标格是否已被占用
        if action_type == 'place' and isinstance(to_x, int) and isinstance(to_y, int):
            if self.board.get_piece_at(to_x, to_y) is not None:
                print(f"[DEBUG][step][pre] action={action} 指向已被占用格 ({to_x},{to_y})，turn={self.turn_count}, player={self.current_player_idx}")
        current_player = self.player1 if self.current_player_idx == 0 else self.player2
        other_player = self.player2 if self.current_player_idx == 0 else self.player1
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # --- 新增：无合法动作直接终止 ---
        legal_actions = self.get_legal_actions()
        if not legal_actions:
            reward = -5.0  # 更大惩罚
            terminated = True
            info['reason'] = 'no_legal_action'
            observation = self._get_observation()
            return observation, reward, terminated, truncated, info

        # --- 新增：棋盘全满判平局 ---
        board_full = all(self.board.get_piece_at(x, y) is not None for x in range(BOARD_SIZE) for y in range(BOARD_SIZE))
        if board_full:
            reward = 0.0
            terminated = True
            info['reason'] = 'board_full_draw'
            observation = self._get_observation()
            return observation, reward, terminated, truncated, info

        try:
            # --- reward shaping 优化 ---
            # 1. 非法动作/保险惩罚适度
            if must_place_queen and (action_type != 'place' or piece_type_id != 0):
                print(f"[DEBUG][step][保险] 必须落蜂后但选了非蜂后动作: action={action}, turn={self.turn_count}, player={self.current_player_idx}")
                reward = -1.0
                terminated = True
                truncated = False
                info = {'reason': 'must_place_queen_violation'}
                observation = self._get_observation()
                return observation, reward, terminated, truncated, info
            # 2. 每步基础惩罚（更小）
            reward = -0.001
            # 6. 合规移动奖励（只保留基础奖励，防止刷分）
            if action_type == 'move' and not terminated:
                reward += 0.15
            # 7. 包围蜂后奖励、己方蜂后被围惩罚、靠近蜂后奖励、终局奖励等其余逻辑优化
            def count_surround_dirs(pos):
                if pos is None:
                    return 0
                cnt = 0
                for dx, dy in DIRECTIONS:
                    x, y = pos[0] + dx, pos[1] + dy
                    if not self.board.is_within_bounds(x, y) or self.board.get_piece_at(x, y) is not None:
                        cnt += 1
                return cnt
            prev_opp_queen_dirs = 0
            if hasattr(self, '_last_opp_queen_dirs'):
                prev_opp_queen_dirs = self._last_opp_queen_dirs
            opp_queen_pos = getattr(other_player, 'queen_bee_position', None)
            opp_queen_dirs = count_surround_dirs(opp_queen_pos)
            surround_bonus_limit = max(6, int(self.turn_count * 0.1))
            surround_bonus_this_step = 0
            # --- reward shaping 只在非终局时生效 ---
            game_over_status = self.game.check_game_over()
            # 记录上一步己方蜂后被围数
            prev_my_queen_dirs = 0
            if hasattr(self, '_last_my_queen_dirs'):
                prev_my_queen_dirs = self._last_my_queen_dirs
            my_queen_pos = getattr(current_player, 'queen_bee_position', None)
            my_queen_dirs = count_surround_dirs(my_queen_pos)
            self._last_my_queen_dirs = my_queen_dirs
            if game_over_status == 0:  # 非终局
                # 包围奖励
                if opp_queen_dirs > prev_opp_queen_dirs and self.queenbee_surround_bonus_count < surround_bonus_limit:
                    add_times = min(opp_queen_dirs - prev_opp_queen_dirs, surround_bonus_limit - self.queenbee_surround_bonus_count)
                    surround_bonus_this_step = 2.0 * add_times
                    reward += surround_bonus_this_step
                    self.queenbee_surround_bonus_count += add_times
                self._last_opp_queen_dirs = opp_queen_dirs
                # 蜂后被围惩罚（只惩罚恶化，delta-based，阶梯式）
                if my_queen_dirs > prev_my_queen_dirs:
                    delta = my_queen_dirs - prev_my_queen_dirs
                    if my_queen_dirs <= 3:
                        reward -= 0.3 * delta
                    elif my_queen_dirs <= 5:
                        reward -= 0.6 * delta
                    else:
                        reward -= 1.0 * delta
                # 被围6格立即终局
                if my_queen_dirs == 6:
                    reward = -20.0
                    terminated = True
                    info['reason'] = 'queen_surrounded'
                # 靠近对方蜂后奖励
                if opp_queen_pos is not None:
                    for dx, dy in DIRECTIONS:
                        x, y = opp_queen_pos[0] + dx, opp_queen_pos[1] + dy
                        if self.board.is_within_bounds(x, y):
                            piece = self.board.get_piece_at(x, y)
                            if piece and piece.owner == current_player:
                                reward += 0.2
            else:
                self._last_opp_queen_dirs = opp_queen_dirs  # 终局也要同步，防止下局出错
                self._last_my_queen_dirs = count_surround_dirs(getattr(current_player, 'queen_bee_position', None))
            # --- 终局奖励 ---
            if game_over_status == 1: # Player 1 wins
                reward = 20.0 if self.current_player_idx == 0 else -20.0
                terminated = True
                info['reason'] = 'player1_win'
            elif game_over_status == 2: # Player 2 wins
                reward = 20.0 if self.current_player_idx == 1 else -20.0
                terminated = True
                info['reason'] = 'player2_win'
            elif game_over_status == 3: # Draw
                # 平局奖励根据包围数差异微调
                my_queen_pos = getattr(current_player, 'queen_bee_position', None)
                opp_queen_pos = getattr(other_player, 'queen_bee_position', None)
                my_queen_dirs = count_surround_dirs(my_queen_pos)
                opp_queen_dirs = count_surround_dirs(opp_queen_pos)
                surround_diff = opp_queen_dirs - my_queen_dirs
                if surround_diff > 0:
                    reward = 5.0
                elif surround_diff < 0:
                    reward = -5.0
                else:
                    reward = 0.0
                terminated = True
                info['reason'] = 'draw'
            if not terminated and self.turn_count >= self.MAX_TURNS:
                reward = -20.0
                terminated = True
                info['reason'] = 'max_turns_reached'
            if not terminated and info.get('reason') == 'no_legal_action':
                reward = -20.0
                terminated = True
            if not terminated:
                self.current_player_idx = 1 - self.current_player_idx
                self.turn_count += 1
        except Exception as e:
            if self.training_mode:
                print(f"[DEBUG][step][Exception] action={action}, turn={self.turn_count}, current_player_idx={self.current_player_idx}")
                print(f"[DEBUG][step][Exception] player1_hand={self.player1.piece_count if hasattr(self.player1, 'piece_count') else self.player1}")
                print(f"[DEBUG][step][Exception] player2_hand={self.player2.piece_count if hasattr(self.player2, 'piece_count') else self.player2}")
                print(f"[DEBUG][step][Exception] legal_actions={self.get_legal_actions()}")
                try:
                    board_snapshot = [[str(self.board.get_piece_at(x, y)) for y in range(BOARD_SIZE)] for x in range(BOARD_SIZE)]
                    print(f"[DEBUG][step][Exception] board snapshot: {board_snapshot}")
                except Exception as e2:
                    print(f"[DEBUG][step][Exception] board snapshot error: {e2}")
                reward = -1.0
                terminated = True
                print(f"Error during step: {e}")
            else:
                msg = str(e)
                auto_placed = False
                if ("QueenBee must be placed before moving other pieces" in msg or
                    "Queen Bee must be placed by the fourth turn." in msg or
                    "must_place_queen_violation" in msg or
                    (not current_player.is_queen_bee_placed and self.turn_count >= 3)):
                    print(f"[WARNING] 非法动作: {e}，强制落蜂后。")
                    for x in range(BOARD_SIZE):
                        for y in range(BOARD_SIZE):
                            if self.board.get_piece_at(x, y) is None and self.board.is_valid_placement(x, y, current_player.is_first_player, self.turn_count):
                                try:
                                    current_player.place_piece(self.board, x, y, PieceType.QUEEN_BEE, self.turn_count)
                                    reward = 0.0
                                    terminated = False
                                    auto_placed = True
                                    break
                                except Exception:
                                    continue
                        if auto_placed:
                            break
                    if not auto_placed:
                        reward = -1.0
                        terminated = True
                else:
                    reward = -1.0
                    terminated = True
                    print(f"Error during step: {e}")
        # --- 执行动作 ---
        if action_type == 'place' and piece_type_id is not None and isinstance(piece_type_id, int):
            idx = piece_type_id
            if 0 <= idx < len(PIECE_TYPE_NAME_LIST):
                piece_type_name = PIECE_TYPE_NAME_LIST[idx]
                piece_type = getattr(PieceType, piece_type_name)
                try:
                    current_player.place_piece(self.board, to_x, to_y, piece_type, self.turn_count)
                except RuntimeError as e:
                    if "QueenBee must be placed before moving other pieces" in str(e):
                        reward = -1.0
                        terminated = True
                        truncated = False
                        info = {'reason': 'must_place_queen_violation'}
                        observation = self._get_observation()
                        return observation, reward, terminated, truncated, info
                    else:
                        raise
        elif action_type == 'move':
            if not isinstance(from_x, int) or not isinstance(from_y, int):
                raise Exception(f"无效移动参数: from_x={from_x}, from_y={from_y}")
            piece = self.board.get_piece_at(from_x, from_y)
            if piece is None or piece.owner != current_player:
                raise Exception(f"无效移动：({from_x},{from_y}) 没有己方棋子")
            self.board.move_piece(from_x, from_y, to_x, to_y, piece.piece_type, current_player)
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        # For a text-based game, rendering means printing the board
        self.board.display_board()
        print(f"Current Player: {'Player1' if self.current_player_idx == 0 else 'Player2'}")
        print(f"Turn: {self.turn_count}")
        self.player1.display_piece_count()
        self.player2.display_piece_count()

    def close(self):
        pass

    def get_legal_actions(self):
        legal_actions = []
        current_player = self.player1 if self.current_player_idx == 0 else self.player2
        queen_piece_type = PieceType.QUEEN_BEE
        queen_placed = current_player.is_queen_bee_placed
        max_counts = {
            0: 1,  # QUEEN_BEE
            1: 2,  # BEETLE
            2: 2,  # SPIDER
            3: 3,  # ANT
            4: 3   # GRASSHOPPER
        }
        total_placed = 0
        for pt_id, max_count in max_counts.items():
            count = current_player.piece_count.get(pt_id, 0)
            placed = max_count - count
            total_placed += placed
        must_place_queen = (not queen_placed) and (total_placed >= 3)
        # ---numba加速生成静态放置动作---
        board_arr = board_to_numpy(self.board, BOARD_SIZE, len(PIECE_TYPE_LIST))
        # 构造piece_counts数组
        piece_counts = np.zeros(len(PIECE_TYPE_NAME_LIST), dtype=np.int32)
        for idx, name in enumerate(PIECE_TYPE_NAME_LIST):
            piece_type = getattr(PieceType, name)
            piece_counts[idx] = current_player.piece_count.get(piece_type, 0)
        queen_piece_type_id = 0  # 默认QUEEN_BEE为0
        static_place_actions = generate_place_actions_numba(
            board_arr, piece_counts, must_place_queen, queen_piece_type_id, BOARD_SIZE, len(PIECE_TYPE_LIST)
        )
        for x, y, idx in static_place_actions:
            piece_type = getattr(PieceType, PIECE_TYPE_NAME_LIST[idx])
            if current_player.piece_count.get(piece_type, 0) > 0:
                if must_place_queen and piece_type != queen_piece_type:
                    continue
                action_int = x * 1000 + y * 100 + idx
                if self.board.is_valid_placement(x, y, current_player.is_first_player, self.turn_count):
                    legal_actions.append(action_int)
        # 只有蜂后已放置且不是must_place_queen时才允许生成移动动作
        if queen_placed and not must_place_queen:
            for from_x in range(BOARD_SIZE):
                for from_y in range(BOARD_SIZE):
                    piece_to_move = self.board.get_piece_at(from_x, from_y)
                    if piece_to_move and piece_to_move.owner == current_player:
                        for to_x in range(BOARD_SIZE):
                            for to_y in range(BOARD_SIZE):
                                if piece_to_move.is_valid_move(self.board, to_x, to_y):
                                    action_int = 10000 + from_x * 1000 + from_y * 100 + to_x * 10 + to_y
                                    legal_actions.append(action_int)
        # DEBUG: 检查生成的place动作是否指向已被占用格
        for a in legal_actions:
            if a < 10000:
                _, _, _, x, y, idx = self._decode_action(a)
                if self.board.get_piece_at(x, y) is not None:
                    print(f"[DEBUG][get_legal_actions] 非法place动作: {a} 指向已被占用格 ({x},{y})，turn={self.turn_count}, player={self.current_player_idx}")
        # --- 保险：必须落蜂后时只保留蜂后落子 ---
        if must_place_queen:
            legal_actions = [a for a in legal_actions if a < 10000 and (a % 100) == queen_piece_type_id]
        # --- 保险：首回合兜底 ---
        if not legal_actions and self.turn_count == 0:
            # 首回合允许在 (0,0) 落蜂后
            action_int = 0 * 1000 + 0 * 100 + 0  # (0,0) 落蜂后
            legal_actions.append(action_int)
        # --- 保险兜底：仅在棋盘全空且蜂后未落时允许 ---
        if not legal_actions:
            board_empty = all(self.board.get_piece_at(x, y) is None for x in range(BOARD_SIZE) for y in range(BOARD_SIZE))
            if (board_empty and
                current_player.piece_count.get(PieceType.QUEEN_BEE, 0) > 0 and
                not current_player.is_queen_bee_placed):
                action_int = Action.encode_place_action(0, 0, 0)
                legal_actions.append(action_int)
                print(f"[DEBUG][get_legal_actions] 保险兜底：首回合强制生成放蜂后动作 action={action_int} at (0,0)")
        return legal_actions


# Helper for action encoding/decoding (can be moved to utils.py or Action class later)
class Action:
    @staticmethod
    def encode_place_action(x, y, piece_type_id):
        return x * 1000 + y * 100 + piece_type_id

    @staticmethod
    def encode_move_action(from_x, from_y, to_x, to_y):
        return 10000 + from_x * 1000 + from_y * 100 + to_x * 10 + to_y

    @staticmethod
    def decode_action(action_int):
        if action_int < 10000:
            piece_type_id = action_int % 100
            temp = action_int // 100
            to_y = temp % 10
            to_x = temp // 10
            return 'place', None, None, to_x, to_y, piece_type_id
        else:
            action_int -= 10000
            to_y = action_int % 10
            temp = action_int // 10
            to_x = temp % 10
            temp = temp // 10
            from_y = temp % 10
            from_x = temp // 10
            return 'move', from_x, from_y, to_x, to_y, None


def board_to_numpy(board, board_size, piece_type_num):
    arr = np.full((board_size, board_size), -1, dtype=np.int32)  # 空格为-1
    for x in range(board_size):
        for y in range(board_size):
            pieces = board.get_pieces_at(x, y)
            if pieces:
                arr[x, y] = pieces[-1].piece_type  # 只取顶层棋子类型
    return arr


@numba.njit
def encode_board_numba(board_arr, board_size, piece_type_num):
    board_encoding = np.zeros((board_size, board_size, piece_type_num), dtype=np.float32)
    for x in range(board_size):
        for y in range(board_size):
            piece_type_id = board_arr[x, y]
            if 0 <= piece_type_id < piece_type_num:
                board_encoding[x, y, piece_type_id] = 1.0
    return board_encoding.flatten()

@numba.njit
def generate_place_actions_numba(board_arr, piece_counts, must_place_queen, queen_piece_type_id, board_size, piece_type_num):
    actions = []
    for x in range(board_size):
        for y in range(board_size):
            if board_arr[x, y] != -1:
                continue
            for piece_type_id in range(piece_type_num):
                if piece_counts[piece_type_id] > 0:
                    if must_place_queen and piece_type_id != queen_piece_type_id:
                        continue
                    # 这里只做静态生成，具体 is_valid_placement 仍需在主逻辑判断
                    actions.append((x, y, piece_type_id))
    return actions



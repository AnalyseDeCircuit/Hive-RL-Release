import numpy as np
from typing import List, Dict, Tuple, Optional
from game import Game
from board import ChessBoard
from player import Player
from piece import Piece, PieceType
from hive_env import Action # Re-use the Action helper for action encoding/decoding
from utils import BOARD_SIZE, PIECE_TYPE_LIST, PIECE_TYPE_NAME_LIST

class GameState:
    """
    Represents the current state of the Hive game for AI processing.
    This class encapsulates the board state, player information, and provides
    methods for state encoding and legal action generation as described in the report.
    """
    def __init__(self, game_instance: Game, board_instance: ChessBoard, current_player_idx: int):
        self.game = game_instance
        self.board = board_instance
        self.current_player_idx = current_player_idx # 0 for player1, 1 for player2

        self.player1_pieces = self.game.player1.piece_count.copy()
        self.player2_pieces = self.game.player2.piece_count.copy()
        self.player1_queen_placed = self.game.player1.is_queen_bee_placed
        self.player2_queen_placed = self.game.player2.is_queen_bee_placed
        self.turn_count = self.game.turn_count

    def get_encoded_state(self) -> np.ndarray:
        """
        Encodes the current game state into an 814-dimensional feature vector.
        As described in the report:
        - 800 dimensions for board state (10x10x8 one-hot encoding for piece types).
        - 10 dimensions for player hand information (normalized counts of 5 basic piece types).
        - 4 dimensions for game state information (current player, turn count, queen bee placed status).
        """
        # 1. Board state (800 dimensions)
        board_encoding = np.zeros((BOARD_SIZE, BOARD_SIZE, len(PIECE_TYPE_LIST)), dtype=np.float32)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                pieces_at_pos = self.board.get_pieces_at(x, y)
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

        for piece_type, count in self.player1_pieces.items():
            piece_type_id = piece_type if isinstance(piece_type, int) else int(getattr(PieceType, str(piece_type), 0))
            if piece_type_id in piece_type_map:
                player1_hand_encoding[piece_type_map[piece_type_id]] = count / max_counts[piece_type_id]
        for piece_type, count in self.player2_pieces.items():
            piece_type_id = piece_type if isinstance(piece_type, int) else int(getattr(PieceType, str(piece_type), 0))
            if piece_type_id in piece_type_map:
                player2_hand_encoding[piece_type_map[piece_type_id]] = count / max_counts[piece_type_id]

        # 3. Game state information (4 dimensions)
        current_player_encoding = np.array([self.current_player_idx], dtype=np.float32)
        turn_count_encoding = np.array([self.turn_count / 50.0], dtype=np.float32) # Normalized by 50 as per report
        player1_queen_placed_encoding = np.array([1.0 if self.player1_queen_placed else 0.0], dtype=np.float32)
        player2_queen_placed_encoding = np.array([1.0 if self.player2_queen_placed else 0.0], dtype=np.float32)

        encoded_state = np.concatenate([
            board_encoding,
            player1_hand_encoding,
            player2_hand_encoding,
            current_player_encoding,
            turn_count_encoding,
            player1_queen_placed_encoding,
            player2_queen_placed_encoding
        ])
        return encoded_state

    def get_legal_actions(self) -> List[int]:
        """
        Generates all legal actions for the current player in the current game state.
        Returns a list of encoded action integers.
        """
        legal_actions = []
        current_player_obj = self.game.player1 if self.current_player_idx == 0 else self.game.player2

        # Generate place actions
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for piece_type_id in PIECE_TYPE_LIST:
                    piece_type = piece_type_id
                    if current_player_obj.piece_count.get(piece_type, 0) > 0:
                        if self.board.is_valid_placement(x, y, current_player_obj.is_first_player, self.turn_count):
                            action_int = Action.encode_place_action(x, y, piece_type_id)
                            legal_actions.append(action_int)

        # Generate move actions
        for from_x in range(BOARD_SIZE):
            for from_y in range(BOARD_SIZE):
                piece_to_move = self.board.get_piece_at(from_x, from_y)
                if piece_to_move and piece_to_move.owner == current_player_obj:
                    for to_x in range(BOARD_SIZE):
                        for to_y in range(BOARD_SIZE):
                            if piece_to_move.is_valid_move(self.board, to_x, to_y):
                                action_int = Action.encode_move_action(from_x, from_y, to_x, to_y)
                                legal_actions.append(action_int)
        return legal_actions



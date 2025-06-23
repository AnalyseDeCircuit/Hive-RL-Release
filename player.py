from typing import Dict, Optional
from piece import PieceType
from utils import PIECE_TYPE_NAME_LIST, PIECE_TYPE_LIST

class Player:
    NO_DLC_PIECE_COUNT = {
        PieceType.QUEEN_BEE: 1,
        PieceType.BEETLE: 2,
        PieceType.SPIDER: 2,
        PieceType.ANT: 3,
        PieceType.GRASSHOPPER: 3
    }
    DLC_PIECE_COUNT = {
        PieceType.QUEEN_BEE: 1,
        PieceType.BEETLE: 2,
        PieceType.SPIDER: 2,
        PieceType.ANT: 3,
        PieceType.GRASSHOPPER: 3,
        PieceType.LADYBUG: 1,
        PieceType.MOSQUITO: 1,
        PieceType.PILLBUG: 1
    }

    def __init__(self, name: str, is_first_player: bool = False, is_ai: bool = False, use_dlc: bool = False):
        self.name = name
        self.is_first_player = is_first_player
        self.is_ai = is_ai
        # 统一piece_count的key为int，防止int/枚举混用导致查不到
        base_dict = Player.DLC_PIECE_COUNT if use_dlc else Player.NO_DLC_PIECE_COUNT
        self.piece_count: Dict[int, int] = {int(k): v for k, v in base_dict.items()}
        self.is_queen_bee_placed: bool = False
        self.queen_bee_position: Optional[tuple] = None

    def get_name(self) -> str:
        return self.name

    def get_piece_count(self, piece_type: int) -> int:
        return self.piece_count.get(piece_type, 0)

    def place_piece(self, board, x: int, y: int, piece_type: int, turn_count: int):
        piece_type = int(piece_type)
        if piece_type == PieceType.QUEEN_BEE and turn_count > 3 and not self.is_queen_bee_placed:
            raise RuntimeError("Queen Bee must be placed by the fourth turn.")
        if self.piece_count.get(piece_type, 0) <= 0:
            raise RuntimeError(f"No {PIECE_TYPE_NAME_LIST[piece_type]} pieces left to place.")

        # Check if QueenBee is placed before moving other pieces
        if piece_type != PieceType.QUEEN_BEE and not self.is_queen_bee_placed and turn_count > 1:
            raise RuntimeError("QueenBee must be placed before moving other pieces.")

        board.place_piece(x, y, piece_type, self)
        self.piece_count[piece_type] -= 1
        if piece_type == PieceType.QUEEN_BEE:
            self.is_queen_bee_placed = True
            self.queen_bee_position = (x, y)

    def move_piece(self, board, from_x: int, from_y: int, to_x: int, to_y: int, piece_type: int):
        piece_type = int(piece_type)
        if not self.is_queen_bee_placed:
            raise RuntimeError("QueenBee has not been placed.")

        board.move_piece(from_x, from_y, to_x, to_y, piece_type, self)
        if piece_type == PieceType.QUEEN_BEE:
            self.queen_bee_position = (to_x, to_y)

    def display_piece_count(self):
        print(f"Piece count for {self.name}:")
        # 每行显示两个棋子，数量为0红色，>0绿色
        from utils import PIECE_TYPE_NAME_LIST, PIECE_TYPE_LIST
        RESET = '\033[0m'
        GREEN = '\033[92m'
        RED = '\033[91m'
        row = []
        for idx, piece_type in enumerate(PIECE_TYPE_LIST):
            count = self.piece_count.get(int(piece_type), 0)
            name = PIECE_TYPE_NAME_LIST[piece_type]
            color = GREEN if count > 0 else RED
            row.append(f"  {name}: {color}{count}{RESET}")
            if len(row) == 2:
                print(' |'.join(row))
                row = []
        if row:
            print(' |'.join(row))

    def clone(self):
        # 动态获取自身类型，保留use_dlc参数
        cls = type(self)
        cloned_player = cls(self.name, self.is_first_player, self.is_ai, use_dlc=(len(self.piece_count) > 5))
        cloned_player.piece_count = self.piece_count.copy()
        cloned_player.is_queen_bee_placed = self.is_queen_bee_placed
        cloned_player.queen_bee_position = self.queen_bee_position
        return cloned_player



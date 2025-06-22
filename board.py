# board.py

from typing import List, Optional
from utils import BOARD_SIZE, DIRECTIONS, PieceType, PIECE_TYPE_NAME_LIST
from piece import Piece, QueenBee, Beetle, Grasshopper, Spider, Ant, Ladybug, Mosquito, Pillbug

class ChessBoard:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        # board is a 3D list: board[x][y] is a list of pieces (stack)
        self.board: List[List[List[Piece]]] = \
            [[[] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def get_width(self) -> int:
        return BOARD_SIZE

    def get_height(self) -> int:
        return BOARD_SIZE

    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def get_piece_at(self, x: int, y: int) -> Optional[Piece]:
        if not self.is_within_bounds(x, y) or not self.board[x][y]:
            return None
        return self.board[x][y][-1]  # Get the top piece of the stack

    def get_pieces_at(self, x: int, y: int) -> List[Piece]:
        if not self.is_within_bounds(x, y):
            return []
        return self.board[x][y]

    def get_adjacent_pieces(self, x: int, y: int) -> List[Piece]:
        adjacent_pieces = []
        for dx, dy in DIRECTIONS:
            adj_x, adj_y = x + dx, y + dy
            if self.is_within_bounds(adj_x, adj_y):
                piece = self.get_piece_at(adj_x, adj_y)
                if piece:
                    adjacent_pieces.append(piece)
        return adjacent_pieces

    def has_adjacent_piece(self, x: int, y: int) -> bool:
        for dx, dy in DIRECTIONS:
            adj_x, adj_y = x + dx, y + dy
            if self.is_within_bounds(adj_x, adj_y) and self.board[adj_x][adj_y]:
                return True
        return False

    def is_position_surrounded(self, x: int, y: int, consider_edge: bool = False) -> bool:
        for dx, dy in DIRECTIONS:
            adj_x, adj_y = x + dx, y + dy
            if self.is_within_bounds(adj_x, adj_y):
                if not self.board[adj_x][adj_y]:  # If any adjacent position is empty
                    return False
            elif not consider_edge:  # If not considering edge as surrounded and out of bounds
                return False
        return True  # All adjacent positions are occupied or out of bounds (and consider_edge is True)

    def place_piece(self, x: int, y: int, piece_type: PieceType, player):
        if not self.is_within_bounds(x, y):
            raise IndexError("Coordinates out of bounds.")

        new_piece: Optional[Piece] = None
        if piece_type == PieceType.QUEEN_BEE:
            new_piece = QueenBee(x, y, player)
        elif piece_type == PieceType.BEETLE:
            new_piece = Beetle(x, y, player)
        elif piece_type == PieceType.GRASSHOPPER:
            new_piece = Grasshopper(x, y, player)
        elif piece_type == PieceType.SPIDER:
            new_piece = Spider(x, y, player)
        elif piece_type == PieceType.ANT:
            new_piece = Ant(x, y, player)
        elif piece_type == PieceType.LADYBUG:
            new_piece = Ladybug(x, y, player)
        elif piece_type == PieceType.MOSQUITO:
            new_piece = Mosquito(x, y, player)
        elif piece_type == PieceType.PILLBUG:
            new_piece = Pillbug(x, y, player)
        else:
            raise ValueError("Invalid piece type.")

        # If the position is empty or the new piece is a Beetle, place it.
        # Otherwise, raise an error (cannot place on top of other pieces unless it\\'s a Beetle).
        if not self.board[x][y] or new_piece.get_piece_type() == PieceType.BEETLE:
            self.board[x][y].append(new_piece)
        else:
            raise RuntimeError("Cannot place piece here. Position is occupied.")

    def move_piece(self, from_x: int, from_y: int, to_x: int, to_y: int, piece_type: PieceType, player):
        if not self.is_within_bounds(from_x, from_y) or not self.is_within_bounds(to_x, to_y):
            raise IndexError("Coordinates out of bounds.")

        if not self.board[from_x][from_y]:
            raise ValueError("No piece at the starting position.")

        piece_to_move = self.board[from_x][from_y][-1]

        if piece_to_move.get_piece_type() != piece_type:
            from utils import PIECE_TYPE_NAME_LIST, PieceType
            # 兼容int和枚举（PieceType无__int__，用value属性或枚举常量比对）
            if isinstance(piece_type, int):
                type_idx = piece_type
            elif hasattr(piece_type, 'value'):
                type_idx = piece_type.value
            else:
                # fallback: 枚举常量比对
                type_idx = next((i for i, name in enumerate(PIECE_TYPE_NAME_LIST) if getattr(PieceType, name) == piece_type), 0)
            raise ValueError(f"No {PIECE_TYPE_NAME_LIST[type_idx]} at the given position.")

        if piece_to_move.get_owner() != player:
            raise ValueError("You can only move your own pieces.")

        if not piece_to_move.is_valid_move(self, to_x, to_y):
            raise ValueError("Invalid move for this piece type.")

        # Remove the piece from the old position
        self.board[from_x][from_y].pop()

        # Update the piece\\'s position and place it at the new position
        piece_to_move.set_position(to_x, to_y)
        self.board[to_x][to_y].append(piece_to_move)

    def display_board(self):
        # 颜色代码
        BLUE = '\033[94m'
        ORANGE = '\033[38;5;208m'
        RESET = '\033[0m'
        # 打印列坐标
        print("   ", end="")
        for x in range(BOARD_SIZE):
            print(f"{x:2d} ", end="")
        print()
        for y in range(BOARD_SIZE):
            if y % 2 == 1:
                print("   ", end="")
            print(f"{y:2d} ", end="")
            for x in range(BOARD_SIZE):
                piece = self.get_piece_at(x, y)
                if piece:
                    name = piece.get_name() or "?"
                    owner_obj = piece.get_owner()
                    owner = owner_obj.get_name() if owner_obj else "?"
                    # 判断玩家1/2
                    if hasattr(owner_obj, 'is_first_player') and owner_obj.is_first_player:
                        color = BLUE
                    else:
                        color = ORANGE
                    print(f"{color}{name[0] if name else '?'}{owner[0] if owner else '?'}{RESET} ", end="")
                else:
                    print(".  ", end="")
            print()
        print("\n  y/x => 横为 x 轴，竖为 y 轴（偶数行缩进模拟六边形，蓝=玩家1，橙=玩家2）")

    def clear_board(self):
        self.board = [[[] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def clone(self):
        cloned_board = ChessBoard()
        cloned_board.clear_board() # Ensure it's empty before copying
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for piece in self.board[x][y]:
                    # Create new piece instances for the cloned board
                    new_piece = type(piece)(piece.x, piece.y, piece.owner) # Assuming piece constructor takes x, y, owner
                    cloned_board.board[x][y].append(new_piece)
        return cloned_board

    def is_valid_placement(self, x: int, y: int, is_first_player: bool, turn_count: int) -> bool:
        # Simplified placement validation for AI. Full validation is in Game class.
        result = None
        if not self.is_within_bounds(x, y):
            result = False
            return result

        # First move can be anywhere on an empty cell
        if turn_count <= 1:
            is_empty = not self.get_piece_at(x, y)
            result = is_empty
            return result

        # Subsequent moves must be adjacent to an existing piece
        if not self.has_adjacent_piece(x, y):
            result = False
            return result

        adjacent_own_pieces = 0
        for dx, dy in DIRECTIONS:
            adj_x, adj_y = x + dx, y + dy
            if self.is_within_bounds(adj_x, adj_y):
                piece = self.get_piece_at(adj_x, adj_y)
                if piece:
                    if (is_first_player and piece.owner.is_first_player) or \
                       (not is_first_player and not piece.owner.is_first_player):
                        adjacent_own_pieces += 1
        if adjacent_own_pieces == 0 and turn_count > 1:
            result = False
            return result
        # 删除对方棋子相邻判定，允许与对方棋子相邻
        result = True
        return result



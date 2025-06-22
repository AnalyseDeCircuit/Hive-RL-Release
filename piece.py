# piece.py

from abc import ABC, abstractmethod
from utils import PieceType, DIRECTIONS

class Piece(ABC):
    def __init__(self, x: int, y: int, piece_type: PieceType, owner):
        self.x = x
        self.y = y
        self.piece_type = piece_type
        self.owner = owner

    def set_position(self, new_x: int, new_y: int):
        self.x = new_x
        self.y = new_y

    def get_piece_type(self) -> PieceType:
        return self.piece_type

    def get_position(self) -> tuple[int, int]:
        return (self.x, self.y)

    def get_owner(self):
        return self.owner

    def set_owner(self, new_owner):
        self.owner = new_owner

    def set_piece_type(self, new_piece_type: PieceType):
        self.piece_type = new_piece_type

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        pass

    def clone(self, new_owner):
        # Create a new instance of the same piece type with the same attributes
        # and assign the new owner (cloned player)
        return type(self)(self.x, self.y, new_owner)

class QueenBee(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.QUEEN_BEE, owner)

    def get_name(self) -> str:
        return "QueenBee"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # 只允许六边形六方向单步移动
        dx = to_x - self.x
        dy = to_y - self.y
        return ((dx, dy) in DIRECTIONS) and board.has_adjacent_piece(to_x, to_y)

class Beetle(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.BEETLE, owner)

    def get_name(self) -> str:
        return "Beetle"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # 只允许六边形六方向单步移动（可叠加）
        dx = to_x - self.x
        dy = to_y - self.y
        return ((dx, dy) in DIRECTIONS) and board.has_adjacent_piece(to_x, to_y)

class Grasshopper(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.GRASSHOPPER, owner)

    def get_name(self) -> str:
        return "Grasshopper"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        if not board.has_adjacent_piece(to_x, to_y): return False

        # Grasshopper moves by jumping over a line of pieces in a straight line
        # It must jump over at least one piece and land on the first empty space.
        if self.x == to_x:  # Vertical move
            if self.y == to_y: return False # Cannot jump to self
            step_y = 1 if to_y > self.y else -1
            current_y = self.y + step_y
            jumped_over_at_least_one = False
            while board.is_within_bounds(self.x, current_y) and not board.get_piece_at(self.x, current_y) is None:
                jumped_over_at_least_one = True
                current_y += step_y
            return jumped_over_at_least_one and current_y == to_y and board.get_piece_at(to_x, to_y) is None
        elif self.y == to_y:  # Horizontal move
            if self.x == to_x: return False # Cannot jump to self
            step_x = 1 if to_x > self.x else -1
            current_x = self.x + step_x
            jumped_over_at_least_one = False
            while board.is_within_bounds(current_x, self.y) and not board.get_piece_at(current_x, self.y) is None:
                jumped_over_at_least_one = True
                current_x += step_x
            return jumped_over_at_least_one and current_x == to_x and board.get_piece_at(to_x, to_y) is None
        elif abs(to_x - self.x) == abs(to_y - self.y): # Diagonal move (for hex grid, but simplified here)
            step_x = 1 if to_x > self.x else -1
            step_y = 1 if to_y > self.y else -1
            current_x, current_y = self.x + step_x, self.y + step_y
            jumped_over_at_least_one = False
            while board.is_within_bounds(current_x, current_y) and not board.get_piece_at(current_x, current_y) is None:
                jumped_over_at_least_one = True
                current_x += step_x
                current_y += step_y
            return jumped_over_at_least_one and current_x == to_x and current_y == to_y and board.get_piece_at(to_x, to_y) is None
        return False

class Spider(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.SPIDER, owner)

    def get_name(self) -> str:
        return "Spider"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # Spider moves exactly three steps around the hive.
        # This is a complex pathfinding problem, for now, we\'ll use the simplified C++ logic.
        # The C++ code uses Manhattan distance, which is not accurate for Hive\'s hex grid.
        distance = abs(to_x - self.x) + abs(to_y - self.y)
        return distance == 3 and board.has_adjacent_piece(to_x, to_y)

class Ant(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.ANT, owner)

    def get_name(self) -> str:
        return "Ant"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # Ant can move any number of steps around the hive, as long as it stays connected.
        # This is a complex pathfinding problem, for now, we\'ll use the simplified C++ logic.
        # The C++ code simply checks if the target position is adjacent to an existing piece.
        return board.has_adjacent_piece(to_x, to_y)

class Ladybug(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.LADYBUG, owner)

    def get_name(self) -> str:
        return "Ladybug"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # Ladybug moves exactly three steps: two on top of the hive, then one down.
        # Similar to Spider, the C++ code uses Manhattan distance.
        distance = abs(to_x - self.x) + abs(to_y - self.y)
        return distance == 3 and board.has_adjacent_piece(to_x, to_y)

class Mosquito(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.MOSQUITO, owner)

    def get_name(self) -> str:
        return "Mosquito"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # Mosquito mimics the movement of any adjacent piece.
        # The C++ code simply checks if the target position is adjacent to an existing piece.
        # The actual mimicry logic is handled in Player.py
        return board.has_adjacent_piece(to_x, to_y)

class Pillbug(Piece):
    def __init__(self, x: int, y: int, owner):
        super().__init__(x, y, PieceType.PILLBUG, owner)

    def get_name(self) -> str:
        return "Pillbug"

    def is_valid_move(self, board, to_x: int, to_y: int) -> bool:
        # 只允许六边形六方向单步移动
        dx = to_x - self.x
        dy = to_y - self.y
        return ((dx, dy) in DIRECTIONS) and board.has_adjacent_piece(to_x, to_y)



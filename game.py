from typing import Optional
from board import ChessBoard
from player import Player
from utils import PieceType

class Game:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Game, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.player1: Optional[Player] = None
        self.player2: Optional[Player] = None
        self.board: ChessBoard = ChessBoard()
        self.current_player: Optional[Player] = None
        self.is_game_over: bool = False
        self.turn_count: int = 1

    @classmethod
    def get_instance(cls):
        return cls()

    def initialize_game(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
        self.current_player = self.player1
        self.is_game_over = False
        self.turn_count = 1
        self.board.clear_board()
        # print(f"Game initialized. {self.player1.get_name()} starts.")

    def restart_game(self):
        if self.player1 and self.player2:
            self.board.clear_board()
            # 重新创建玩家，重置棋子库存，保留AIPlayer类型和use_dlc参数
            p1_cls = type(self.player1)
            p2_cls = type(self.player2)
            if p1_cls.__name__ == 'AIPlayer':
                self.player1 = p1_cls(
                    self.player1.name,
                    self.player1.is_first_player,
                    True,
                    getattr(self.player1, 'epsilon', 0.0),
                    getattr(self.player1, 'learning_rate', 0.01),
                    getattr(self.player1, 'discount_factor', 0.99),
                    (len(self.player1.piece_count) > 5)
                )
            else:
                self.player1 = p1_cls(self.player1.name, self.player1.is_first_player, self.player1.is_ai, use_dlc=(len(self.player1.piece_count) > 5))
            if p2_cls.__name__ == 'AIPlayer':
                self.player2 = p2_cls(
                    self.player2.name,
                    self.player2.is_first_player,
                    True,
                    getattr(self.player2, 'epsilon', 0.0),
                    getattr(self.player2, 'learning_rate', 0.01),
                    getattr(self.player2, 'discount_factor', 0.99),
                    (len(self.player2.piece_count) > 5)
                )
            else:
                self.player2 = p2_cls(self.player2.name, self.player2.is_first_player, self.player2.is_ai, use_dlc=(len(self.player2.piece_count) > 5))
            self.current_player = self.player1
            self.is_game_over = False
            self.turn_count = 1
            print("Game has been restarted.")
        else:
            print("Game not initialized. Cannot restart.")

    def switch_player(self):
        if self.current_player == self.player1:
            self.current_player = self.player2
        else:
            self.current_player = self.player1
            self.turn_count += 1

    def check_game_over(self) -> int: # Returns 0: ongoing, 1: player1 wins, 2: player2 wins, 3: draw
        player1_queen_surrounded = False
        player2_queen_surrounded = False
        player1_queen_found = False
        player2_queen_found = False

        # Find QueenBee in all stack levels and check if surrounded
        for x in range(self.board.get_width()):
            for y in range(self.board.get_height()):
                # 遍历该格子所有层的棋子，防止蜂后被覆盖
                pieces = self.board.get_pieces_at(x, y)
                for piece in pieces:
                    if piece.get_piece_type() == PieceType.QUEEN_BEE:
                        owner = piece.get_owner()
                        # 使用consider_edge=True，边缘也算被包围
                        surrounded = self.board.is_position_surrounded(x, y, consider_edge=True)
                        if owner == self.player1:
                            player1_queen_found = True
                            if surrounded:
                                player1_queen_surrounded = True
                        elif owner == self.player2:
                            player2_queen_found = True
                            if surrounded:
                                player2_queen_surrounded = True
                        # 一个格子不会有多个蜂后，找到后退出当前格子层遍历
                        break

        # 只有蜂后已放置且被包围才算输，否则游戏继续
        if (player1_queen_found and player1_queen_surrounded) or (player2_queen_found and player2_queen_surrounded):
            self.is_game_over = True
            if player1_queen_found and player1_queen_surrounded and player2_queen_found and player2_queen_surrounded:
                return 3 # Draw
            elif player1_queen_found and player1_queen_surrounded:
                return 2 # Player 2 wins
            else:
                return 1 # Player 1 wins
        return 0 # Game ongoing

    def get_winner(self) -> Optional[Player]:
        game_status = self.check_game_over()
        if game_status == 1:
            return self.player1
        elif game_status == 2:
            return self.player2
        else:
            return None

    def play_turn(self):
        if self.is_game_over:
            print("Game is already over.")
            return

        if self.current_player is not None:
            print(f"{self.current_player.get_name()}'s turn.")
        else:
            print("当前玩家未初始化。")

        valid_input = False
        while not valid_input:
            action = input("Enter action (P for place, M for move): ").strip().upper()

            try:
                if action == 'P':
                    x, y, piece_type_str = input("Enter coordinates (x y) and piece type (0-7): ").split()
                    x, y, piece_type = int(x), int(y), int(piece_type_str)
                    # ---保险：手动落子时也用环境合法性校验---
                    from hive_env import HiveEnv, Action
                    env = HiveEnv(training_mode=False)
                    env.board = self.board
                    env.player1 = self.player1  # type: ignore
                    env.player2 = self.player2  # type: ignore
                    env.current_player_idx = 0 if self.current_player == self.player1 else 1
                    env.turn_count = self.turn_count
                    legal_actions = env.get_legal_actions()
                    action_int = Action.encode_place_action(x, y, piece_type)
                    if action_int not in legal_actions:
                        print("[保险] 非法落子：该位置不可放置或已被占用！请重新输入。")
                        continue
                    if self.current_player is not None:
                        self.current_player.place_piece(self.board, x, y, piece_type, self.turn_count)
                    valid_input = True
                elif action == 'M':
                    x, y, to_x, to_y = map(int, input("Enter from coordinates (x y) and to coordinates (toX toY): ").split())
                    piece_type_str = input("Enter piece type to move (0-7): ").strip()
                    piece_type = int(piece_type_str)
                    # ---保险：手动移动时也用环境合法性校验---
                    from hive_env import HiveEnv, Action
                    env = HiveEnv(training_mode=False)
                    env.board = self.board
                    env.player1 = self.player1  # type: ignore
                    env.player2 = self.player2  # type: ignore
                    env.current_player_idx = 0 if self.current_player == self.player1 else 1
                    env.turn_count = self.turn_count
                    legal_actions = env.get_legal_actions()
                    action_int = Action.encode_move_action(x, y, to_x, to_y)
                    if action_int not in legal_actions:
                        print("[保险] 非法移动：该移动不可执行！请重新输入。")
                        continue
                    if self.current_player is not None:
                        self.current_player.move_piece(self.board, x, y, to_x, to_y, piece_type)
                    valid_input = True
                else:
                    print("Invalid action. Please enter 'P' for place or 'M' for move.")
            except (ValueError, IndexError, RuntimeError) as e:
                print(f"Invalid move: {e}")
                print("Please try again.")

        self.display_board()
        self.display_turn_count()

        game_status = self.check_game_over()
        if game_status != 0:
            winner = self.get_winner()
            if winner:
                print(f"{winner.get_name()} wins the game!")
            else:
                print("The game ended in a draw!")
            return

        self.switch_player()

    def display_board(self):
        self.board.display_board()
        # 输出六个方向说明
        print("六个方向坐标增量: (1,0) 右，(-1,0) 左，(0,1) 下，(0,-1) 上，(1,-1) 右上，(-1,1) 左下")
        # 自动输出蜂后包围状态
        for player, label in [(self.player1, "1"), (self.player2, "2")]:
            pos = getattr(player, 'queen_bee_position', None)
            if player and getattr(player, 'is_queen_bee_placed', False) and pos is not None:
                x, y = pos
                if self.board.is_position_surrounded(x, y, consider_edge=True):
                    print(f"\033[91m[提示] 玩家{label}的蜂后已被包围！\033[0m")
                else:
                    print(f"\033[92m[提示] 玩家{label}的蜂后未被包围。\033[0m")

    def display_turn_count(self):
        print(f"Current Turn: {self.turn_count}")

    def end_game(self):
        self.is_game_over = True
        print("Game has ended.")

    def get_current_player(self) -> Optional[Player]:
        return self.current_player

    def clone(self):
        cloned_game = Game.get_instance()
        if self.player1 is not None:
            cloned_game.player1 = self.player1.clone()
        else:
            cloned_game.player1 = None
        if self.player2 is not None:
            cloned_game.player2 = self.player2.clone()
        else:
            cloned_game.player2 = None
        cloned_game.board = self.board.clone()
        cloned_game.current_player = cloned_game.player1 if self.current_player == self.player1 else cloned_game.player2
        cloned_game.is_game_over = self.is_game_over
        cloned_game.turn_count = self.turn_count
        return cloned_game



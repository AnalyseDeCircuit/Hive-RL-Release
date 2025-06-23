# main.py

from game import Game
from player import Player
from ai_player import AIPlayer
from ai_trainer import AITrainer
from ai_evaluator import AIEvaluator
import os

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def display_main_menu():
    clear_screen()
    print("\n--- Hive Game ---")
    print("1. Human vs. Human")
    print("2. Human vs. AI")
    print("3. AI Training")
    print("4. Evaluate AI")
    print("5. Exit Game")
    return input("Choose an option: ").strip()

def get_player_names():
    player1_name = input("Enter Player 1's name: ").strip()
    player2_name = input("Enter Player 2's name: ").strip()
    return player1_name, player2_name

def game_loop(game: Game):
    exit_game_loop = False
    while not exit_game_loop:
        clear_screen()

        game.display_turn_count()

        current_player = game.get_current_player()
        if current_player:
            print(f"Current Player: {current_player.get_name()}")
            print("Player's Hand (Remaining Pieces):")
            current_player.display_piece_count()

        game.display_board()

        print("\n--- Game Menu ---")
        print("1. Play Turn")
        print("2. Restart Game")
        print("3. End Current Game")
        print("4. Return to Main Menu")
        choice = input("Choose an option: ").strip()

        if choice == '1':
            from hive_env import Action
            from utils import PIECE_TYPE_NAME_LIST, PieceType, BOARD_SIZE, PIECE_TYPE_LIST
            if isinstance(current_player, AIPlayer):
                print(f"{current_player.get_name()} is thinking...")
                current_player_idx = 0 if current_player == game.player1 else 1
                action = current_player.select_action(None, game, game.board, current_player_idx, debug=False)
                # 如果AI无动作且蜂后未落，强制生成一个放蜂后动作
                if action is None and not getattr(current_player, 'is_queen_bee_placed', False):
                    print("[DEBUG] AI保险：无动作且蜂后未落，强制生成放蜂后动作 (0,0)")
                    action = Action.encode_place_action(0, 0, 0)
                if action is None:
                    # debug输出当前可落子空格数
                    empty_count = sum(
                        1 for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)
                        if game.board.get_piece_at(x, y) is None
                    )
                    print(f"[DEBUG] AI无合法动作，当前空格数: {empty_count}, 蜂后已落: {getattr(current_player, 'is_queen_bee_placed', False)}")
                    # 额外debug：蜂后已落但无动作时，输出AI可用棋子、手牌、AIPlayer动作生成方法名
                    if getattr(current_player, 'is_queen_bee_placed', False):
                        print("[DEBUG] AI蜂后已落但无动作，可能为AIPlayer动作生成逻辑异常。请检查ai_player.py的_get_legal_actions_from_game_state和select_action实现。")
                        if hasattr(current_player, 'hand'):
                            print(f"[DEBUG] AI手牌: {getattr(current_player, 'hand', None)}")
                        if hasattr(current_player, 'get_name'):
                            print(f"[DEBUG] AI名称: {current_player.get_name()}")
                        # 如果AIPlayer有get_legal_actions_from_game_state方法，可尝试直接输出
                        if hasattr(current_player, '_get_legal_actions_from_game_state'):
                            try:
                                actions_debug = current_player._get_legal_actions_from_game_state(game, game.board, current_player_idx)
                                print(f"[DEBUG] AI _get_legal_actions_from_game_state 返回: {actions_debug}")
                            except Exception as e:
                                print(f"[DEBUG] 调用_get_legal_actions_from_game_state异常: {e}")
                    print("AI has no legal moves. Game might be stuck or over.")
                    game.is_game_over = True
                else:
                    action_type, from_x, from_y, to_x, to_y, piece_type_id = Action.decode_action(action)
                    if action_type == 'place' and isinstance(piece_type_id, int) and 0 <= piece_type_id < len(PIECE_TYPE_NAME_LIST):
                        piece_type = getattr(PieceType, PIECE_TYPE_NAME_LIST[piece_type_id])
                        print(f"AI places {getattr(piece_type, 'name', str(piece_type))} at ({to_x}, {to_y})")
                        safe_piece_type = getattr(piece_type, 'value', piece_type)
                        if not isinstance(safe_piece_type, int):
                            try:
                                safe_piece_type = int(safe_piece_type)
                            except Exception:
                                safe_piece_type = 0
                        if all(isinstance(v, int) and v is not None for v in (to_x, to_y, safe_piece_type)) and hasattr(current_player, 'place_piece') and current_player is not None:
                            try:
                                current_player.place_piece(game.board, to_x, to_y, safe_piece_type, game.turn_count)
                                game.switch_player()
                            except RuntimeError as e:
                                msg = str(e)
                                if ("QueenBee must be placed before moving other pieces" in msg or
                                    "Queen Bee must be placed by the fourth turn." in msg or
                                    "must_place_queen_violation" in msg or
                                    (not getattr(current_player, 'is_queen_bee_placed', False) and game.turn_count >= 3)):
                                    print(f"[WARNING] 非法动作: {e}，强制落蜂后。")
                                    # 强制落蜂后
                                    placed = False
                                    for x in range(BOARD_SIZE):
                                        for y in range(BOARD_SIZE):
                                            if game.board.get_piece_at(x, y) is None:
                                                try:
                                                    current_player.place_piece(game.board, x, y, 0, game.turn_count)
                                                    placed = True
                                                    game.switch_player()
                                                    break
                                                except Exception:
                                                    continue
                                        if placed:
                                            break
                                    if not placed:
                                        print("[ERROR] 无法强制落蜂后，跳过回合。")
                                else:
                                    print(f"[ERROR] place_piece异常: {e}")
                        else:
                            print("AI action decode error: invalid coordinates/type for placement or player method missing.")
                    elif action_type == 'move' and all(isinstance(v, int) and v is not None for v in (from_x, from_y, to_x, to_y)) and hasattr(current_player, 'move_piece') and current_player is not None:
                        if isinstance(from_x, int) and isinstance(from_y, int):
                            moved_piece = game.board.get_piece_at(from_x, from_y) if game.board is not None else None
                        else:
                            moved_piece = None
                        if moved_piece:
                            safe_piece_type = getattr(moved_piece.piece_type, 'value', None)
                            if not isinstance(safe_piece_type, int):
                                safe_piece_type = 0
                            print(f"AI moves {getattr(moved_piece.piece_type, 'name', str(moved_piece.piece_type))} from ({from_x}, {from_y}) to ({to_x}, {to_y})")
                            if all(isinstance(v, int) and v is not None for v in (from_x, from_y, to_x, to_y, safe_piece_type)):
                                try:
                                    current_player.move_piece(game.board, from_x, from_y, to_x, to_y, safe_piece_type)
                                    game.switch_player()
                                except RuntimeError as e:
                                    msg = str(e)
                                    if ("QueenBee must be placed before moving other pieces" in msg or
                                        "Queen Bee must be placed by the fourth turn." in msg or
                                        "must_place_queen_violation" in msg or
                                        (not getattr(current_player, 'is_queen_bee_placed', False) and game.turn_count >= 3)):
                                        print(f"[WARNING] 非法move动作: {e}，强制落蜂后。")
                                        # 强制落蜂后
                                        placed = False
                                        for x in range(BOARD_SIZE):
                                            for y in range(BOARD_SIZE):
                                                if game.board.get_piece_at(x, y) is None:
                                                    try:
                                                        current_player.place_piece(game.board, x, y, 0, game.turn_count)
                                                        placed = True
                                                        game.switch_player()
                                                        break
                                                    except Exception:
                                                        continue
                                            if placed:
                                                break
                                        if not placed:
                                            print("[ERROR] 无法强制落蜂后，跳过回合。")
                                    else:
                                        print(f"[ERROR] move_piece异常: {e}")
                            else:
                                print("AI move action decode error: invalid coordinates/type.")
                        else:
                            print("AI tried to move a non-existent piece. This should not happen.")
                    else:
                        print("AI action decode error or illegal action.")
            else: # Human player's turn
                valid_input = False
                from utils import PIECE_TYPE_LIST
                while not valid_input:
                    action = input("Enter action (P for place, M for move): ").strip().upper()
                    try:
                        if action == 'P':
                            x, y, piece_type_str = input("Enter coordinates (x y) and piece type (0-7): ").split()
                            x, y, piece_type = int(x), int(y), int(piece_type_str)
                            if piece_type not in PIECE_TYPE_LIST or not all(isinstance(v, int) for v in (x, y, piece_type)):
                                print("Invalid piece type or coordinates. Please enter valid numbers.")
                                continue
                            if hasattr(current_player, 'place_piece') and current_player is not None:
                                current_player.place_piece(game.board, x, y, piece_type, game.turn_count)
                                game.switch_player()
                                valid_input = True
                            else:
                                print("Current player cannot place pieces.")
                        elif action == 'M':
                            coords = input("Enter from coordinates (x y) and to coordinates (toX toY): ").split()
                            if len(coords) != 4:
                                print("Please enter exactly four numbers for coordinates.")
                                continue
                            x, y, to_x, to_y = map(int, coords)
                            piece_type_str = input("Enter piece type to move (0-7): ").strip()
                            piece_type = int(piece_type_str)
                            if piece_type not in PIECE_TYPE_LIST or not all(isinstance(v, int) for v in (x, y, to_x, to_y, piece_type)):
                                print("Invalid piece type or coordinates. Please enter valid numbers.")
                                continue
                            if hasattr(current_player, 'move_piece') and current_player is not None:
                                current_player.move_piece(game.board, x, y, to_x, to_y, piece_type)
                                game.switch_player()
                                valid_input = True
                            else:
                                print("Current player cannot move pieces.")
                        else:
                            print("Invalid action. Please enter 'P' for place or 'M' for move.")
                    except (ValueError, IndexError, RuntimeError, TypeError) as e:
                        print(f"Invalid move: {e}")
                        print("Please try again.")
        elif choice == '2':
            game.restart_game()
        elif choice == '3':
            game.end_game()
            exit_game_loop = True # End current game and return to main menu
        elif choice == '4':
            exit_game_loop = True # Return to main menu
        else:
            print("Invalid option. Please choose a number between 1 and 4.")

        if not exit_game_loop:
            input("\nPress Enter to continue...")

        game_status = game.check_game_over()
        if game_status != 0:
            winner = game.get_winner()
            if winner:
                print(f"{winner.get_name()} wins the game!")
            else:
                print("The game ended in a draw!")
            input("\nPress Enter to continue...")
            exit_game_loop = True

def human_vs_ai_game_loop():
    from hive_env import HiveEnv
    game = Game.get_instance()
    player1_name = input("Enter Human Player's name: ").strip()
    # 如需支持DLC，改为use_dlc=True
    use_dlc = False
    player1 = Player(player1_name, is_first_player=True, use_dlc=use_dlc)
    
    # Load AI model
    ai_player = AIPlayer("AI_Player", is_first_player=False, epsilon=0.0, use_dlc=use_dlc) # No exploration for playing
    try:
        ai_player.neural_network.load_model("./ai_model.npz")
        print("AI model loaded successfully.")
    except FileNotFoundError:
        print("AI model not found. AI will play randomly.")

    # 用HiveEnv游玩模式包装game，确保保险逻辑分离
    env = HiveEnv(training_mode=False)
    # 可选：如需用env驱动主循环，可在此处替换game_loop逻辑
    game.initialize_game(player1, ai_player)
    game_loop(game)

def ai_training_loop():
    print("\n--- AI Training ---")
    trainer = AITrainer()
    num_episodes = int(input("Enter number of training episodes (e.g., 1000): "))
    trainer.train(num_episodes=num_episodes)
    input("\nAI training complete. Press Enter to return to main menu...")

def ai_evaluation_loop():
    print("\n--- AI Evaluation ---")
    evaluator = AIEvaluator()
    num_games = int(input("Enter number of evaluation games (e.g., 100): "))
    evaluator.evaluate(num_games=num_games)
    input("\nAI evaluation complete. Press Enter to return to main menu...")

def main():
    game = Game.get_instance()
    
    while True:
        main_menu_choice = display_main_menu()

        if main_menu_choice == '1':
            player1_name, player2_name = get_player_names()
            player1 = Player(player1_name, is_first_player=True)
            player2 = Player(player2_name, is_first_player=False)
            game.initialize_game(player1, player2)
            game_loop(game)
        elif main_menu_choice == '2':
            human_vs_ai_game_loop()
        elif main_menu_choice == '3':
            ai_training_loop()
        elif main_menu_choice == '4':
            ai_evaluation_loop()
        elif main_menu_choice == '5':
            print("Exiting game. Goodbye!")
            break
        else:
            print("Invalid option. Please choose a number between 1 and 5.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()



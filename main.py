# main.py

from game import Game
from player import Player
from ai_player import AIPlayer
from ai_trainer import AITrainer
from ai_evaluator import AIEvaluator
import os
import glob

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def display_main_menu():
    clear_screen()
    # 黄色字符画标题
    print("\033[93m")  # 设置黄色
    print(r"""
  _    _ _______      ________    _____          __  __ ______ 
 | |  | |_   _\ \    / /  ____|  / ____|   /\   |  \/  |  ____|
 | |__| | | |  \ \  / /| |__    | |  __   /  \  | \  / | |__   
 |  __  | | |   \ \/ / |  __|   | | |_ | / /\ \ | |\/| |  __|  
 | |  | |_| |_   \  /  | |____  | |__| |/ ____ \| |  | | |____ 
 |_|  |_|_____|   \/   |______|  \_____/_/    \_\_|  |_|______|                                                                                                                             
    """)
    print("\033[0m")  # 重置颜色
    # 青色分隔线
    print("\033[96m" + "═" * 24 + "\033[0m")
    # 白色菜单选项
    print("\033[97m1. Human vs Human\033[0m")
    print("\033[97m2. Human vs AI\033[0m")
    print("\033[97m3. AI Training\033[0m")
    print("\033[97m4. Evaluate AI(Not available now)\033[0m")
    
    # 亮红色退出选项
    print("\033[91m5. Exit Game\033[0m")
    
    # 青色分隔线
    print("\033[96m" + "═" * 24 + "\033[0m")
    
    # 黄色输入提示
    return input("\033[93m› Choose an option (1-5): \033[0m").strip()

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
        print("\033[96m1. Play Turn\033[0m   \033[97m落子/移动棋子\033[0m")
        print("\033[96m2. Restart Game\033[0m   \033[97m重新开始本局\033[0m")
        print("\033[96m3. End Current Game\033[0m   \033[97m结束当前对局\033[0m")
        print("\033[96m4. Return to Main Menu\033[0m   \033[97m返回主菜单\033[0m")
        print("\033[93m5. Show Rules & Piece Abilities\033[0m   \033[97m查看游戏规则与棋子能力\033[0m")
        # 选中玩家颜色（蓝=玩家1，橙=玩家2）
        player_color = '\033[94m' if current_player == game.player1 else '\033[38;5;208m'
        player_name = current_player.get_name() if current_player else "?"
        prompt = f"{player_color}For {player_name}, Choose an option (1-5): \033[0m"
        choice = input(prompt).strip()

        if choice == '1':
            from hive_env import Action
            from utils import PIECE_TYPE_NAME_LIST, PieceType, BOARD_SIZE, PIECE_TYPE_LIST
            if isinstance(current_player, AIPlayer):
                print(f"{current_player.get_name()} is thinking...")
                current_player_idx = 0 if current_player == game.player1 else 1
                action = current_player.select_action(None, game, game.board, current_player_idx, debug=False)
                # 如果AI无动作且蜂后未落，强制生成一个放蜂后动作（遍历全盘找合法点）
                if action is None and not getattr(current_player, 'is_queen_bee_placed', False):
                    print("[DEBUG] AI保险：无动作且蜂后未落，遍历全盘强制生成放蜂后动作")
                    placed = False
                    for x in range(BOARD_SIZE):
                        for y in range(BOARD_SIZE):
                            if game.board.get_piece_at(x, y) is None and game.board.is_valid_placement(x, y, current_player.is_first_player, game.turn_count):
                                action = Action.encode_place_action(x, y, 0)
                                placed = True
                                break
                        if placed:
                            break
                    if not placed:
                        print("[ERROR] 无法强制落蜂后，跳过回合。")
                        action = None
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
                            # 直接使用实际棋子的类型（忽略类型检查）
                            safe_piece_type = moved_piece.piece_type  # type: ignore
                            print(f"AI moves {getattr(moved_piece.piece_type, 'name', str(moved_piece.piece_type))} from ({from_x}, {from_y}) to ({to_x}, {to_y})")
                            try:
                                # 确保坐标和类型为 int
                                assert isinstance(from_x, int) and isinstance(from_y, int) and isinstance(to_x, int) and isinstance(to_y, int)
                                assert isinstance(safe_piece_type, int)
                                # 调用时忽略静态类型检查
                                current_player.move_piece(game.board, from_x, from_y, to_x, to_y, safe_piece_type)  # type: ignore
                                game.switch_player()
                            except Exception as e:
                                print(f"[ERROR] move_piece 异常: {e}")
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
        elif choice == '5':
            print("\n\033[95m【Hive 游戏规则简述】\033[0m")
            print("- 目标：围住对方的蜂后(Queen Bee)，六个方向都被棋子或边界包围即判负。\n- 每回合可选择落子或移动己方棋子。\n- 第4回合前必须落下蜂后。\n- 棋子不能断开蜂群。\n- 棋盘为10x10，坐标(x, y)横为x轴，竖为y轴。\n")
            print("\033[95m【棋子能力说明】\033[0m")
            print("- \033[93mQueen Bee(蜂后)\033[0m: 六方向单步移动。\n- \033[94mBeetle(甲虫)\033[0m: 六方向单步，可爬到其他棋子上。\n- \033[92mSpider(蜘蛛)\033[0m: 必须连续移动三步，不能回头。\n- \033[96mAnt(蚂蚁)\033[0m: 可沿蜂群边缘任意步数移动。\n- \033[91mGrasshopper(蚂蚱)\033[0m: 跳过一条直线上的所有棋子，落到第一个空格。\n- \033[95mLadybug(瓢虫)\033[0m: 先在蜂群上走两步，再落下一步。\n- \033[97mMosquito(蚊子)\033[0m: 模仿相邻棋子的能力。\n- \033[90mPillbug(鼠妇)\033[0m: 六方向单步，并可搬运相邻棋子。\n")
            input("\033[93m按回车返回游戏菜单...\033[0m")
            continue
        else:
            print("Invalid option. Please choose a number between 1 and 5.")
            input("\nPress Enter to continue...")

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


def select_model():
    
    model_files = glob.glob("models/*/*.npz")
    if not model_files:
        print("未找到任何模型文件，AI将随机下棋。")
        return None
    print("可用模型列表：")
    for idx, file in enumerate(model_files):
        print(f"{idx}: {file}")
    sel = input("请输入要加载的模型编号（回车默认最新）:").strip()
    if sel == '':
        return model_files[-1]
    try:
        sel = int(sel)
        return model_files[sel]
    except Exception:
        print("输入无效，使用最新模型。")
        return model_files[-1]

def human_vs_ai_game_loop():
    from hive_env import HiveEnv
    game = Game.get_instance()
    player1_name = input("Enter Human Player's name: ").strip()
    # DLC option
    use_dlc = input("Enable DLC pieces? (y/n) [n]: ").strip().lower() == 'y'
    player1 = Player(player1_name, is_first_player=True, use_dlc=use_dlc)

    # 进入人机对战分支后立即询问模型
    model_path = select_model()
    ai_player = AIPlayer("AI_Player", is_first_player=False, epsilon=0.0, use_dlc=use_dlc)
    if model_path:
        try:
            ai_player.neural_network.load_model(model_path)
            print(f"AI模型已加载: {model_path}")
        except Exception as e:
            print(f"AI模型加载失败: {e}，AI将随机下棋。")
    else:
        print("未选择模型，AI将随机下棋。")

    env = HiveEnv(training_mode=False, use_dlc=use_dlc)
    game.initialize_game(player1, ai_player)
    game_loop(game)

def ai_training_loop():
    print("\n--- AI Training ---")
    # 二级菜单：新建或继续训练
    print("请选择训练模式：")
    print("1. 训练新AI（新建模型目录）")
    print("2. 继续训练已有AI（断点续训）")
    mode = input("输入选项(1-2，回车默认1): ").strip()
    trainer = None
    if mode == '2':
        # 尝试断点续训
        model_dirs = sorted([d for d in glob.glob("models/*") if os.path.isdir(d)])
        if not model_dirs:
            print("未找到任何历史AI模型，切换到新建训练。")
            # 新建：询问DLC
            use_dlc = input("Enable DLC pieces for training? (y/n) [n]: ").strip().lower() == 'y'
            trainer = AITrainer(force_new=True, use_dlc=use_dlc)
        else:
            # 列出历史模型目录
            print("可用历史AI模型：")
            for idx, d in enumerate(model_dirs):
                print(f"{idx}: {d}")
            sel = input("请输入要继续训练的AI目录编号（回车默认最新）:").strip()
            if sel == '':
                model_dir = model_dirs[-1]
            else:
                try:
                    model_dir = model_dirs[int(sel)]
                except Exception:
                    print("输入无效，使用最新目录。")
                    model_dir = model_dirs[-1]
            # 根据目录名后缀判断是否 DLC
            suffix = os.path.basename(model_dir).split('_')[-1]
            use_dlc = (suffix == 'dlc')
            # 解析 run_prefix
            npz_files = sorted(glob.glob(os.path.join(model_dir, "*_final.npz")))
            if not npz_files:
                print("该目录下无模型文件，切换到新建训练。")
                trainer = AITrainer(force_new=True, use_dlc=use_dlc)
            else:
                latest_npz = os.path.basename(npz_files[-1])
                run_prefix = latest_npz.split("_final.npz")[0]
                trainer = AITrainer(custom_dir=model_dir, custom_prefix=run_prefix, use_dlc=use_dlc)
                print(f"[断点续训] 继续训练: {model_dir}, DLC={'启用' if use_dlc else '禁用'}")
    else:
        # 新建训练：询问DLC
        use_dlc = input("Enable DLC pieces for training? (y/n) [n]: ").strip().lower() == 'y'
        trainer = AITrainer(force_new=True, use_dlc=use_dlc)
    print("\n训练将无限进行，请随时按 Ctrl+C 终止并自动保存断点。\n")
    trainer.train()
    input("\nAI training complete. Press Enter to return to main menu...")

def ai_evaluation_loop():
    print("\n--- AI Evaluation ---")
    # DLC option for evaluation
    use_dlc = input("Enable DLC pieces for evaluation? (y/n) [n]: ").strip().lower() == 'y'
    evaluator = AIEvaluator(use_dlc=use_dlc)
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



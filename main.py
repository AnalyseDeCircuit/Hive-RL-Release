# main.py

from game import Game
from player import Player
from ai_player import AIPlayer
from ai_trainer import AITrainer
import json  # 用于 ensemble 配置保存
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
    print("\033[97m4. Evaluate AI & Plots\033[0m")
    print("\033[97m5. Real-time Training Monitor\033[0m")  # 新增实时监控选项
    
    # 亮红色退出选项
    print("\033[91m6. Exit Game\033[0m")
    
    # 青色分隔线
    print("\033[96m" + "═" * 24 + "\033[0m")
    
    # 黄色输入提示
    return input("\033[93m› Choose an option (1-6): \033[0m").strip()

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
                            # 放置棋子：内部循环捕获坐标输入错误，不跳出action循环
                            while True:
                                coord = input("Enter coordinates (x y) and piece type (0-7): ").split()
                                if len(coord) != 3:
                                    print("Please enter exactly three numbers: x y piece_type.")
                                    continue
                                try:
                                    x, y, piece_type = map(int, coord)
                                except ValueError:
                                    print("Invalid numbers. Please enter valid integers.")
                                    continue
                                if piece_type not in PIECE_TYPE_LIST:
                                    print("Invalid piece type. Choose 0-7.")
                                    continue
                                break
                            # 执行放置
                            try:
                                current_player.place_piece(game.board, x, y, piece_type, game.turn_count)
                                game.switch_player()
                                valid_input = True
                            except Exception as e:
                                print(f"Invalid move: {e}")
                                # 放置失败后返回action选择
                            
                        elif action == 'M':
                            # 移动棋子：内部循环捕获坐标输入错误
                            while True:
                                parts = input("Enter from coordinates (x y) and to coordinates (toX toY): ").split()
                                if len(parts) != 4:
                                    print("Please enter exactly four numbers: fromX fromY toX toY.")
                                    continue
                                try:
                                    from_x, from_y, to_x, to_y = map(int, parts)
                                except ValueError:
                                    print("Invalid numbers. Please enter valid integers.")
                                    continue
                                break
                            # 检查栈顶棋子归属
                            moved_piece = game.board.get_piece_at(from_x, from_y)
                            if moved_piece is None or moved_piece.get_owner() != current_player:
                                print("Invalid move: no your piece at the source coordinates.")
                                continue
                            piece_type = moved_piece.get_piece_type()
                            try:
                                current_player.move_piece(game.board, from_x, from_y, to_x, to_y, piece_type)  # type: ignore
                                game.switch_player()
                                valid_input = True
                            except Exception as e:
                                print(f"Invalid move: {e}")
                                # 移动失败后返回action选择
                            
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
        # 断点续训逻辑
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

    # 训练模式子菜单：选择训练类型
    print("\n请选择训练类型：")
    print("1. 并行采样基础训练 (Parallel Sampling)")
    print("2. 自我对弈精炼训练 (Self-Play)")
    print("3. Ensemble 投票训练 (训练多模型用于投票)")
    print("4. 对抗式鲁棒化训练 (Adversarial Training)")
    print("5. 课程学习 (Curriculum Learning)")
    train_type = input("输入选项(1-5，回车默认1): ").strip()
    if train_type == '2':
        # 自我对弈训练
        episodes = input("输入自我对弈训练局数(回车默认10000): ").strip()
        num_eps = int(episodes) if episodes.isdigit() else 10000
        trainer.self_play_train(num_episodes=num_eps)
    elif train_type == '3':
        # Ensemble 投票训练
        count = input("输入投票模型数量 N (回车默认3): ").strip()
        N = int(count) if count.isdigit() else 3
        ensemble_paths = []
        for i in range(N):
            print(f"开始训练第 {i+1}/{N} 个模型...")
            sub_trainer = AITrainer(force_new=True, use_dlc=use_dlc)
            sub_trainer.train()
            path = os.path.join(sub_trainer.model_dir, f"{sub_trainer.run_prefix}_final.npz")
            ensemble_paths.append(path)
        # 保存 ensemble 配置
        cfg_path = os.path.join("models", "ensemble.json")
        with open(cfg_path, 'w') as f:
            json.dump(ensemble_paths, f)
        print(f"Ensemble 配置已保存到 {cfg_path}")
    elif train_type == '4':
        # 对抗式鲁棒化训练
        episodes = input("输入对抗训练局数(回车默认10000): ").strip()
        num_eps = int(episodes) if episodes.isdigit() else 10000
        trainer.adversarial_train(num_episodes=num_eps)
    elif train_type == '5':
        # 课程学习训练，无需指定局数，按 Ctrl+C 切换阶段
        print("课程学习模式：按 Ctrl+C 结束当前阶段进入下一阶段。")
        print("使用强化的信号处理机制，确保退出时模型被正确保存。")
        trainer.curriculum_train_with_signal_handling()
    else:
        # 并行采样基础训练
        trainer.train()
    input("\n训练结束，按回车返回主菜单...")
    return

def evaluate_menu():
    clear_screen()
    while True:
        clear_screen()
        print("\n--- Evaluate AI & Plots ---")
        print("1. Plot Reward Curve")
        print("2. Plot Loss Curve")
        print("3. Plot Win-Rate Curve")
        print("4. Plot Stats Curve")
        print("5. Back to Main Menu")
        choice = input("Choose an option (1-5): ").strip()
        if choice == '1':
            # Plot Reward Curve logic
            model_path = select_model()
            if model_path:
                model_dir = os.path.dirname(model_path)
                prefix = os.path.basename(model_path).split("_final.npz")[0]
                reward_file = os.path.join(model_dir, f"{prefix}_reward_history.npy")
                print(f"Running plot_reward_curve.py for {reward_file}...")
                os.system(f"python plot_reward_curve.py \"{reward_file}\"")
            else:
                print("No model selected.")
            input("Press Enter to continue...")
        elif choice == '2':
            # Plot Loss Curve logic
            model_path = select_model()
            if model_path:
                model_dir = os.path.dirname(model_path)
                prefix = os.path.basename(model_path).split("_final.npz")[0]
                loss_file = os.path.join(model_dir, f"{prefix}_loss_history.npy")
                print(f"Running plot_loss_curve.py for {loss_file}...")
                os.system(f"python plot_loss_curve.py \"{loss_file}\"")
            else:
                print("No model selected.")
            input("Press Enter to continue...")
        elif choice == '3':
            # Plot Win-Rate Curve logic
            model_path = select_model()
            if model_path:
                model_dir = os.path.dirname(model_path)
                prefix = os.path.basename(model_path).split("_final.npz")[0]
                end_stats_file = os.path.join(model_dir, f"{prefix}_end_stats_history.npy")
                print(f"Running plot_win_rate_curve.py for {end_stats_file}...")
                os.system(f"python plot_win_rate_curve.py \"{end_stats_file}\"")
            else:
                print("No model selected.")
            input("Press Enter to continue...")
        elif choice == '4':
            # Plot Stats Curve logic
            model_path = select_model()
            if model_path:
                model_dir = os.path.dirname(model_path)
                prefix = os.path.basename(model_path).split("_final.npz")[0]
                print(f"Running plot_stats_curve.py for stats in {model_dir}...")
                os.system(f"python plot_stats_curve.py \"{model_dir}\"")
            else:
                print("No model selected.")
            input("Press Enter to continue...")
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please choose a number between 1 and 5.")
            input("Press Enter to continue...")

def real_time_monitor_menu():
    """实时训练监控菜单"""
    clear_screen()
    print("\033[96m" + "═" * 32 + "\033[0m")
    print("\033[93m📊 实时训练监控\033[0m")
    print("\033[96m" + "═" * 32 + "\033[0m")
    
    try:
        # 检查是否有训练数据
        if not os.path.exists("models"):
            print("\033[91m❌ 未找到models目录！\033[0m")
            print("请先开始AI训练")
            input("\nPress Enter to return...")
            return
        
        # 查找是否有奖励文件
        import glob
        reward_files = glob.glob("models/*/DQN_reward_history.npy") + glob.glob("models/*/*_reward_history.npy")
        
        if not reward_files:
            print("\033[91m❌ 未找到训练数据文件！\033[0m")
            print("请确保:")
            print("  1. 已开始AI训练")
            print("  2. 训练至少完成了几个episodes")
            print("  3. models/目录下有*_reward_history.npy文件")
            print("")
            print("支持的文件路径:")
            print("  - models/*/DQN_reward_history.npy")
            print("  - models/*/*_reward_history.npy")
            input("\nPress Enter to return...")
            return
        
        # 显示找到的文件
        print(f"\033[92m✅ 找到 {len(reward_files)} 个训练文件\033[0m")
        latest_file = max(reward_files, key=lambda x: os.path.getmtime(x))
        print(f"最新文件: {latest_file}")
        print("")
        
        # 尝试导入简化版监控器
        try:
            from start_monitor import SimpleRealTimeMonitor
            
            print("\033[97m正在启动简化版实时监控...\033[0m")
            print("\033[90m提示: 这是一个简化版，专门为训练过程设计\033[0m")
            print("")
            
            # 创建监控器
            monitor = SimpleRealTimeMonitor(update_interval=5)  # 5秒更新一次
            
            print("\033[92m✅ 监控器已启动\033[0m")
            print("\033[90m使用提示:")
            print("  - 自动每5秒更新数据")
            print("  - 关闭图表窗口退出监控")
            print("  - 支持实时显示奖励曲线和统计\033[0m")
            print("")
            
            # 启动监控
            monitor.start()
            
        except ImportError:
            # 尝试导入完整版监控器
            try:
                from real_time_monitor import RealTimeTrainingMonitor
                
                print("\033[97m正在启动完整版实时监控...\033[0m")
                print("\033[90m提示: 如果启动失败，请使用简化版\033[0m")
                print("")
                
                # 创建监控器
                monitor = RealTimeTrainingMonitor(update_interval=5)
                
                print("\033[92m✅ 监控器已启动\033[0m")
                print("\033[90m快捷键提示:")
                print("  R - 重置视图")
                print("  S - 保存截图") 
                print("  Q - 退出监控")
                print("  关闭窗口 - 返回主菜单\033[0m")
                print("")
                
                # 启动监控
                monitor.start_monitoring()
                
            except ImportError as e:
                print(f"\033[91m❌ 导入监控模块失败: {e}\033[0m")
                print("请确保matplotlib等依赖已正确安装:")
                print("  pip install matplotlib numpy")
                
    except Exception as e:
        print(f"\033[91m❌ 启动监控失败: {e}\033[0m")
        print("建议:")
        print("  1. 确保训练正在进行")
        print("  2. 检查文件权限")
        print("  3. 重启程序重试")
    
    input("\nPress Enter to return to main menu...")

def main():
    game = Game.get_instance()
    
    while True:
        main_menu_choice = display_main_menu()

        if main_menu_choice == '1':
            player1_name, player2_name = get_player_names()
            # DLC option for Human vs Human
            use_dlc = input("Enable DLC pieces for both players? (y/n) [n]: ").strip().lower() == 'y'
            player1 = Player(player1_name, is_first_player=True, use_dlc=use_dlc)
            player2 = Player(player2_name, is_first_player=False, use_dlc=use_dlc)
            game.initialize_game(player1, player2)
            game_loop(game)
        elif main_menu_choice == '2':
            human_vs_ai_game_loop()
        elif main_menu_choice == '3':
            ai_training_loop()
        elif main_menu_choice == '4':
            evaluate_menu()
        elif main_menu_choice == '5':
            real_time_monitor_menu()
        elif main_menu_choice == '6':
            print("Exiting game. Goodbye!")
            break
        else:
            print("Invalid option. Please choose a number between 1 and 6.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()



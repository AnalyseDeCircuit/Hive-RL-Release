"""
evaluate.py

综合评估脚本：
1. 可视化训练曲线（Reward、Episode Steps、Illegal Actions、QueenBee Steps）
2. 对最新模型与基准策略（Random, Heuristic）批量对局，统计胜率、平均回合数
3. 持久化评估报告及图表
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import argparse
from hive_env import HiveEnv, Action
from ai_player import AIPlayer
from ai_evaluator import RandomPlayer

# 启发式玩家（与 benchmark 保持一致）
class HeuristicPlayer(RandomPlayer):
    def select_action(self, env, game_state, board, current_player_idx):
        legal = env.get_legal_actions()
        if not legal:
            return None
        # 强制第4回合放蜂后
        must = (env.turn_count == 3 and not getattr(env.game.player1 if current_player_idx==0 else env.game.player2, 'is_queen_bee_placed', False))
        if must:
            queens = [a for a in legal if Action.decode_action(a)[0]=='place' and Action.decode_action(a)[5]==0]
            if queens:
                return queens[0]
        # 前4步优先放蜂后
        if env.turn_count < 4:
            queens = [a for a in legal if Action.decode_action(a)[0]=='place' and Action.decode_action(a)[5]==0]
            if queens:
                return queens[0]
        return np.random.choice(legal)

# 读取 npy 训练统计并绘图
def plot_training_stats(models_dir, output_dir):
    files = glob.glob(os.path.join(models_dir, '*', '*_reward_history.npy'))
    for reward_file in files:
        prefix = os.path.basename(reward_file).split('_reward_history.npy')[0]
        base = os.path.dirname(reward_file)
        # 加载曲线
        rewards = np.load(reward_file, allow_pickle=True)
        steps = np.load(os.path.join(base, f'{prefix}_steps_history.npy'), allow_pickle=True)
        illegal = np.load(os.path.join(base, f'{prefix}_illegal_history.npy'), allow_pickle=True)
        queen_steps = np.load(os.path.join(base, f'{prefix}_queenbee_step_history.npy'), allow_pickle=True)
        # 绘图
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(); plt.plot(rewards); plt.title(f'{prefix} Reward'); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.savefig(os.path.join(output_dir, f'{prefix}_reward.png'))
        plt.figure(); plt.plot(steps); plt.title(f'{prefix} Episode Steps'); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.savefig(os.path.join(output_dir, f'{prefix}_steps.png'))
        plt.figure(); plt.plot(illegal); plt.title(f'{prefix} Illegal Actions'); plt.xlabel('Episode'); plt.ylabel('Count'); plt.savefig(os.path.join(output_dir, f'{prefix}_illegal.png'))
        plt.figure(); plt.plot(queen_steps); plt.title(f'{prefix} QueenBee Placement Step'); plt.xlabel('Episode'); plt.ylabel('Step'); plt.savefig(os.path.join(output_dir, f'{prefix}_queenbee.png'))
    print(f'Training stats plots saved to {output_dir}')

# 对局函数
def run_match(args):
    model_path, OppClass, match_id = args
    env = HiveEnv(training_mode=False)
    ai = AIPlayer.load_from_file(model_path)
    opp = OppClass('Opp', False, True)
    first = np.random.choice([0,1])
    env.reset(); env.current_player_idx = first
    if first==0: env.game.player1, env.game.player2 = ai, opp
    else:       env.game.player1, env.game.player2 = opp, ai
    term=False; trunc=False; turns=0
    obs, info = env.reset()
    while not term and not trunc:
        idx = env.current_player_idx
        player = ai if idx==first else opp
        action = player.select_action(env, env.game, env.board, idx)
        if action is None: break
        try:
            obs, _, term, trunc, _ = env.step(action)
        except Exception as e:
            # 非法动作视为对手获胜，终止对局
            winner_name = opp.get_name() if idx==first else ai.get_name()
            return (match_id, OppClass.__name__, winner_name, turns)
        turns +=1
    winner = env.game.get_winner()
    return (match_id, OppClass.__name__, winner.get_name() if winner else 'Draw', turns)

# 批量对战评估
def run_battle(model_path, opponents, num_games, procs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tasks = [(model_path, Opp, f'{Opp.__name__}_{i+1}') for Opp in opponents for i in range(num_games)]
    with Pool(procs) as pool:
        res = pool.map(run_match, tasks)
    # 保存 CSV
    csv_file = os.path.join(output_dir, 'evaluation_results.csv')
    import csv
    with open(csv_file,'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['match_id','opponent','winner','turns']); writer.writerows(res)
    print(f'Battle results saved to {csv_file}')
    # 控制台统计
    for Opp in opponents:
        recs = [r for r in res if r[1]==Opp.__name__]
        total=len(recs); wins=sum(1 for r in recs if r[2]=='AI'); draws=sum(1 for r in recs if r[2]=='Draw')
        avg_turns=sum(r[3] for r in recs)/total if total else 0
        print(f'{Opp.__name__}: Win {wins/total:.2%}, Draw {draws/total:.2%}, AvgTurns {avg_turns:.1f}')

# 主入口
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Hive-RL 综合评估')
    parser.add_argument('--plot_dir', type=str, default='eval_plots')
    parser.add_argument('--battle_out', type=str, default='eval_battle')
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--model', type=str, default=None, help='指定模型文件，否则自动最新')
    args = parser.parse_args()
    # 查找最新训练目录
    models_dir = 'models'
    if args.model:
        model_path = args.model
    else:
        # latest final model
        files = glob.glob(os.path.join(models_dir,'*','*_final.npz'))
        files.sort(key=os.path.getmtime)
        model_path = files[-1]
    print(f'Evaluating model: {model_path}')
    # 绘制训练曲线
    plot_training_stats(models_dir, args.plot_dir)
    # 批量对战
    run_battle(model_path, [RandomPlayer, HeuristicPlayer], args.games, args.procs, args.battle_out)

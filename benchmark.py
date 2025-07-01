#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark.py

自动化基准测试脚本：
1) 自动查找最新训练模型（models/*/*_final.npz）
2) 定义 RandomPlayer 和启发式 HeuristicPlayer
3) 并行批量运行对局，统计胜率、回合数
4) 输出 CSV 报表及控制台汇总
"""
import os
import glob
import argparse
import csv
import random
from multiprocessing import Pool
from hive_env import HiveEnv, Action
from ai_player import AIPlayer
from ai_evaluator import RandomPlayer


class HeuristicPlayer(RandomPlayer):
    """简单启发式玩家：优先放蜂后，否则随机"""
    def select_action(self, env, game_state, board, current_player_idx):
        legal = env.get_legal_actions()
        if not legal:
            return None
        # 强制第4回合放蜂后
        must = (env.turn_count == 3 and not getattr(env.game.player1 if current_player_idx==0 else env.game.player2, 'is_queen_bee_placed', False))
        if must:
            queens = [a for a in legal if Action.decode_action(a)[0]=='place' and Action.decode_action(a)[5]==0]
            if queens:
                return random.choice(queens)
        # 前4步优先放蜂后
        if env.turn_count < 4:
            queens = [a for a in legal if Action.decode_action(a)[0]=='place' and Action.decode_action(a)[5]==0]
            if queens:
                return random.choice(queens)
        return random.choice(legal)


def find_latest_model(models_dir='models'):
    """查找 models_dir 下最新 *_final.npz 模型文件"""
    pattern = os.path.join(models_dir, '*', '*_final.npz')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'未在 {models_dir} 中找到任何 *_final.npz 模型')
    files.sort(key=os.path.getmtime)
    return files[-1]


def run_match(args):
    model_path, OppClass, match_id = args
    env = HiveEnv(training_mode=False)
    # 加载模型
    ai = AIPlayer.load_from_file(model_path)
    opp = OppClass('Opp', False, True)
    # 随机先手
    first = random.choice([0,1])
    env.reset()
    env.current_player_idx = first
    # 设置游戏玩家顺序
    if first == 0:
        env.game.player1, env.game.player2 = ai, opp
    else:
        env.game.player1, env.game.player2 = opp, ai
    turns = 0
    terminated = False
    truncated = False
    obs, info = env.reset()
    while not terminated and not truncated:
        idx = env.current_player_idx
        player = ai if idx == first else opp
        action = player.select_action(env, env.game, env.board, idx)
        if action is None:
            break
        obs, reward, terminated, truncated, info = env.step(action)
        turns += 1
    winner = env.game.get_winner()
    winner_name = winner.get_name() if winner else 'Draw'
    return (match_id, OppClass.__name__, winner_name, turns)


def run_benchmark(model_path, opponents, num_games, processes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tasks = [(model_path, Opp, f'{Opp.__name__}_{i+1}') for Opp in opponents for i in range(num_games)]
    with Pool(processes=processes) as pool:
        results = pool.map(run_match, tasks)
    # 写入 CSV
    csv_file = os.path.join(output_dir, 'benchmark_results.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['match_id','opponent','winner','turns'])
        writer.writerows(results)
    # 控制台汇总
    print('Benchmark Summary:')
    for Opp in opponents:
        recs = [r for r in results if r[1]==Opp.__name__]
        total = len(recs)
        wins = sum(1 for r in recs if r[2] not in ('Draw','Opp'))
        draws = sum(1 for r in recs if r[2]=='Draw')
        avg_turns = sum(r[3] for r in recs) / total if total else 0
        print(f"  {Opp.__name__}: Win {wins/total:.2%}, Draw {draws/total:.2%}, AvgTurns {avg_turns:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hive-RL 基准测试')
    parser.add_argument('--model', type=str, help='模型文件路径 (*.npz)', default=None)
    parser.add_argument('--games', type=int, help='每个对手局数', default=500)
    parser.add_argument('--procs', type=int, help='并行进程数', default=4)
    parser.add_argument('--out', type=str, help='输出目录', default='benchmark_logs')
    args = parser.parse_args()
    model_path = args.model or find_latest_model('models')
    opponents = [RandomPlayer, HeuristicPlayer]
    print(f'Using model: {model_path}')
    run_benchmark(model_path, opponents, args.games, args.procs, args.out)

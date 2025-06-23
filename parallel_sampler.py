import multiprocessing as mp
from hive_env import HiveEnv, Action
from ai_player import AIPlayer
import numpy as np

def worker_process(queue, player_args, env_args, episode_per_worker=1):
    env = HiveEnv(**env_args)
    ai = AIPlayer(**player_args)
    while True:
        for _ in range(episode_per_worker):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            illegal_action_count = 0
            queenbee_step = -1
            current_player_idx = env.current_player_idx
            while not terminated and not truncated:
                legal_actions = env.get_legal_actions()
                # ---蜂后未落兜底---
                if not legal_actions:
                    # 检查蜂后是否未落，若未落则强制生成放蜂后动作
                    current_player = env.game.player1 if env.current_player_idx == 0 else env.game.player2
                    if not getattr(current_player, 'is_queen_bee_placed', False):
                        # 生成所有可放蜂后动作
                        queenbee_actions = []
                        for a in range(env.action_space.n):
                            try:
                                decoded = Action.decode_action(a)
                                if decoded[0] == 'place' and decoded[5] == 0:
                                    queenbee_actions.append(a)
                            except Exception:
                                continue
                        if queenbee_actions:
                            action = np.random.choice(queenbee_actions)
                        else:
                            action = None
                    else:
                        action = None
                    info = {'reason': 'no_legal_action'}
                    illegal_action_count += 1
                else:
                    action = ai.select_action(env, env.game, env.board, env.current_player_idx)
                # 记录蜂后落下的步数
                if queenbee_step == -1:
                    current_player = env.game.player1 if env.current_player_idx == 0 else env.game.player2
                    if getattr(current_player, 'is_queen_bee_placed', False):
                        queenbee_step = env.game.turn_count
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                # put 10元组，结构与主进程一致
                queue.put((obs, action, reward, next_obs, terminated, episode_reward, episode_steps, illegal_action_count, queenbee_step, info))
                obs = next_obs

# 主进程示例
if __name__ == '__main__':
    num_workers = 4
    queue = mp.Queue(maxsize=32)
    player_args = dict(name='AI_Parallel', is_first_player=True)
    env_args = dict(training_mode=True)
    workers = [mp.Process(target=worker_process, args=(queue, player_args, env_args)) for _ in range(num_workers)]
    for w in workers:
        w.daemon = True
        w.start()
    # 主进程收集经验
    replay_buffer = []
    while True:
        sample = queue.get()
        replay_buffer.append(sample)
        # 可在此处批量训练AIPlayer
        if len(replay_buffer) >= 128:
            # ...批量训练代码...
            replay_buffer = []

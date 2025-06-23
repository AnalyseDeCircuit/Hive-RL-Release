import multiprocessing as mp
from hive_env import HiveEnv
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
            episode = []
            while not terminated and not truncated:
                current_player_idx = env.current_player_idx
                action = ai.select_action(env, env.game, env.board, current_player_idx)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode.append((obs, action, reward, next_obs, terminated))
                obs = next_obs
            queue.put(episode)

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
        episode = queue.get()
        replay_buffer.extend(episode)
        # 可在此处批量训练AIPlayer
        if len(replay_buffer) >= 128:
            # ...批量训练代码...
            replay_buffer = []

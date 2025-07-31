import multiprocessing as mp
import signal
import sys
from hive_env import HiveEnv, Action
from ai_player import AIPlayer
import numpy as np

def worker_process(queue, player_args, env_args, episode_per_worker=1, reward_shaper_config=None, epsilon_sync_queue=None):
    """Worker进程函数，增强信号处理"""
    
    # 设置信号处理 - worker进程忽略SIGINT，让主进程处理
    def signal_handler(signum, frame):
        print(f"[Worker] 收到信号 {signum}，正在优雅退出...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 支持从 env_args 传入 reward shaping 函数
        shaping_func = env_args.pop('reward_shaping_func', None)
        env = HiveEnv(**env_args)
        
        # 修复：正确传递和创建reward_shaper
        if reward_shaper_config:
            try:
                from improved_reward_shaping import HiveRewardShaper
                phase = reward_shaper_config.get('phase', 'foundation')
                env.reward_shaper = HiveRewardShaper(phase)
                print(f"[Worker] 成功加载奖励整形器: {phase}")
            except ImportError:
                print("[Worker] 奖励整形模块未找到，使用原始奖励")
        
        # 如果传入 shaping_func，则包装 step (向后兼容)
        if shaping_func is not None:
            original_step = env.step
            def shaped_step(action):
                obs, reward, terminated, truncated, info = original_step(action)
                return obs, shaping_func(reward, terminated), terminated, truncated, info
            env.step = shaped_step
        ai = AIPlayer(**player_args)
        
        # 修复：添加episode计数器限制，防止无限循环
        episode_count = 0
        while episode_count < episode_per_worker:  # 修复：使用计数器限制
            # 检查epsilon同步队列
            if epsilon_sync_queue is not None:
                try:
                    # 非阻塞检查是否有新的epsilon值
                    new_epsilon = epsilon_sync_queue.get_nowait()
                    ai.epsilon = new_epsilon
                    print(f"[Worker] 更新epsilon: {new_epsilon:.4f}")
                except:
                    # 队列为空，继续使用当前epsilon
                    pass
        
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
                        # 避免静态类型报错，动态获取 action_space.n
                        total_actions = getattr(env.action_space, 'n', 0)
                        for a in range(total_actions):
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
                
                # 只在episode结束时put数据，大幅提升性能
                # 同时，worker不进行训练，训练留给主进程统一处理
                if terminated or truncated:
                    # put 10元组，只在episode结束时发送
                    queue.put((obs, action, reward, next_obs, terminated, episode_reward, episode_steps, illegal_action_count, queenbee_step, info))
                
                obs = next_obs
        
        # 完成一个episode，增加计数器
        episode_count += 1
    
    except Exception as e:
        print(f"[Worker] 发生异常: {e}")
        print("[Worker] 正在退出...")
    finally:
        print("[Worker] Worker进程正常退出")

# 主进程示例
if __name__ == '__main__':
    num_workers = 4
    queue = mp.Queue(maxsize=32)
    player_args = dict(name='AI_Parallel', is_first_player=True)
    env_args = dict(training_mode=True)
    workers = [mp.Process(target=worker_process, args=(queue, player_args, env_args, 10, None, None)) for _ in range(num_workers)]
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

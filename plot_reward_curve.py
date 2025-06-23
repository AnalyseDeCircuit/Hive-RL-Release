import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

# 兼容中文路径和中文标签
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def find_latest_reward_file():
    # 查找 models/*/xxx_reward_history.npy，按时间排序取最新
    candidates = glob.glob(os.path.join("models", "*", "*_reward_history.npy"))
    if not candidates:
        # 兼容旧格式
        if os.path.exists("reward_history.npy"):
            return "reward_history.npy"
        else:
            print("[ERROR] 未找到任何 reward_history.npy 文件！")
            sys.exit(1)
    candidates.sort()
    return candidates[-1]

def find_latest_model_file():
    # 查找 models/*/xxx_final.npz 或 ep最大值的npz
    candidates = glob.glob(os.path.join("models", "*", "*_final.npz"))
    if not candidates:
        # 退而求其次，找ep最大值的npz
        candidates = glob.glob(os.path.join("models", "*", "*_ep*.npz"))
        if not candidates:
            print("[WARN] 未找到任何模型文件！")
            return None
        candidates.sort()
        return candidates[-1]
    candidates.sort()
    return candidates[-1]

# 支持命令行参数 --window N
window = 100
reward_file = None
for i, arg in enumerate(sys.argv[1:]):
    if arg.startswith('--window'):
        try:
            window = int(sys.argv[i+2])
        except Exception:
            print("[WARN] --window 参数无效，使用默认100")
    elif not arg.startswith('-') and reward_file is None:
        reward_file = arg
if reward_file is None:
    reward_file = find_latest_reward_file()

print(f"[INFO] 加载reward曲线文件: {reward_file}")
rewards = np.load(reward_file)
rewards = np.squeeze(rewards)  # 兼容shape为(N,1)或(1,N)

# 自动查找并输出最新模型文件
model_file = find_latest_model_file()
if model_file:
    print(f"[INFO] 最新模型文件: {model_file}")

plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Episode Reward", alpha=0.5)

# 计算滑动平均
if len(rewards) >= window:
    avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), avg, label=f"Moving Avg({window})", color='red')
else:
    print(f"[WARN] 奖励曲线长度不足{window}，不绘制滑动平均")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Hive-RL Training Reward Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

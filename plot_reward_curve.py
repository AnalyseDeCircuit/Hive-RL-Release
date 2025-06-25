import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import re

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

# 自动查找同目录下的统计文件
reward_dir = os.path.dirname(reward_file)
reward_prefix = re.sub(r'_reward_history.*$', '', os.path.basename(reward_file))
steps_file = os.path.join(reward_dir, f"{reward_prefix}_steps_history.npy")
illegal_file = os.path.join(reward_dir, f"{reward_prefix}_illegal_history.npy")
queenbee_file = os.path.join(reward_dir, f"{reward_prefix}_queenbee_step_history.npy")
end_stats_file = os.path.join(reward_dir, f"{reward_prefix}_end_stats_history.npy")

# 加载统计数据
steps = np.load(steps_file) if os.path.exists(steps_file) else None
illegal = np.load(illegal_file) if os.path.exists(illegal_file) else None
queenbee = np.load(queenbee_file) if os.path.exists(queenbee_file) else None
end_stats = np.load(end_stats_file, allow_pickle=True) if os.path.exists(end_stats_file) else None

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 修改为1行3列，以绘制Reward、End Stats和分布图
# 1. Reward 曲线
ax = axes[0]
ax.plot(rewards, label="Episode Reward", alpha=0.5)
if len(rewards) >= window:
    avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(rewards)), avg, label=f"Moving Avg({window})", color='red')

# 添加累计平均奖励曲线
cumsum = np.cumsum(rewards)
cumavg = cumsum / (np.arange(len(rewards)) + 1)
ax.plot(cumavg, label="Cumulative Avg Reward", color='green', alpha=0.7)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("Reward Curve")
ax.legend()
ax.grid(True)

# 2. 终局统计累计折线图
ax = axes[1]
if end_stats is not None and len(end_stats) > 0:
    labels = list(end_stats[0].keys())
    data = {k: [d[k] for d in end_stats] for k in labels}
    x = np.arange(len(end_stats))  # 每集一采样，与 reward 对齐
    x = np.insert(x, 0, 0)  # 首点补0
    for k in labels:
        cumsum = np.cumsum(data[k])
        cumsum = np.insert(cumsum, 0, 0)  # 首点补0
        ax.plot(x, cumsum, label=k)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Count")
    ax.set_title("End State Statistics (Cumulative)")
    ax.legend()
    ax.grid(True)
else:
    ax.set_visible(False)

# 3. reward分布直方图（异常分析）
ax = axes[2]  # 修改为一行三列中的第三个子图
if rewards is not None and len(rewards) > 0:
    REWARD_ABS_LIMIT = 200
    bins = np.linspace(min(-REWARD_ABS_LIMIT, np.min(rewards)), max(REWARD_ABS_LIMIT, np.max(rewards)), 41)
    ax.hist(rewards, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(REWARD_ABS_LIMIT, color='red', linestyle='--', label=f'+{REWARD_ABS_LIMIT}上限')
    ax.axvline(-REWARD_ABS_LIMIT, color='red', linestyle='--', label=f'-{REWARD_ABS_LIMIT}下限')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution (Histogram)')
    ax.legend()
    ax.grid(True)
else:
    ax.text(0.5, 0.5, "No reward data", ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()

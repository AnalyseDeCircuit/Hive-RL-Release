import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 兼容中文路径和中文标签
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 支持命令行参数：第一个位置参数为 end_stats_history.npy 文件路径
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
    end_stats_file = sys.argv[1]
else:
    # 自动查找最新文件
    candidates = []
    for root, _, files in os.walk('models'):
        for f in files:
            if f.endswith('_end_stats_history.npy'):
                candidates.append(os.path.join(root, f))
    if not candidates:
        print('[ERROR] 未找到任何 end_stats_history.npy 文件！')
        sys.exit(1)
    candidates.sort()
    end_stats_file = candidates[-1]

print(f"[INFO] 加载 end stats 文件: {end_stats_file}")
end_stats = np.load(end_stats_file, allow_pickle=True)
# 转换为列表
try:
    stats_list = list(end_stats)
except Exception:
    stats_list = end_stats.tolist()

# 提取每局胜利记录
wins = [s.get('win', 0) for s in stats_list]
wins = np.array(wins)
episodes = np.arange(1, len(wins) + 1)
# 累计胜率 (%)
cum_wins = np.cumsum(wins)
cum_win_rate = cum_wins / episodes * 100

# 绘图
plt.figure(figsize=(12,6))
plt.plot(episodes, cum_win_rate, label='Cumulative Win Rate (%)', color='blue')
plt.xlabel('Episode')
plt.ylabel('Win Rate (%)')
plt.title('Win Rate Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

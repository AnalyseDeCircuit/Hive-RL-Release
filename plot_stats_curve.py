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

# 支持命令行参数：第一个位置参数为 prefix 或完整模型目录
arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None
if arg and os.path.isdir(arg):
    model_dir = arg
    prefix = os.path.basename(arg)
elif arg:
    prefix = arg
    model_dir = os.path.join('models', prefix)
else:
    # 自动查找最新前缀
    candidates = glob.glob(os.path.join('models', '*', '*_steps_history.npy'))
    if not candidates:
        print('[ERROR] 未找到任何 steps_history.npy 文件！')
        sys.exit(1)
    # 提取前缀
    prefixes = [os.path.basename(f).split('_steps_history.npy')[0] for f in candidates]
    prefixes = sorted(set(prefixes))
    prefix = prefixes[-1]
    model_dir = os.path.join('models', prefix)

# 构造文件路径
steps_file = os.path.join(model_dir, f"{prefix}_steps_history.npy")
illegal_file = os.path.join(model_dir, f"{prefix}_illegal_history.npy")
queen_file = os.path.join(model_dir, f"{prefix}_queenbee_step_history.npy")

# 加载数据
steps = np.load(steps_file) if os.path.exists(steps_file) else None
illegal = np.load(illegal_file) if os.path.exists(illegal_file) else None
queen = np.load(queen_file) if os.path.exists(queen_file) else None

# 绘图
plt.figure(figsize=(12, 6))
plotted = False
if steps is not None and len(steps) > 0:
    plt.plot(np.arange(len(steps)), steps, label='Steps per Episode', alpha=0.5)
    plotted = True
if illegal is not None and len(illegal) > 0:
    plt.plot(np.arange(len(illegal)), illegal, label='Illegal Actions per Episode', alpha=0.5)
    plotted = True
if queen is not None and len(queen) > 0:
    plt.plot(np.arange(len(queen)), queen, label='QueenBee Place Step', alpha=0.5)
    plotted = True
if not plotted:
    print('[WARN] 没有可用的统计数据！')
    sys.exit(0)

plt.xlabel('Episode')
plt.ylabel('Count')
plt.title('Statistics per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
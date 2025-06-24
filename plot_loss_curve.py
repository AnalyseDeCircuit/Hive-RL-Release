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

def find_latest_loss_file():
    # 查找 models/*/xxx_loss_history.npy，按时间排序取最新
    candidates = glob.glob(os.path.join("models", "*", "*_loss_history.npy"))
    if not candidates:
        print("[ERROR] 未找到任何 loss_history.npy 文件！")
        sys.exit(1)
    candidates.sort()
    return candidates[-1]

# 支持命令行参数 --window N
window = 100
loss_file = None
for i, arg in enumerate(sys.argv[1:]):
    if arg.startswith('--window'):
        try:
            window = int(sys.argv[i+2])
        except Exception:
            print("[WARN] --window 参数无效，使用默认100")
    elif not arg.startswith('-') and loss_file is None:
        loss_file = arg
if loss_file is None:
    loss_file = find_latest_loss_file()

print(f"[INFO] 加载loss曲线文件: {loss_file}")
losses = np.load(loss_file)
losses = np.squeeze(losses)

plt.figure(figsize=(12,6))
plt.plot(losses, label="Batch Loss", alpha=0.5)
if len(losses) >= window:
    avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(losses)), avg, label=f"Moving Avg({window})", color='red')
plt.xlabel("Batch/Step")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

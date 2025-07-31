#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
start_monitor.py

独立的实时训练监控启动脚本
用法: python start_monitor.py
"""

import sys
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 兼容中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class SimpleRealTimeMonitor:
    """简化的实时监控器"""
    
    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.rewards = []
        self.episodes = []
        self.moving_avg = []
        self.last_update = 0
        
    def find_latest_reward_file(self):
        """查找最新的奖励文件"""
        patterns = [
            "models/*/DQN_reward_history.npy",
            "models/*/*_reward_history.npy", 
            "models/*reward_history.npy",
            "reward_history.npy"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # 按修改时间排序，取最新的
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                print(f"[Monitor] 找到奖励文件: {files[0]}")
                return files[0]
        
        print("[Monitor] 未找到奖励文件，支持的文件模式:")
        for pattern in patterns:
            print(f"  - {pattern}")
        return None
    
    def load_data(self):
        """加载训练数据"""
        reward_file = self.find_latest_reward_file()
        if not reward_file:
            return False
            
        try:
            # 检查文件是否更新
            mtime = os.path.getmtime(reward_file)
            if mtime <= self.last_update:
                return False
                
            self.last_update = mtime
            
            # 加载数据
            rewards = np.load(reward_file)
            if len(rewards) == len(self.rewards):
                return False  # 没有新数据
                
            self.rewards = rewards.tolist()
            self.episodes = list(range(1, len(self.rewards) + 1))
            
            # 计算移动平均
            window = min(50, len(self.rewards))
            if len(self.rewards) >= window:
                self.moving_avg = []
                for i in range(len(self.rewards)):
                    start_idx = max(0, i - window + 1)
                    avg = np.mean(self.rewards[start_idx:i+1])
                    self.moving_avg.append(avg)
            
            print(f"[Monitor] 数据已更新: {len(self.rewards)} episodes, 最新奖励: {self.rewards[-1]:.3f}")
            return True
            
        except Exception as e:
            print(f"[Monitor] 加载数据失败: {e}")
            return False
    
    def setup_plot(self):
        """设置图表"""
        plt.ion()  # 开启交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Hive-RL 实时训练监控', fontsize=14, fontweight='bold')
        
        # 奖励曲线
        self.ax1.set_title('奖励曲线')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', alpha=0.5, label='原始奖励')
        self.line2, = self.ax1.plot([], [], 'r-', linewidth=2, label='移动平均(50)')
        self.ax1.legend()
        
        # 统计信息
        self.ax2.set_title('训练统计')
        self.ax2.axis('off')
        self.stats_text = self.ax2.text(0.1, 0.9, '', fontsize=11, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
    def update_plot(self):
        """更新图表"""
        if not self.episodes:
            return
            
        # 更新奖励曲线
        self.line1.set_data(self.episodes, self.rewards)
        if self.moving_avg:
            self.line2.set_data(self.episodes, self.moving_avg)
        
        # 自动调整范围
        if self.rewards:
            self.ax1.set_xlim(0, max(len(self.episodes), 100))
            y_min, y_max = min(self.rewards), max(self.rewards)
            y_range = y_max - y_min if y_max != y_min else 1
            self.ax1.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # 更新统计信息
        self.update_stats()
        
        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_stats(self):
        """更新统计信息"""
        if not self.rewards:
            return
            
        total_episodes = len(self.rewards)
        recent_count = min(100, total_episodes)
        recent_rewards = self.rewards[-recent_count:]
        
        current_reward = self.rewards[-1]
        avg_reward = np.mean(recent_rewards)
        best_reward = max(self.rewards)
        worst_reward = min(self.rewards)
        
        stats_text = f"""训练进度统计

总Episodes: {total_episodes:,}

奖励统计 (最近{recent_count}局):
  当前: {current_reward:.3f}
  平均: {avg_reward:.3f}
  最佳: {best_reward:.3f}
  最差: {worst_reward:.3f}

使用提示:
  - 关闭窗口退出监控
  - 每{self.update_interval}秒自动更新
  - 数据来源: 最新训练文件"""
        
        self.stats_text.set_text(stats_text)
    
    def start(self):
        """开始监控"""
        print("🚀 启动Hive-RL实时训练监控...")
        print("📁 查找训练文件...")
        
        # 检查是否有训练数据
        if not self.find_latest_reward_file():
            print("❌ 未找到训练数据文件！")
            print("请确保:")
            print("  1. 已开始AI训练")
            print("  2. models/目录下有*_reward_history.npy文件")
            return
        
        print("✅ 找到训练数据，设置监控界面...")
        
        # 设置图表
        self.setup_plot()
        
        # 初始加载
        if self.load_data():
            self.update_plot()
        
        print("📊 监控界面已启动")
        print(f"🔄 每{self.update_interval}秒自动更新")
        print("💡 关闭图表窗口可退出监控")
        
        # 主循环
        try:
            while plt.get_fignums():  # 检查窗口是否还在
                if self.load_data():
                    self.update_plot()
                    print(f"📈 数据已更新 (Episodes: {len(self.episodes)})")
                
                # 等待更新间隔
                plt.pause(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断，退出监控")
        except Exception as e:
            print(f"❌ 监控过程中出错: {e}")
        finally:
            plt.close('all')
            print("👋 监控已退出")


def main():
    """主函数"""
    print("=" * 50)
    print("🎮 Hive-RL 实时训练监控工具")
    print("=" * 50)
    
    # 创建监控器
    monitor = SimpleRealTimeMonitor(update_interval=5)
    
    # 开始监控
    monitor.start()


if __name__ == "__main__":
    main()

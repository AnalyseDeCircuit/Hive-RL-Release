#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
real_time_monitor.py

实时训练监控工具 - 在训练过程中实时显示奖励曲线和性能指标
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import time
import json
from typing import Optional, List, Dict
import threading
import queue

# 兼容中文路径和中文标签
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

class RealTimeTrainingMonitor:
    """实时训练监控器"""
    
    def __init__(self, model_dir: str = "models", update_interval: int = 5):
        self.model_dir = model_dir
        self.update_interval = update_interval  # 更新间隔(秒)
        
        # 数据存储
        self.episodes = []
        self.rewards = []
        self.moving_avg_rewards = []
        self.loss_history = []
        self.win_rates = []
        self.epsilon_history = []
        
        # 训练统计
        self.current_phase = "Unknown"
        self.total_episodes = 0
        self.training_time = 0
        
        # 图形界面
        self.fig = None
        self.axes = None
        self.lines = {}
        self.stats_text = None
        
        # 文件监控
        self.last_modified_times = {}
        
    def find_latest_training_files(self) -> Dict[str, Optional[str]]:
        """查找最新的训练文件"""
        files: Dict[str, Optional[str]] = {
            'reward': None,
            'loss': None,
            'meta': None,
            'end_stats': None
        }
        
        # 查找最新的训练目录
        pattern = os.path.join(self.model_dir, "*/")
        dirs = glob.glob(pattern)
        if not dirs:
            return files
            
        # 按时间排序，取最新的
        dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_dir = dirs[0]
        
        # 查找各种文件
        for file_type in ['reward_history', 'loss_history', 'meta', 'end_stats_history']:
            pattern = os.path.join(latest_dir, f"*_{file_type}.npy")
            if file_type == 'meta':
                pattern = os.path.join(latest_dir, f"*_meta.json")
            
            candidates = glob.glob(pattern)
            if candidates:
                candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                if file_type == 'reward_history':
                    files['reward'] = candidates[0]
                elif file_type == 'loss_history':
                    files['loss'] = candidates[0]
                elif file_type == 'meta':
                    files['meta'] = candidates[0]
                elif file_type == 'end_stats_history':
                    files['end_stats'] = candidates[0]
        
        return files
    
    def load_training_data(self) -> bool:
        """加载训练数据"""
        files = self.find_latest_training_files()
        
        updated = False
        
        # 加载奖励历史 - 主要数据源
        if files['reward'] and self._is_file_updated(files['reward']):
            try:
                rewards = np.load(files['reward'])
                if len(rewards) > len(self.rewards):
                    self.rewards = rewards.tolist()
                    self.episodes = list(range(1, len(self.rewards) + 1))
                    
                    # 计算移动平均
                    window = min(100, len(self.rewards))
                    if len(self.rewards) >= window:
                        self.moving_avg_rewards = []
                        for i in range(len(self.rewards)):
                            start_idx = max(0, i - window + 1)
                            avg = np.mean(self.rewards[start_idx:i+1])
                            self.moving_avg_rewards.append(avg)
                    
                    # 基于episode数量估算epsilon历史（训练过程中的合理估算）
                    self._estimate_epsilon_history()
                    
                    updated = True
                    print(f"[Monitor] 奖励数据已更新: {len(self.rewards)} episodes")
            except Exception as e:
                print(f"[Monitor] 加载奖励数据失败: {e}")
        
        # 加载损失历史 - 可选数据源
        if files['loss'] and self._is_file_updated(files['loss']):
            try:
                loss_data = np.load(files['loss'])
                if len(loss_data) > len(self.loss_history):
                    self.loss_history = loss_data.tolist()
                    updated = True
                    print(f"[Monitor] 损失数据已更新: {len(self.loss_history)} steps")
            except Exception as e:
                print(f"[Monitor] 加载损失数据失败: {e}")
        
        # 尝试加载元数据 - 完全可选，失败不影响监控
        if files['meta'] and self._is_file_updated(files['meta']):
            try:
                with open(files['meta'], 'r') as f:
                    meta = json.load(f)
                    if 'player1_epsilon' in meta:
                        # 如果有实际的epsilon数据，更新估算
                        current_epsilon = meta['player1_epsilon']
                        print(f"[Monitor] 元数据已更新: epsilon={current_epsilon:.3f}")
                    updated = True
            except Exception as e:
                # 训练过程中meta文件可能不存在，这是正常的
                pass
        
        return updated
    
    def _estimate_epsilon_history(self):
        """基于episode数量估算epsilon历史"""
        if not self.episodes:
            return
            
        total_episodes = len(self.episodes)
        
        # 基于典型的课程学习参数估算epsilon衰减
        # 这些是基于修复后的课程学习配置的合理估算
        if total_episodes <= 35000:
            # Foundation阶段: 0.9 -> 0.75
            start_eps, end_eps = 0.9, 0.75
            decay_episodes = min(10000, total_episodes)
        elif total_episodes <= 80000:  # 35000 + 45000
            # Strategy阶段: 0.75 -> 0.35
            foundation_episodes = 35000
            strategy_episodes = total_episodes - foundation_episodes
            decay_episodes = min(15000, strategy_episodes)
            
            # Foundation部分保持0.75，Strategy部分衰减
            foundation_part = [0.75] * foundation_episodes
            if strategy_episodes <= decay_episodes:
                progress = strategy_episodes / decay_episodes
                current_eps = 0.75 * (1 - progress) + 0.35 * progress
                strategy_part = np.linspace(0.75, current_eps, strategy_episodes).tolist()
            else:
                strategy_part = (np.linspace(0.75, 0.35, decay_episodes).tolist() + 
                               [0.35] * (strategy_episodes - decay_episodes))
            
            self.epsilon_history = foundation_part + strategy_part
            return
        else:
            # Mastery阶段: 0.35 -> 0.05
            foundation_episodes = 35000
            strategy_episodes = 45000
            mastery_episodes = total_episodes - foundation_episodes - strategy_episodes
            decay_episodes = min(12000, mastery_episodes)
            
            # 组合所有阶段
            foundation_part = [0.75] * foundation_episodes
            strategy_part = np.linspace(0.75, 0.35, 15000).tolist() + [0.35] * (strategy_episodes - 15000)
            
            if mastery_episodes <= decay_episodes:
                progress = mastery_episodes / decay_episodes
                current_eps = 0.35 * (1 - progress) + 0.05 * progress
                mastery_part = np.linspace(0.35, current_eps, mastery_episodes).tolist()
            else:
                mastery_part = (np.linspace(0.35, 0.05, decay_episodes).tolist() + 
                              [0.05] * (mastery_episodes - decay_episodes))
            
            self.epsilon_history = foundation_part + strategy_part + mastery_part
            return
        
        # 简单线性衰减（用于单阶段训练或其他情况）
        if total_episodes <= 10000:
            decay_episodes = total_episodes
        else:
            decay_episodes = 10000
            
        if total_episodes <= decay_episodes:
            progress = total_episodes / decay_episodes
            current_eps = start_eps * (1 - progress) + end_eps * progress
            self.epsilon_history = np.linspace(start_eps, current_eps, total_episodes).tolist()
        else:
            decay_part = np.linspace(start_eps, end_eps, decay_episodes).tolist()
            constant_part = [end_eps] * (total_episodes - decay_episodes)
            self.epsilon_history = decay_part + constant_part
    
    def _is_file_updated(self, filepath: str) -> bool:
        """检查文件是否更新"""
        try:
            mtime = os.path.getmtime(filepath)
            if filepath not in self.last_modified_times or mtime > self.last_modified_times[filepath]:
                self.last_modified_times[filepath] = mtime
                return True
        except OSError:
            pass
        return False
    
    def setup_plots(self):
        """设置图形界面"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Hive-RL 实时训练监控', fontsize=16, fontweight='bold')
        
        # 奖励曲线 (左上)
        ax_reward = self.axes[0, 0]
        ax_reward.set_title('奖励曲线')
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Reward')
        ax_reward.grid(True, alpha=0.3)
        self.lines['reward'], = ax_reward.plot([], [], 'b-', alpha=0.3, label='原始奖励')
        self.lines['reward_avg'], = ax_reward.plot([], [], 'r-', linewidth=2, label='移动平均(100)')
        ax_reward.legend()
        
        # 损失曲线 (右上)
        ax_loss = self.axes[0, 1]
        ax_loss.set_title('训练损失')
        ax_loss.set_xlabel('Training Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        self.lines['loss'], = ax_loss.plot([], [], 'g-', linewidth=1, label='Loss')
        ax_loss.legend()
        
        # Epsilon衰减 (左下)
        ax_epsilon = self.axes[1, 0]
        ax_epsilon.set_title('Epsilon 衰减曲线')
        ax_epsilon.set_xlabel('Episode')
        ax_epsilon.set_ylabel('Epsilon')
        ax_epsilon.grid(True, alpha=0.3)
        self.lines['epsilon'], = ax_epsilon.plot([], [], 'm-', linewidth=2, label='Epsilon')
        ax_epsilon.legend()
        
        # 训练统计 (右下)
        ax_stats = self.axes[1, 1]
        ax_stats.set_title('训练统计')
        ax_stats.axis('off')
        self.stats_text = ax_stats.text(0.1, 0.9, '', fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
        # 设置交互功能
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def update_plots(self):
        """更新图表"""
        if not self.episodes or self.fig is None or self.axes is None:
            return
        
        # 更新奖励曲线
        self.lines['reward'].set_data(self.episodes, self.rewards)
        if self.moving_avg_rewards:
            self.lines['reward_avg'].set_data(self.episodes, self.moving_avg_rewards)
        
        # 自动调整奖励图的范围
        ax_reward = self.axes[0, 0]
        if self.rewards:
            ax_reward.set_xlim(0, max(len(self.episodes), 100))
            y_min, y_max = min(self.rewards), max(self.rewards)
            y_range = y_max - y_min
            ax_reward.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # 更新损失曲线
        if self.loss_history:
            loss_episodes = list(range(1, len(self.loss_history) + 1))
            self.lines['loss'].set_data(loss_episodes, self.loss_history)
            
            ax_loss = self.axes[0, 1]
            ax_loss.set_xlim(0, max(len(self.loss_history), 100))
            if len(self.loss_history) > 10:
                # 排除异常值后设置范围
                sorted_loss = sorted(self.loss_history)
                y_min = sorted_loss[len(sorted_loss)//10]  # 排除最小10%
                y_max = sorted_loss[len(sorted_loss)*9//10]  # 排除最大10%
                ax_loss.set_ylim(max(0, y_min * 0.9), y_max * 1.1)
        
        # 更新epsilon曲线
        if self.epsilon_history:
            self.lines['epsilon'].set_data(self.episodes, self.epsilon_history)
            ax_epsilon = self.axes[1, 0]
            ax_epsilon.set_xlim(0, max(len(self.episodes), 100))
            ax_epsilon.set_ylim(0, 1.0)
        
        # 更新统计信息
        self._update_stats_text()
        
        if self.fig and self.fig.canvas:
            self.fig.canvas.draw()
    
    def _update_stats_text(self):
        """更新统计文本"""
        if not self.rewards or self.stats_text is None:
            return
        
        # 计算统计信息
        total_episodes = len(self.rewards)
        recent_episodes = min(100, total_episodes)
        recent_rewards = self.rewards[-recent_episodes:] if recent_episodes > 0 else []
        
        current_reward = self.rewards[-1] if self.rewards else 0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        best_reward = max(self.rewards) if self.rewards else 0
        worst_reward = min(self.rewards) if self.rewards else 0
        
        current_epsilon = self.epsilon_history[-1] if self.epsilon_history else 0
        
        # 估算训练速度 (episodes/分钟)
        training_speed = "N/A"
        if hasattr(self, '_start_time') and total_episodes > 10:
            elapsed_time = time.time() - self._start_time
            speed = total_episodes / (elapsed_time / 60)  # episodes per minute
            training_speed = f"{speed:.1f} ep/min"
        
        stats_text = f"""训练进度统计
        
当前阶段: {self.current_phase}
总Episodes: {total_episodes:,}
        
奖励统计 (最近{recent_episodes}局):
  当前: {current_reward:.3f}
  平均: {avg_reward:.3f}
  最佳: {best_reward:.3f}
  最差: {worst_reward:.3f}
        
训练参数:
  当前Epsilon: {current_epsilon:.3f}
  训练速度: {training_speed}
        
快捷键:
  R - 重置视图
  S - 保存截图
  Q - 退出监控"""
        
        self.stats_text.set_text(stats_text)
    
    def _on_key_press(self, event):
        """键盘快捷键处理"""
        if event.key == 'r':
            # 重置视图
            if self.axes is not None:
                for ax in self.axes.flat:
                    ax.relim()
                    ax.autoscale()
                if self.fig and self.fig.canvas:
                    self.fig.canvas.draw()
                print("[Monitor] 视图已重置")
        
        elif event.key == 's':
            # 保存截图
            if self.fig is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"training_monitor_{timestamp}.png"
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"[Monitor] 截图已保存: {filename}")
        
        elif event.key == 'q':
            # 退出
            if self.fig is not None:
                plt.close(self.fig)
                print("[Monitor] 监控已退出")
    
    def start_monitoring(self):
        """开始实时监控"""
        print("[Monitor] 启动实时训练监控...")
        print("[Monitor] 查找训练文件...")
        
        files = self.find_latest_training_files()
        if not any(files.values()):
            print("[Monitor] 未找到训练文件，请先开始训练")
            return
        
        print(f"[Monitor] 找到训练文件:")
        for key, filepath in files.items():
            if filepath:
                print(f"  {key}: {filepath}")
        
        self._start_time = time.time()
        self.setup_plots()
        
        # 初始加载数据
        self.load_training_data()
        self.update_plots()
        
        print("[Monitor] 监控界面已启动")
        print("[Monitor] 快捷键: R-重置视图, S-保存截图, Q-退出")
        
        # 启动定时更新
        def update_data():
            while plt.get_fignums():  # 检查窗口是否还存在
                try:
                    if self.load_training_data():
                        self.update_plots()
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"[Monitor] 更新数据时出错: {e}")
                    time.sleep(self.update_interval)
        
        # 在后台线程中运行数据更新
        update_thread = threading.Thread(target=update_data, daemon=True)
        update_thread.start()
        
        # 显示界面
        plt.show()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Hive-RL 实时训练监控')
    parser.add_argument('--model-dir', default='models', help='模型文件目录')
    parser.add_argument('--update-interval', type=int, default=5, help='更新间隔(秒)')
    
    args = parser.parse_args()
    
    monitor = RealTimeTrainingMonitor(
        model_dir=args.model_dir,
        update_interval=args.update_interval
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n[Monitor] 用户中断，退出监控")
    except Exception as e:
        print(f"[Monitor] 监控过程中出现错误: {e}")


if __name__ == "__main__":
    main()

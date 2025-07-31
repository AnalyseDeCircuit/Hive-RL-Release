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
from datetime import datetime
from typing import Optional, List, Dict
import threading
import queue

# 设置matplotlib使用英文，避免中文字体问题
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

class RealTimeTrainingMonitor:
    """Real-time Training Monitor"""
    
    def __init__(self, model_dir: str = "models", update_interval: int = 5):
        self.model_dir = model_dir
        self.update_interval = update_interval  # Update interval in seconds
        
        # Data storage
        self.episodes = []
        self.rewards = []
        self.moving_avg_rewards = []
        self.loss_history = []
        self.win_rates = []
        self.epsilon_history = []
        self.episode_steps = []  # 新增：记录每局步数
        
        # Training statistics
        self.current_phase = "Unknown"
        self.total_episodes = 0
        self.training_start_time = None  # 训练开始时间
        self.first_episode_time = None   # 第一个episode的时间
        self.episodes_per_min = 0.0
        
        # GUI components
        self.fig = None
        self.axes = None
        self.lines = {}
        self.stats_text = None
        self.phase_text = None  # 新增
        
        # 文件监控
        self.last_modified_times = {}
        
        # 训练速度计算相关
        self.episodes_at_start = 0  # 监控开始时的episode数量
        
    def find_latest_training_files(self) -> Dict[str, Optional[str]]:
        """查找最新的训练文件"""
        files: Dict[str, Optional[str]] = {
            'reward': None,
            'loss': None,
            'meta': None,
            'end_stats': None,
            'steps': None
        }
        
        # 查找最新的训练目录
        pattern = os.path.join(self.model_dir, "*/")
        dirs = glob.glob(pattern)
        if not dirs:
            return files
            
        # 按时间排序，取最新的
        dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # 尝试从多个目录中查找文件，因为有些文件可能在不同的训练运行中
        for latest_dir in dirs[:3]:  # 检查最新的3个目录
            print(f"[Monitor] Checking directory: {latest_dir}")
            
            # 查找各种文件
            for file_type in ['reward_history', 'loss_history', 'meta', 'end_stats_history', 'steps_history']:
                if files.get(file_type.split('_')[0] if '_' in file_type else file_type):
                    continue  # 如果已经找到该类型文件，跳过
                    
                pattern = os.path.join(latest_dir, f"*_{file_type}.npy")
                if file_type == 'meta':
                    pattern = os.path.join(latest_dir, f"*_meta.json")
                
                candidates = glob.glob(pattern)
                if candidates:
                    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    if file_type == 'reward_history':
                        files['reward'] = candidates[0]
                        print(f"[Monitor] Found reward file: {candidates[0]}")
                    elif file_type == 'loss_history':
                        files['loss'] = candidates[0]
                        print(f"[Monitor] Found loss file: {candidates[0]}")
                    elif file_type == 'meta':
                        files['meta'] = candidates[0]
                        print(f"[Monitor] Found meta file: {candidates[0]}")
                    elif file_type == 'end_stats_history':
                        files['end_stats'] = candidates[0]
                        print(f"[Monitor] Found end_stats file: {candidates[0]}")
                    elif file_type == 'steps_history':
                        files['steps'] = candidates[0]
                        print(f"[Monitor] Found steps file: {candidates[0]}")
        
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
                    
                    # 计算训练速度 - 修复计算方式
                    self._update_training_speed()
                    
                    # 尝试检测当前训练阶段
                    self._detect_training_phase()
                    
                    updated = True
                    print(f"[Monitor] Reward data updated: {len(self.rewards)} episodes")
            except Exception as e:
                print(f"[Monitor] Failed to load reward data: {e}")
        
        # 加载每局步数历史 - 增强版
        steps_loaded = False
        
        # 方法1：从找到的steps文件加载
        if files['steps'] and self._is_file_updated(files['steps']):
            try:
                steps_data = np.load(files['steps'])
                if len(steps_data) > len(self.episode_steps):
                    self.episode_steps = steps_data.tolist()
                    steps_loaded = True
                    updated = True
                    print(f"[Monitor] Episode steps data updated: {len(self.episode_steps)} episodes from {files['steps']}")
            except Exception as e:
                print(f"[Monitor] Failed to load episode steps data: {e}")
        
        # 方法2：如果没有找到steps文件或加载失败，进行更广泛的搜索
        # 重要修复：每次都重新搜索最新文件，不依赖文件更新检查
        if not steps_loaded:
            print("[Monitor] Searching for latest steps files in all training directories...")
            try:
                # 搜索所有可能的步数文件模式
                all_dirs = glob.glob(os.path.join(self.model_dir, "*/"))
                all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                latest_steps_file = None
                latest_steps_data = None
                max_episodes = len(self.episode_steps)
                
                for search_dir in all_dirs[:5]:  # 检查最新的5个目录
                    steps_patterns = [
                        os.path.join(search_dir, "*_steps_history.npy"),
                        os.path.join(search_dir, "*steps*.npy"),
                        os.path.join(search_dir, "*episode_steps*.npy")
                    ]
                    
                    for pattern in steps_patterns:
                        step_files = glob.glob(pattern)
                        if step_files:
                            step_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                            try:
                                # 每次都尝试加载，不依赖_is_file_updated检查
                                steps_data = np.load(step_files[0])
                                # 选择episode数最多的文件作为最新数据
                                if len(steps_data) > max_episodes:
                                    max_episodes = len(steps_data)
                                    latest_steps_file = step_files[0]
                                    latest_steps_data = steps_data
                            except Exception as e:
                                print(f"[Monitor] Failed to load {step_files[0]}: {e}")
                                continue
                
                # 使用找到的最新数据
                if latest_steps_file and latest_steps_data is not None:
                    if len(latest_steps_data) > len(self.episode_steps):
                        self.episode_steps = latest_steps_data.tolist()
                        steps_loaded = True
                        updated = True
                        print(f"[Monitor] Episode steps data found and updated: {len(self.episode_steps)} episodes from {latest_steps_file}")
                    else:
                        print(f"[Monitor] Found steps file {latest_steps_file} but data is not newer ({len(latest_steps_data)} <= {len(self.episode_steps)})")
                        
                if not steps_loaded:
                    print("[Monitor] No newer steps history files found in any training directory")
                    
            except Exception as e:
                print(f"[Monitor] Error during comprehensive steps file search: {e}")
        
        # 方法3：如果还是没有数据，尝试从其他历史文件推断
        if not steps_loaded and not self.episode_steps and self.rewards:
            print("[Monitor] No steps data found, will show 'N/A' for step statistics")
            # 不设置默认值，保持为空列表，这样统计显示会是"N/A"
        
        # Load loss history - optional data source
        if files['loss'] and self._is_file_updated(files['loss']):
            try:
                loss_data = np.load(files['loss'])
                if len(loss_data) > len(self.loss_history):
                    self.loss_history = loss_data.tolist()
                    updated = True
                    print(f"[Monitor] Loss data updated: {len(self.loss_history)} steps")
            except Exception as e:
                print(f"[Monitor] Failed to load loss data: {e}")
        
        # Try to load metadata - completely optional, failure won't affect monitoring
        if files['meta'] and self._is_file_updated(files['meta']):
            try:
                with open(files['meta'], 'r') as f:
                    meta = json.load(f)
                    if 'player1_epsilon' in meta:
                        # If actual epsilon data exists, update estimation
                        current_epsilon = meta['player1_epsilon']
                        print(f"[Monitor] Metadata updated: epsilon={current_epsilon:.3f}")
                    updated = True
            except Exception as e:
                # Meta file might not exist during training, this is normal
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
        """Setup plot interface"""
        self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 12))  # 改为3x2布局
        self.fig.suptitle('Hive-RL Real-time Training Monitor', fontsize=16, fontweight='bold')
        
        # Reward curve (top left)
        ax_reward = self.axes[0, 0]
        ax_reward.set_title('Reward Curve')
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Reward')
        ax_reward.grid(True, alpha=0.3)
        self.lines['reward'], = ax_reward.plot([], [], 'b-', alpha=0.3, label='Raw Reward')
        self.lines['reward_avg'], = ax_reward.plot([], [], 'r-', linewidth=2, label='Moving Avg (100)')
        ax_reward.legend()
        
        # Loss curve (top right)
        ax_loss = self.axes[0, 1]
        ax_loss.set_title('Training Loss')
        ax_loss.set_xlabel('Training Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        self.lines['loss'], = ax_loss.plot([], [], 'g-', linewidth=1, label='Loss')
        ax_loss.legend()
        
        # Episode steps time series (middle left)
        ax_steps = self.axes[1, 0]
        ax_steps.set_title('Episode Steps (Time Series)')
        ax_steps.set_xlabel('Episode')
        ax_steps.set_ylabel('Steps per Episode')
        ax_steps.grid(True, alpha=0.3)
        self.lines['steps'], = ax_steps.plot([], [], 'orange', alpha=0.5, label='Steps per Episode')
        self.lines['steps_avg'], = ax_steps.plot([], [], 'darkorange', linewidth=2, label='Moving Avg (100)')
        ax_steps.legend()
        
        # Episode steps histogram (middle right) - 新增频率分布图
        ax_steps_hist = self.axes[1, 1]
        ax_steps_hist.set_title('Episode Steps Distribution')
        ax_steps_hist.set_xlabel('Steps per Episode')
        ax_steps_hist.set_ylabel('Frequency')
        ax_steps_hist.grid(True, alpha=0.3)
        # 这个会在update中动态绘制
        
        # Epsilon decay (bottom left)
        ax_epsilon = self.axes[2, 0]
        ax_epsilon.set_title('Epsilon Decay Curve')
        ax_epsilon.set_xlabel('Episode')
        ax_epsilon.set_ylabel('Epsilon')
        ax_epsilon.grid(True, alpha=0.3)
        self.lines['epsilon'], = ax_epsilon.plot([], [], 'm-', linewidth=2, label='Epsilon')
        ax_epsilon.legend()
        
        # Training statistics (bottom right)
        ax_stats = self.axes[2, 1]
        ax_stats.set_title('Training Statistics')
        ax_stats.axis('off')
        self.stats_text = ax_stats.text(0.1, 0.9, '', fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
        # Setup interactive functionality
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def update_plots(self):
        """Update plots"""
        if not self.episodes or self.fig is None or self.axes is None:
            return
        
        # Update reward curve
        self.lines['reward'].set_data(self.episodes, self.rewards)
        if self.moving_avg_rewards:
            self.lines['reward_avg'].set_data(self.episodes, self.moving_avg_rewards)
        
        # Auto-adjust reward plot range
        ax_reward = self.axes[0, 0]
        if self.rewards:
            ax_reward.set_xlim(0, max(len(self.episodes), 100))
            y_min, y_max = min(self.rewards), max(self.rewards)
            y_range = y_max - y_min
            ax_reward.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Update loss curve
        if self.loss_history:
            loss_episodes = list(range(1, len(self.loss_history) + 1))
            self.lines['loss'].set_data(loss_episodes, self.loss_history)
            
            ax_loss = self.axes[0, 1]
            ax_loss.set_xlim(0, max(len(self.loss_history), 100))
            if len(self.loss_history) > 10:
                # 修复：更好的异常值处理，防止尖峰超出范围
                sorted_loss = sorted(self.loss_history)
                # 去除最极端的5%异常值
                lower_bound = len(sorted_loss) // 20  # 去除最低5%
                upper_bound = len(sorted_loss) * 19 // 20  # 去除最高5%
                
                if upper_bound > lower_bound:
                    y_min = sorted_loss[lower_bound]
                    y_max = sorted_loss[upper_bound]
                    
                    # 确保范围合理，不会太小
                    y_range = y_max - y_min
                    if y_range < 0.001:  # 防止范围过小
                        y_center = (y_max + y_min) / 2
                        y_min = y_center - 0.001
                        y_max = y_center + 0.001
                    
                    # 添加15%的边距
                    margin = y_range * 0.15
                    ax_loss.set_ylim(max(0, y_min - margin), y_max + margin)
                else:
                    # 备用方案：使用中位数范围
                    median_loss = np.median(self.loss_history)
                    ax_loss.set_ylim(0, median_loss * 3)
        
        # Update episode steps curve - 修复数据长度不匹配问题
        if self.episode_steps and len(self.episode_steps) > 0:
            # 检查数据长度匹配情况
            steps_count = len(self.episode_steps)
            rewards_count = len(self.rewards)
            
            if steps_count < rewards_count * 0.1:  # 如果步数数据太少（少于奖励数据的10%）
                print(f"[Monitor] Warning: Steps data ({steps_count}) much smaller than rewards data ({rewards_count})")
                
                # 检查是否可能是训练刚开始或者实时保存延迟
                if steps_count == 0:
                    reason = "Training may have just started or steps data not yet saved"
                elif rewards_count > 15000:  # 大量奖励数据但步数很少
                    reason = "Steps data appears to be from an older training run"
                else:
                    reason = "Steps data may be delayed due to save frequency"
                
                print(f"[Monitor] Likely reason: {reason}")
                
                # 时序图显示警告信息
                ax_steps = self.axes[1, 0]
                ax_steps.clear()
                ax_steps.set_title('Episode Steps (Data Mismatch)')
                ax_steps.text(0.5, 0.5, f'Steps data: {steps_count} episodes\nReward data: {rewards_count} episodes\n\n{reason}\n\nIf training is active, steps data\nshould appear soon.', 
                             ha='center', va='center', fontsize=9, 
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
                ax_steps.set_xlim(0, 1)
                ax_steps.set_ylim(0, 1)
                ax_steps.axis('off')
                
                # 频率图也显示警告
                ax_steps_hist = self.axes[1, 1]
                ax_steps_hist.clear()
                ax_steps_hist.set_title('Episode Steps Distribution (No Data)')
                ax_steps_hist.text(0.5, 0.5, 'No current steps data\nfor frequency analysis', 
                                  ha='center', va='center', fontsize=10,
                                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
                ax_steps_hist.set_xlim(0, 1)
                ax_steps_hist.set_ylim(0, 1)
                ax_steps_hist.axis('off')
                
            else:
                # 数据匹配良好，正常显示图表
                steps_episodes = list(range(1, len(self.episode_steps) + 1))
                self.lines['steps'].set_data(steps_episodes, self.episode_steps)
                
                # 计算步数移动平均
                window = min(100, len(self.episode_steps))
                if len(self.episode_steps) >= window:
                    steps_moving_avg = []
                    for i in range(len(self.episode_steps)):
                        start_idx = max(0, i - window + 1)
                        avg = np.mean(self.episode_steps[start_idx:i+1])
                        steps_moving_avg.append(avg)
                    self.lines['steps_avg'].set_data(steps_episodes, steps_moving_avg)
                
                # 更新时序图
                ax_steps = self.axes[1, 0]
                ax_steps.set_xlim(0, max(len(self.episode_steps), 100))
                if self.episode_steps:
                    y_min, y_max = min(self.episode_steps), max(self.episode_steps)
                    y_range = y_max - y_min
                    if y_range > 0:
                        ax_steps.set_ylim(max(0, y_min - y_range * 0.1), y_max + y_range * 0.1)
                    else:
                        # 如果所有步数都相同，设置一个合理的范围
                        ax_steps.set_ylim(max(0, y_min - 5), y_max + 5)
                
                # 更新频率分布图 - 新增
                ax_steps_hist = self.axes[1, 1]
                ax_steps_hist.clear()
                ax_steps_hist.set_title('Episode Steps Distribution')
                ax_steps_hist.set_xlabel('Steps per Episode')
                ax_steps_hist.set_ylabel('Frequency')
                ax_steps_hist.grid(True, alpha=0.3)
                
                # 计算频率分布
                if len(self.episode_steps) > 10:  # 至少10个数据点才画直方图
                    # 使用自适应的bins数量
                    bins = min(50, max(10, len(set(self.episode_steps))))
                    
                    # 绘制直方图
                    n, bins_edges, patches = ax_steps_hist.hist(self.episode_steps, bins=bins, alpha=0.7, color='orange', edgecolor='darkorange')
                    
                    # 特别标注低步数的情况
                    step_counts = {}
                    for step in self.episode_steps:
                        step_counts[step] = step_counts.get(step, 0) + 1
                    
                    # 标注特殊的步数
                    low_step_colors = {2: 'red', 3: 'purple', 4: 'blue', 5: 'green'}
                    for i, (n_count, left_edge, right_edge) in enumerate(zip(n, bins_edges[:-1], bins_edges[1:])):
                        center = (left_edge + right_edge) / 2
                        if center <= 5 and n_count > 0:  # 低步数且有数据
                            color = low_step_colors.get(int(center), 'black')
                            patches[i].set_facecolor(color)
                            patches[i].set_alpha(0.8)
                            # 添加数值标签
                            ax_steps_hist.text(center, n_count + max(n) * 0.01, f'{int(n_count)}', 
                                             ha='center', va='bottom', fontweight='bold', color=color)
                    
                    # 添加统计信息
                    unique_steps = len(set(self.episode_steps))
                    min_steps = min(self.episode_steps)
                    max_steps = max(self.episode_steps)
                    
                    # 计算低步数比例
                    low_steps_count = sum(1 for s in self.episode_steps if s <= 5)
                    low_steps_ratio = (low_steps_count / len(self.episode_steps)) * 100
                    
                    info_text = f"Total: {len(self.episode_steps)} episodes\nUnique steps: {unique_steps}\nRange: {min_steps}-{max_steps}\n≤5 steps: {low_steps_count} ({low_steps_ratio:.1f}%)"
                    ax_steps_hist.text(0.98, 0.98, info_text, transform=ax_steps_hist.transAxes, 
                                      fontsize=8, verticalalignment='top', horizontalalignment='right',
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                else:
                    ax_steps_hist.text(0.5, 0.5, f'Need more data\n({len(self.episode_steps)} episodes)', 
                                      ha='center', va='center', fontsize=10,
                                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
                    ax_steps_hist.set_xlim(0, 1)
                    ax_steps_hist.set_ylim(0, 1)
                
                print(f"[Monitor] Steps curve updated: {len(self.episode_steps)} episodes, latest: {self.episode_steps[-1]} steps")
        else:
            # 没有步数数据，显示提示信息
            ax_steps = self.axes[1, 0]
            ax_steps.clear()
            ax_steps.set_title('Episode Steps (No Data)')
            ax_steps.text(0.5, 0.5, 'No episode steps data found.\n\nThis could mean:\n1. Training just started\n2. Current training not saving steps\n3. Steps file not yet created\n\nCheck if ai_trainer.py is saving\nsteps_history.npy file.', 
                         ha='center', va='center', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
            ax_steps.set_xlim(0, 1)
            ax_steps.set_ylim(0, 1)
            ax_steps.axis('off')
            
            # 频率图也显示无数据
            ax_steps_hist = self.axes[1, 1]
            ax_steps_hist.clear()
            ax_steps_hist.set_title('Episode Steps Distribution (No Data)')
            ax_steps_hist.text(0.5, 0.5, 'No steps data available\nfor frequency analysis', 
                              ha='center', va='center', fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
            ax_steps_hist.set_xlim(0, 1)
            ax_steps_hist.set_ylim(0, 1)
            ax_steps_hist.axis('off')
        
        # Update epsilon curve
        if self.epsilon_history:
            self.lines['epsilon'].set_data(self.episodes, self.epsilon_history)
            ax_epsilon = self.axes[2, 0]
            ax_epsilon.set_xlim(0, max(len(self.episodes), 100))
            ax_epsilon.set_ylim(0, 1.0)
            print(f"[Monitor] Epsilon curve updated: {len(self.epsilon_history)} episodes, latest: {self.epsilon_history[-1]:.4f}")
        else:
            ax_epsilon = self.axes[2, 0]
            ax_epsilon.clear()
            ax_epsilon.set_title('Epsilon Decay (No Data)')
            ax_epsilon.text(0.5, 0.5, 'No epsilon data found.\n\nEpsilon values track exploration\nvs exploitation balance.', 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            ax_epsilon.set_xlim(0, 1)
            ax_epsilon.set_ylim(0, 1)
            ax_epsilon.axis('off')
        
        # Update training statistics
        ax_stats = self.axes[2, 1]
        ax_stats.clear()
        ax_stats.set_title('Training Statistics')
        ax_stats.axis('off')
        
        # 计算统计信息
        if self.rewards and len(self.rewards) > 0:
            reward_stats = {
                'Episodes': len(self.rewards),
                'Avg Reward': np.mean(self.rewards),
                'Best Reward': np.max(self.rewards),
                'Worst Reward': np.min(self.rewards),
                'Recent Avg (100)': np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
            }
        else:
            reward_stats = {'Episodes': 0}
        
        if self.episode_steps and len(self.episode_steps) > 0:
            steps_stats = {
                'Avg Steps': np.mean(self.episode_steps),
                'Min Steps': np.min(self.episode_steps),
                'Max Steps': np.max(self.episode_steps),
                'Recent Avg (100)': np.mean(self.episode_steps[-100:]) if len(self.episode_steps) >= 100 else np.mean(self.episode_steps)
            }
        else:
            steps_stats = {}
        
        # 格式化显示
        stats_text = "REWARD STATISTICS:\n"
        for key, value in reward_stats.items():
            if isinstance(value, float):
                stats_text += f"{key}: {value:.3f}\n"
            else:
                stats_text += f"{key}: {value}\n"
        
        if steps_stats:
            stats_text += "\nSTEPS STATISTICS:\n"
            for key, value in steps_stats.items():
                if isinstance(value, float):
                    stats_text += f"{key}: {value:.1f}\n"
                else:
                    stats_text += f"{key}: {value}\n"
        else:
            stats_text += "\nSTEPS STATISTICS:\nNo current data available\n"
        
        # 添加训练时间信息
        stats_text += f"\nMONITOR INFO:\n"
        stats_text += f"Last Update: {datetime.now().strftime('%H:%M:%S')}\n"
        
        # 添加训练路径信息 - 使用安全的方式
        latest_files = self.find_latest_training_files()
        if latest_files and latest_files['reward']:
            model_dir = os.path.dirname(latest_files['reward'])
            model_name = os.path.basename(model_dir)
            stats_text += f"Model: {model_name}\n"
        else:
            stats_text += f"Model Dir: {self.model_dir}\n"
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Draw the updated figure
        if self.fig and self.fig.canvas:
            self.fig.canvas.draw()
    
    def _update_stats_text(self):
        """Update statistics text"""
        if not self.rewards or self.stats_text is None:
            return
        
        # Calculate statistics
        total_episodes = len(self.rewards)
        recent_episodes = min(100, total_episodes)
        recent_rewards = self.rewards[-recent_episodes:] if recent_episodes > 0 else []
        
        current_reward = self.rewards[-1] if self.rewards else 0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        best_reward = max(self.rewards) if self.rewards else 0
        worst_reward = min(self.rewards) if self.rewards else 0
        
        current_epsilon = self.epsilon_history[-1] if self.epsilon_history else 0
        
        # Estimate training speed (episodes/minute) - 修复计算方式
        if hasattr(self, 'episodes_per_min'):
            training_speed = f"{self.episodes_per_min:.1f} ep/min"
        else:
            training_speed = "Calculating..."
        
        # 计算每局平均步数 - 处理数据不匹配情况
        avg_steps = "N/A"
        max_steps = "N/A" 
        min_steps = "N/A"
        steps_status = ""
        
        if self.episode_steps and len(self.episode_steps) > 0:
            steps_count = len(self.episode_steps)
            rewards_count = len(self.rewards)
            
            if steps_count < rewards_count * 0.1:
                # 数据不匹配，可能是旧的步数数据
                recent_steps = self.episode_steps[-min(recent_episodes, steps_count):]
                if recent_steps:
                    avg_steps = f"{np.mean(recent_steps):.1f}"
                    max_steps = f"{max(recent_steps):.0f}"
                    min_steps = f"{min(recent_steps):.0f}"
                steps_status = f"(⚠️ {steps_count} old episodes vs {rewards_count} current)"
            else:
                # 数据匹配良好
                recent_steps = self.episode_steps[-recent_episodes:] if recent_episodes > 0 else self.episode_steps
                if recent_steps:
                    avg_steps = f"{np.mean(recent_steps):.1f}"
                    max_steps = f"{max(recent_steps):.0f}"
                    min_steps = f"{min(recent_steps):.0f}"
                steps_status = f"({steps_count} episodes recorded)"
        else:
            steps_status = "(No steps data found)"
        
        stats_text = f"""Training Progress Statistics
        
Total Episodes: {total_episodes:,}
Training Speed: {training_speed}
        
Reward Statistics (Recent {recent_episodes}):
  Current: {current_reward:.3f}
  Average: {avg_reward:.3f}
  Best: {best_reward:.3f}
  Worst: {worst_reward:.3f}
        
Episode Steps {steps_status}:
  Average: {avg_steps}
  Max: {max_steps}
  Min: {min_steps}
        
Current Epsilon: {current_epsilon:.3f}
        
Hotkeys: R-Reset, S-Save, Q-Exit"""
        
        self.stats_text.set_text(stats_text)
    
    def _on_key_press(self, event):
        """Keyboard shortcut handler"""
        if event.key == 'r':
            # Reset view
            if self.axes is not None:
                for ax in self.axes.flat:
                    ax.relim()
                    ax.autoscale()
                if self.fig and self.fig.canvas:
                    self.fig.canvas.draw()
                print("[Monitor] View reset")
        
        elif event.key == 's':
            # Save screenshot
            if self.fig is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"training_monitor_{timestamp}.png"
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"[Monitor] Screenshot saved: {filename}")
        
        elif event.key == 'q':
            # Exit
            if self.fig is not None:
                plt.close(self.fig)
                print("[Monitor] Monitor exited")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print("[Monitor] Starting real-time training monitor...")
        print("[Monitor] Looking for training files...")
        
        files = self.find_latest_training_files()
        if not any(files.values()):
            print("[Monitor] No training files found, please start training first")
            return
        
        print(f"[Monitor] Found training files:")
        for key, filepath in files.items():
            if filepath:
                print(f"  {key}: {filepath}")
        
        self._start_time = time.time()
        self.setup_plots()
        
        # Initial data loading
        self.load_training_data()
        self.update_plots()
        
        print("[Monitor] Monitor interface started")
        print("[Monitor] Hotkeys: R-Reset view, S-Save screenshot, Q-Exit")
        
        # Start timed updates
        def update_data():
            while plt.get_fignums():  # Check if window still exists
                try:
                    if self.load_training_data():
                        self.update_plots()
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"[Monitor] Error updating data: {e}")
                    time.sleep(self.update_interval)
        
        # Run data update in background thread
        update_thread = threading.Thread(target=update_data, daemon=True)
        update_thread.start()
        
        # Show interface
        plt.show()
    
    def _update_training_speed(self):
        """计算训练速度 - 修复训练中启动监控的计算"""
        if not self.rewards or len(self.rewards) < 10:
            self.episodes_per_min = 0.0
            return
        
        # 记录第一次看到数据的时间
        if self.first_episode_time is None:
            self.first_episode_time = time.time()
            self.episodes_at_start = len(self.rewards)
            return
        
        # 计算从监控开始到现在的speed
        current_time = time.time()
        elapsed_minutes = (current_time - self.first_episode_time) / 60.0
        
        if elapsed_minutes > 0.1:  # 至少6秒后才计算
            episodes_since_start = len(self.rewards) - self.episodes_at_start
            self.episodes_per_min = episodes_since_start / elapsed_minutes
        else:
            self.episodes_per_min = 0.0
    
    def _detect_training_phase(self):
        """检测当前训练阶段"""
        total_episodes = len(self.rewards)
        
        # 基于episode数量和epsilon值推断阶段
        if total_episodes < 35000:
            self.current_phase = "Foundation Phase"
        elif total_episodes < 80000:
            self.current_phase = "Strategy Phase"
        elif total_episodes < 120000:
            self.current_phase = "Mastery Phase"
        else:
            self.current_phase = "Extended Training"
        
        # 尝试从epsilon值精确判断
        if self.epsilon_history:
            current_epsilon = self.epsilon_history[-1]
            if current_epsilon > 0.7:
                self.current_phase = "Foundation Phase (Early)"
            elif current_epsilon > 0.5:
                self.current_phase = "Foundation Phase (Late)"
            elif current_epsilon > 0.3:
                self.current_phase = "Strategy Phase"
            elif current_epsilon > 0.1:
                self.current_phase = "Mastery Phase"
            else:
                self.current_phase = "Fine-tuning Phase"
    
    def _update_phase_text(self):
        """更新阶段信息文本"""
        if not hasattr(self, 'phase_text') or self.phase_text is None:
            return
        
        total_episodes = len(self.rewards)
        current_epsilon = self.epsilon_history[-1] if self.epsilon_history else 0
        
        # 计算阶段进度
        phase_progress = ""
        if "Foundation" in self.current_phase and total_episodes < 35000:
            progress = (total_episodes / 35000) * 100
            phase_progress = f"Progress: {progress:.1f}%"
        elif "Strategy" in self.current_phase and total_episodes < 80000:
            progress = ((total_episodes - 35000) / 45000) * 100
            phase_progress = f"Progress: {progress:.1f}%"
        elif "Mastery" in self.current_phase and total_episodes < 120000:
            progress = ((total_episodes - 80000) / 40000) * 100
            phase_progress = f"Progress: {progress:.1f}%"
        
        # 估算剩余时间
        eta = "N/A"
        if hasattr(self, 'episodes_per_min') and self.episodes_per_min > 0:
            remaining_episodes = 0
            if total_episodes < 35000:
                remaining_episodes = 35000 - total_episodes
            elif total_episodes < 80000:
                remaining_episodes = 80000 - total_episodes
            elif total_episodes < 120000:
                remaining_episodes = 120000 - total_episodes
            
            if remaining_episodes > 0:
                eta_minutes = remaining_episodes / self.episodes_per_min
                if eta_minutes < 60:
                    eta = f"{eta_minutes:.0f} min"
                else:
                    eta = f"{eta_minutes/60:.1f} hours"
        
        phase_text = f"""Training Phase Information

Current Phase: {self.current_phase}
{phase_progress}

Phase Targets:
  Foundation: 35K episodes
  Strategy: 45K episodes  
  Mastery: 40K episodes

ETA to next phase: {eta}

Current Epsilon: {current_epsilon:.3f}
Total Episodes: {total_episodes:,}"""
        
        self.phase_text.set_text(phase_text)


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Hive-RL Real-time Training Monitor')
    parser.add_argument('--model-dir', default='models', help='Model directory')
    parser.add_argument('--update-interval', type=int, default=5, help='Update interval (seconds)')
    
    args = parser.parse_args()
    
    monitor = RealTimeTrainingMonitor(
        model_dir=args.model_dir,
        update_interval=args.update_interval
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n[Monitor] User interrupt, exiting monitor")
    except Exception as e:
        print(f"[Monitor] Error during monitoring: {e}")


if __name__ == "__main__":
    main()

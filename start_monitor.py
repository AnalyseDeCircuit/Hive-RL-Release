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

# Setup matplotlib for English display to avoid font issues
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class SimpleRealTimeMonitor:
    """Simple Real-time Monitor"""
    
    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.rewards = []
        self.episodes = []
        self.moving_avg = []
        self.episode_steps = []  # 新增：每局步数
        self.last_update = 0
        
    def find_latest_reward_file(self):
        """Find latest reward file"""
        patterns = [
            "models/*/DQN_reward_history.npy",
            "models/*/*_reward_history.npy", 
            "models/*reward_history.npy",
            "reward_history.npy"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Sort by modification time, take the latest
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                print(f"[Monitor] Found reward file: {files[0]}")
                return files[0]
        
        print("[Monitor] No reward file found, supported file patterns:")
        for pattern in patterns:
            print(f"  - {pattern}")
        return None
    
    def load_data(self):
        """Load training data"""
        reward_file = self.find_latest_reward_file()
        if not reward_file:
            return False
            
        try:
            # Check if file is updated
            mtime = os.path.getmtime(reward_file)
            if mtime <= self.last_update:
                return False
                
            self.last_update = mtime
            
            # Load data
            rewards = np.load(reward_file)
            if len(rewards) == len(self.rewards):
                return False  # No new data
                
            self.rewards = rewards.tolist()
            self.episodes = list(range(1, len(self.rewards) + 1))
            
            # Calculate moving average
            window = min(50, len(self.rewards))
            if len(self.rewards) >= window:
                self.moving_avg = []
                for i in range(len(self.rewards)):
                    start_idx = max(0, i - window + 1)
                    avg = np.mean(self.rewards[start_idx:i+1])
                    self.moving_avg.append(avg)
            
            # 尝试加载步数数据
            steps_file = reward_file.replace('_reward_history.npy', '_steps_history.npy')
            if os.path.exists(steps_file):
                try:
                    steps_data = np.load(steps_file)
                    if len(steps_data) == len(self.rewards):
                        self.episode_steps = steps_data.tolist()
                except Exception as e:
                    print(f"[Monitor] Failed to load steps data: {e}")
            
            print(f"[Monitor] Data updated: {len(self.rewards)} episodes, latest reward: {self.rewards[-1]:.3f}")
            return True
            
        except Exception as e:
            print(f"[Monitor] Failed to load data: {e}")
            return False
    
    def setup_plot(self):
        """Setup plots"""
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Hive-RL Real-time Training Monitor', fontsize=14, fontweight='bold')
        
        # Reward curve
        self.ax1.set_title('Reward Curve')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', alpha=0.5, label='Raw Reward')
        self.line2, = self.ax1.plot([], [], 'r-', linewidth=2, label='Moving Avg (50)')
        self.ax1.legend()
        
        # Statistics
        self.ax2.set_title('Training Statistics')
        self.ax2.axis('off')
        self.stats_text = self.ax2.text(0.1, 0.9, '', fontsize=11, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
    def update_plot(self):
        """Update plots"""
        if not self.episodes:
            return
            
        # Update reward curve
        self.line1.set_data(self.episodes, self.rewards)
        if self.moving_avg:
            self.line2.set_data(self.episodes, self.moving_avg)
        
        # Auto-adjust range
        if self.rewards:
            self.ax1.set_xlim(0, max(len(self.episodes), 100))
            y_min, y_max = min(self.rewards), max(self.rewards)
            y_range = y_max - y_min if y_max != y_min else 1
            self.ax1.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Update statistics
        self.update_stats()
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_stats(self):
        """Update statistics"""
        if not self.rewards:
            return
            
        total_episodes = len(self.rewards)
        recent_count = min(100, total_episodes)
        recent_rewards = self.rewards[-recent_count:]
        
        current_reward = self.rewards[-1]
        avg_reward = np.mean(recent_rewards)
        best_reward = max(self.rewards)
        worst_reward = min(self.rewards)
        
        # 计算每局平均步数
        avg_steps = "N/A"
        max_steps = "N/A"
        min_steps = "N/A"
        if self.episode_steps:
            recent_steps = self.episode_steps[-recent_count:]
            avg_steps = f"{np.mean(recent_steps):.1f}"
            max_steps = f"{max(recent_steps):.0f}"
            min_steps = f"{min(recent_steps):.0f}"
        
        stats_text = f"""Training Progress Statistics

Total Episodes: {total_episodes:,}

Reward Statistics (Recent {recent_count} episodes):
  Current: {current_reward:.3f}
  Average: {avg_reward:.3f}
  Best: {best_reward:.3f}
  Worst: {worst_reward:.3f}

Episode Steps (Recent {recent_count} episodes):
  Average: {avg_steps}
  Max: {max_steps}
  Min: {min_steps}

Usage Tips:
  - Close window to exit monitor
  - Auto-update every {self.update_interval} seconds
  - Data source: Latest training files"""
        
        self.stats_text.set_text(stats_text)
    
    def start(self):
        """Start monitoring"""
        print("🚀 Starting Hive-RL Real-time Training Monitor...")
        print("📁 Looking for training files...")
        
        # Check if training data exists
        if not self.find_latest_reward_file():
            print("❌ No training data files found!")
            print("Please ensure:")
            print("  1. AI training has started")
            print("  2. models/ directory contains *_reward_history.npy files")
            return
        
        print("✅ Found training data, setting up monitoring interface...")
        
        # Setup charts
        self.setup_plot()
        
        # Initial load
        if self.load_data():
            self.update_plot()
        
        print("📊 Monitoring interface launched")
        print(f"🔄 Auto-update every {self.update_interval} seconds")
        print("💡 Close chart window to exit monitor")
        
        # Main loop
        try:
            while plt.get_fignums():  # Check if window is still open
                if self.load_data():
                    self.update_plot()
                    print(f"📈 Data updated (Episodes: {len(self.episodes)})")
                
                # Wait for update interval
                plt.pause(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted, exiting monitor")
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
        finally:
            plt.close('all')
            print("👋 Monitor exited")


def main():
    """Main function"""
    print("=" * 50)
    print("🎮 Hive-RL Real-time Training Monitor")
    print("=" * 50)
    
    # Create monitor
    monitor = SimpleRealTimeMonitor(update_interval=5)
    
    # Start monitoring
    monitor.start()


if __name__ == "__main__":
    main()

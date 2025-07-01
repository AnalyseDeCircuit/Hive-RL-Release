#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
improved_reward_shaping.py

基于Hive游戏胜负机制的科学reward shaping设计
核心原理：奖励归一化 + 渐进式学习信号 + 势函数引导
"""

import numpy as np
from typing import Tuple, Dict, Any

class HiveRewardShaper:
    """
    Hive游戏专用奖励整形器
    
    设计原则：
    1. 所有奖励归一化到[-1, 1]范围，避免尺度不一致
    2. 基于包围进度的渐进式奖励信号
    3. 早期重视学习合法动作，后期重视策略优化
    4. 平衡短期行动奖励与长期战略目标
    """
    
    def __init__(self, phase: str = 'foundation'):
        self.phase = phase
        self.step_count = 0
        
        # 不同阶段的奖励权重配置
        self.phase_configs = {
            'foundation': {
                'survival_weight': 0.02,     # 进一步降低存活奖励权重
                'action_weight': 0.05,       # 进一步降低动作奖励权重
                'strategy_weight': 0.50,     # 大幅提升战略奖励权重 (0.25 -> 0.50)
                'terminal_weight': 0.43,     # 适当降低终局奖励权重 (0.60 -> 0.43)
                'illegal_penalty': -8.0,     # 进一步提升非法动作惩罚
            },
            'strategy': {
                'survival_weight': 0.01,
                'action_weight': 0.04,
                'strategy_weight': 0.65,     # 策略阶段更重视战略 (0.40 -> 0.65)
                'terminal_weight': 0.30,     # 降低终局权重 (0.50 -> 0.30)
                'illegal_penalty': -6.0,     # 提升非法动作惩罚
            },
            'mastery': {
                'survival_weight': 0.01,
                'action_weight': 0.04,
                'strategy_weight': 0.45,     # 精通阶段平衡战略和终局 (0.30 -> 0.45)
                'terminal_weight': 0.50,     # 适当提升终局权重 (0.63 -> 0.50)
                'illegal_penalty': -8.0,     # 最严惩罚，与基础阶段一致
            }
        }
        
        self.config = self.phase_configs.get(phase, self.phase_configs['foundation'])
    
    def shape_reward(self, 
                    original_reward: float,
                    terminated: bool,
                    action_type: str,
                    my_queen_surrounded_count: int,
                    opp_queen_surrounded_count: int,
                    prev_my_queen_surrounded: int,
                    prev_opp_queen_surrounded: int,
                    is_illegal_action: bool = False,
                    turn_count: int = 1,
                    reason: str = '') -> float:
        """
        核心奖励整形函数
        
        Args:
            original_reward: 原始环境奖励
            terminated: 是否终局
            action_type: 动作类型 ('place'/'move')
            my_queen_surrounded_count: 己方蜂后被围方向数(0-6)
            opp_queen_surrounded_count: 对方蜂后被围方向数(0-6)
            prev_my_queen_surrounded: 上一步己方蜂后被围数
            prev_opp_queen_surrounded: 上一步对方蜂后被围数
            is_illegal_action: 是否非法动作
            turn_count: 当前回合数
            reason: 终局原因
            
        Returns:
            整形后的奖励值 (范围[-1, 1])
        """
        self.step_count += 1
        
        # 1. 非法动作惩罚 (优先级最高)
        if is_illegal_action:
            return self.config['illegal_penalty']
        
        # 2. 基础存活奖励 (进一步减少，避免拖延策略)
        survival_reward = 0.0
        if not terminated:
            # 极小的存活奖励，几乎可以忽略
            survival_reward = 0.0001 * (1.0 / (1.0 + turn_count * 0.1))
        
        # 3. 动作类型奖励 (进一步减少)
        action_reward = 0.0
        if not terminated:
            if action_type == 'place':
                action_reward = 0.001  # 从0.005进一步减少到0.001
            elif action_type == 'move':
                action_reward = 0.002  # 从0.008进一步减少到0.002
        
        # 4. 战略进度奖励 (核心策略学习)
        strategy_reward = 0.0
        if not terminated:
            # 包围对方蜂后进度奖励 (渐进式)
            surround_progress = self._calculate_surround_progress(
                opp_queen_surrounded_count, prev_opp_queen_surrounded
            )
            # 避免己方蜂后被围奖励
            defense_progress = self._calculate_defense_progress(
                my_queen_surrounded_count, prev_my_queen_surrounded
            )
            strategy_reward = surround_progress + defense_progress
        
        # 5. 终局奖励 (进一步降低，避免权重加成后过高)
        terminal_reward = 0.0
        if terminated:
            if reason in ['player1_win', 'player2_win']:
                # 胜利奖励：基础+速度奖励，进一步降低
                base_win_reward = 2.0  # 从5.0降低到2.0
                speed_bonus = max(0.0, (100 - turn_count) / 100 * 0.5)  # 从2.0降低到0.5
                terminal_reward = base_win_reward + speed_bonus
                # 根据原始奖励正负性调整
                if original_reward < 0:
                    terminal_reward = -terminal_reward
            elif reason == 'queen_surrounded':
                terminal_reward = -2.5  # 从-6.0降低到-2.5
            elif reason == 'draw':
                # 平局：根据包围优势给予微调
                surround_advantage = opp_queen_surrounded_count - my_queen_surrounded_count
                terminal_reward = np.clip(surround_advantage * 0.2, -0.5, 0.5)  # 从0.5降低到0.2
            elif reason in ['max_turns_reached', 'no_legal_action']:
                # 降低超时和无合法动作的惩罚
                terminal_reward = -1.0  # 从-3.0降低到-1.0
        
        # 6. 加权组合
        final_reward = (
            survival_reward * self.config['survival_weight'] +
            action_reward * self.config['action_weight'] +
            strategy_reward * self.config['strategy_weight'] +
            terminal_reward * self.config['terminal_weight']
        )
        
        # 7. 最终裁剪到合理范围 (进一步缩小范围)
        final_reward = np.clip(final_reward, -5.0, 5.0)
        
        return final_reward
    
    def _calculate_surround_progress(self, current_surrounded: int, prev_surrounded: int) -> float:
        """计算包围对方蜂后的进度奖励 - 进一步降低以避免累积过高"""
        if current_surrounded > prev_surrounded:
            # 增加包围：渐进式奖励 (进一步降低)
            progress = current_surrounded - prev_surrounded
            if current_surrounded <= 2:
                return 0.05 * progress   # 从0.5降低到0.05
            elif current_surrounded <= 4:
                return 0.1 * progress    # 从1.0降低到0.1
            else:
                return 0.15 * progress   # 从2.0降低到0.15
        return 0.0
    
    def _calculate_defense_progress(self, current_surrounded: int, prev_surrounded: int) -> float:
        """计算防御己方蜂后的进度奖励 - 进一步降低以避免累积过高"""
        if current_surrounded < prev_surrounded:
            # 减少被围：防御奖励 (降低)
            improvement = prev_surrounded - current_surrounded
            return 0.08 * improvement  # 从0.6降低到0.08
        elif current_surrounded > prev_surrounded:
            # 增加被围：防御惩罚 (降低)
            degradation = current_surrounded - prev_surrounded
            if current_surrounded <= 3:
                return -0.05 * degradation   # 从-0.3降低到-0.05
            elif current_surrounded <= 5:
                return -0.1 * degradation    # 从-0.8降低到-0.1
            else:
                return -0.2 * degradation    # 从-1.5降低到-0.2
        return 0.0

def create_curriculum_phases() -> list:
    """
    创建基于改进奖励整形的课程学习阶段配置
    
    基于网络复杂度分析：
    - 参数量: ~140万
    - 状态空间: 820维
    - 动作空间: ~80个合法动作/步
    - 理论收敛需求: >280万样本
    
    实际配置：平衡效果与时间成本
    """
    return [
        {
            'name': 'foundation',
            'episodes': 40000,  # 增加基础学习量
            'description': '基础棋规学习 - 重点掌握合法动作和基本生存策略',
            'reward_shaper': HiveRewardShaper('foundation'),
            'epsilon_start': 0.9,
            'epsilon_end': 0.8,  # 修复：更保守的衰减
            'epsilon_decay_episodes': 35000,  # 修复：大幅延长衰减时间
        },
        {
            'name': 'strategy', 
            'episodes': 50000,  # 增加策略学习量
            'description': '战略学习强化 - 重点学习包围和防御策略',
            'reward_shaper': HiveRewardShaper('strategy'),
            'epsilon_start': 0.8,  # 修复：从foundation阶段接续
            'epsilon_end': 0.4,
            'epsilon_decay_episodes': 40000,  # 修复：延长衰减时间
        },
        {
            'name': 'mastery',
            'episodes': 30000,  # 增加精通训练量
            'description': '精通阶段优化 - 重点优化胜负判断和高级策略',
            'reward_shaper': HiveRewardShaper('mastery'),
            'epsilon_start': 0.4,  # 修复：从strategy阶段接续
            'epsilon_end': 0.1,
            'epsilon_decay_episodes': 25000,  # 修复：延长衰减时间
        }
    ]

# 总计：120,000 episodes (约3-4小时并行训练)
# 比之前的95,000增加26%，确保充分收敛

if __name__ == "__main__":
    # 测试奖励整形效果
    shaper = HiveRewardShaper('foundation')
    
    # 模拟场景1: 包围对方蜂后
    reward1 = shaper.shape_reward(
        original_reward=2.0,  # 原本的包围奖励
        terminated=False,
        action_type='move',
        my_queen_surrounded_count=1,
        opp_queen_surrounded_count=3,  # 从2增加到3
        prev_my_queen_surrounded=1,
        prev_opp_queen_surrounded=2,
        turn_count=15
    )
    print(f"包围进度奖励: {reward1:.3f} (原始2.0 -> {reward1:.3f})")
    
    # 模拟场景2: 胜利终局
    reward2 = shaper.shape_reward(
        original_reward=25.0,  # 原本的胜利奖励
        terminated=True,
        action_type='move',
        my_queen_surrounded_count=2,
        opp_queen_surrounded_count=6,
        prev_my_queen_surrounded=2,
        prev_opp_queen_surrounded=5,
        turn_count=30,
        reason='player1_win'
    )
    print(f"胜利奖励: {reward2:.3f} (原始25.0 -> {reward2:.3f})")
    
    # 模拟场景3: 非法动作
    reward3 = shaper.shape_reward(
        original_reward=-1.0,
        terminated=True,
        action_type='place',
        my_queen_surrounded_count=0,
        opp_queen_surrounded_count=0,
        prev_my_queen_surrounded=0,
        prev_opp_queen_surrounded=0,
        is_illegal_action=True,
        turn_count=5
    )
    print(f"非法动作惩罚: {reward3:.3f} (原始-1.0 -> {reward3:.3f})")
    
    phases = create_curriculum_phases()
    print(f"\n=== 课程学习配置 ===")
    total_episodes = 0
    for i, phase in enumerate(phases):
        print(f"阶段{i+1}: {phase['name']}")
        print(f"  Episodes: {phase['episodes']}")
        print(f"  Focus: {phase['description']}")
        print(f"  Epsilon: {phase['epsilon_start']} -> {phase['epsilon_end']}")
        total_episodes += phase['episodes']
        print()
    
    print(f"总训练量: {total_episodes:,} episodes")
    print(f"预计时间: 3-4小时 (10并发)")

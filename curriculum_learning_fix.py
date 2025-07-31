#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
curriculum_learning_fix.py

修复课程学习中的训练退化问题的解决方案
"""

import os
import json
import numpy as np
import torch
import copy

class AntiCatastrophicForgettingTrainer:
    """
    防止灾难性遗忘的课程学习训练器
    """
    
    def __init__(self, base_trainer):
        self.base_trainer = base_trainer
        self.phase_models = {}  # 保存每个阶段的最佳模型
        self.phase_buffers = {}  # 每个阶段独立的经验池
        self.knowledge_distillation_enabled = True
        
    def curriculum_train_fixed(self):
        """
        修复后的课程学习训练方法
        """
        from improved_reward_shaping import create_curriculum_phases_fixed
        phases = create_curriculum_phases_fixed()
        
        for phase_idx, phase in enumerate(phases):
            print(f"\n[Curriculum-Fixed] 阶段 {phase_idx+1}: {phase['name']}")
            
            # 1. 保存上一阶段的最佳模型
            if phase_idx > 0:
                self._save_phase_model(phase_idx - 1)
            
            # 2. 清理经验池，避免不同阶段奖励信号污染
            self._reset_experience_buffer(phase['name'])
            
            # 3. 使用渐进式学习率
            self._adjust_learning_rate_for_phase(phase_idx)
            
            # 4. 训练当前阶段
            self._train_phase_with_stabilization(phase, phase_idx)
            
            # 5. 知识蒸馏：保持前一阶段的核心知识
            if phase_idx > 0 and self.knowledge_distillation_enabled:
                self._knowledge_distillation(phase_idx)
                
        print("[Curriculum-Fixed] 所有阶段训练完成")
    
    def _save_phase_model(self, phase_idx):
        """保存阶段最佳模型"""
        phase_model_path = os.path.join(
            self.base_trainer.model_dir, 
            f"phase_{phase_idx}_best.pth"
        )
        # 深拷贝模型状态
        model_state = copy.deepcopy(self.base_trainer.player1_ai.neural_network.state_dict())
        torch.save(model_state, phase_model_path)
        self.phase_models[phase_idx] = phase_model_path
        print(f"[Phase-Save] 阶段 {phase_idx} 模型已保存")
    
    def _reset_experience_buffer(self, phase_name):
        """重置经验池，避免不同阶段数据污染"""
        if hasattr(self.base_trainer.player1_ai, 'replay_buffer'):
            # 保存当前阶段的重要经验（可选）
            if len(self.base_trainer.player1_ai.replay_buffer) > 1000:
                # 保留少量高质量经验作为"知识种子"
                important_experiences = self.base_trainer.player1_ai.replay_buffer[-500:]
                self.phase_buffers[phase_name] = important_experiences
            
            # 清空经验池
            self.base_trainer.player1_ai.replay_buffer.clear()
            print(f"[Buffer-Reset] {phase_name} 阶段经验池已重置")
    
    def _adjust_learning_rate_for_phase(self, phase_idx):
        """为不同阶段调整学习率，防止过度学习"""
        # 后期阶段使用更小的学习率，保持稳定性
        if phase_idx == 0:
            lr = 0.001  # 基础阶段：正常学习率
        elif phase_idx == 1:
            lr = 0.0008  # 策略阶段：略微降低
        else:
            lr = 0.0005  # 精通阶段：保守学习率
            
        # 更新优化器学习率
        for param_group in self.base_trainer.player1_ai.optimizer.param_groups:
            param_group['lr'] = lr
        print(f"[LR-Adjust] 阶段 {phase_idx} 学习率设置为: {lr}")
    
    def _train_phase_with_stabilization(self, phase, phase_idx):
        """带稳定化机制的阶段训练"""
        # 记录训练前的性能基线
        baseline_performance = self._evaluate_performance() if phase_idx > 0 else None
        
        # 训练
        self.base_trainer.train(
            max_episodes=phase['episodes'],
            curriculum_epsilon_config={
                'start': phase['epsilon_start'],
                'end': phase['epsilon_end'],
                'decay_episodes': phase.get('epsilon_decay_episodes', 5000),
                'total_episodes_before': len(self.base_trainer.average_rewards)
            }
        )
        
        # 检查是否发生退化
        if baseline_performance and phase_idx > 0:
            current_performance = self._evaluate_performance()
            if current_performance < baseline_performance * 0.8:  # 性能下降20%以上
                print(f"[Degradation-Detected] 阶段 {phase_idx} 发生训练退化!")
                self._recover_from_degradation(phase_idx)
    
    def _knowledge_distillation(self, current_phase_idx):
        """知识蒸馏：保持前一阶段的核心能力"""
        if current_phase_idx - 1 not in self.phase_models:
            return
            
        print(f"[Knowledge-Distillation] 开始阶段 {current_phase_idx} 的知识蒸馏")
        
        # 加载前一阶段的教师模型
        teacher_model_path = self.phase_models[current_phase_idx - 1]
        teacher_state = torch.load(teacher_model_path)
        
        # 创建教师网络
        teacher_network = copy.deepcopy(self.base_trainer.player1_ai.neural_network)
        teacher_network.load_state_dict(teacher_state)
        teacher_network.eval()
        
        # 蒸馏训练：学生网络学习教师网络的输出
        student_network = self.base_trainer.player1_ai.neural_network
        
        # 简化的蒸馏过程：在重要状态上保持一致性
        distillation_loss_weight = 0.3  # 蒸馏损失权重
        
        for _ in range(100):  # 少量蒸馏步骤
            if len(self.base_trainer.player1_ai.replay_buffer) > 32:
                # 从经验池采样
                batch = self.base_trainer.player1_ai.replay_buffer[-32:]
                states = np.array([exp[0] for exp in batch])
                
                # 获取教师和学生的预测
                with torch.no_grad():
                    teacher_predictions = teacher_network.predict_batch(states)
                student_predictions = student_network.predict_batch(states)
                
                # 计算蒸馏损失
                distillation_loss = torch.nn.functional.mse_loss(
                    student_predictions, 
                    torch.tensor(teacher_predictions, dtype=torch.float32)
                )
                
                # 反向传播
                self.base_trainer.player1_ai.optimizer.zero_grad()
                (distillation_loss * distillation_loss_weight).backward()
                self.base_trainer.player1_ai.optimizer.step()
        
        print(f"[Knowledge-Distillation] 阶段 {current_phase_idx} 蒸馏完成")
    
    def _evaluate_performance(self):
        """简单的性能评估"""
        if len(self.base_trainer.average_rewards) < 100:
            return 0.0
        # 返回最近100个episode的平均奖励
        return np.mean(self.base_trainer.average_rewards[-100:])
    
    def _recover_from_degradation(self, phase_idx):
        """从训练退化中恢复"""
        print(f"[Recovery] 尝试从阶段 {phase_idx} 的训练退化中恢复")
        
        if phase_idx - 1 in self.phase_models:
            # 回滚到前一阶段的模型
            teacher_model_path = self.phase_models[phase_idx - 1]
            teacher_state = torch.load(teacher_model_path)
            
            # 部分回滚：只恢复部分网络权重
            current_state = self.base_trainer.player1_ai.neural_network.state_dict()
            
            # 混合权重：70%当前 + 30%前一阶段
            mixed_state = {}
            for key in current_state:
                mixed_state[key] = 0.7 * current_state[key] + 0.3 * teacher_state[key]
            
            self.base_trainer.player1_ai.neural_network.load_state_dict(mixed_state)
            print("[Recovery] 已应用权重混合恢复")


def create_curriculum_phases_fixed():
    """
    修复后的课程学习阶段配置
    重点：渐进式复杂度增长，避免奖励权重跳跃
    """
    from improved_reward_shaping import HiveRewardShaper
    
    return [
        {
            'name': 'foundation',
            'episodes': 30000,
            'description': '基础规则学习 - 专注合法动作和基本生存',
            'reward_shaper': HiveRewardShaperFixed('foundation'),
            'epsilon_start': 0.9,
            'epsilon_end': 0.7,
            'epsilon_decay_episodes': 8000,
        },
        {
            'name': 'strategy',
            'episodes': 40000,
            'description': '战略发展 - 学习攻防平衡',
            'reward_shaper': HiveRewardShaperFixed('strategy'),
            'epsilon_start': 0.7,
            'epsilon_end': 0.4,
            'epsilon_decay_episodes': 10000,
        },
        {
            'name': 'mastery',
            'episodes': 50000,
            'description': '高级策略 - 精通复杂局面',
            'reward_shaper': HiveRewardShaperFixed('mastery'),
            'epsilon_start': 0.4,
            'epsilon_end': 0.1,
            'epsilon_decay_episodes': 15000,
        }
    ]


class HiveRewardShaperFixed:
    """
    修复后的奖励整形器 - 防止权重跳跃
    """
    
    def __init__(self, phase: str = 'foundation'):
        self.phase = phase
        
        # 修复后的渐进式权重配置
        self.phase_configs = {
            'foundation': {
                'survival_weight': 0.05,     # 适度的存活权重
                'action_weight': 0.15,       # 重视基础动作学习
                'strategy_weight': 0.20,     # 基础阶段战略权重较低
                'terminal_weight': 0.60,     # 重视终局结果
                'illegal_penalty': -5.0,
            },
            'strategy': {
                'survival_weight': 0.03,     # 逐步降低存活权重
                'action_weight': 0.10,       # 降低动作权重
                'strategy_weight': 0.40,     # 逐步提升战略权重
                'terminal_weight': 0.47,     # 保持终局权重
                'illegal_penalty': -6.0,
            },
            'mastery': {
                'survival_weight': 0.02,     # 最低存活权重
                'action_weight': 0.08,       # 最低动作权重
                'strategy_weight': 0.50,     # 最高战略权重
                'terminal_weight': 0.40,     # 适度终局权重
                'illegal_penalty': -8.0,
            }
        }
        
        self.config = self.phase_configs.get(phase, self.phase_configs['foundation'])
    
    def shape_reward(self, **kwargs):
        """使用修复后的权重配置进行奖励整形"""
        # 这里可以复用原有的shape_reward逻辑，但使用新的权重配置
        pass

# Hive-RL 开发者快速上手指南

## 🎯 项目概览

Hive-RL 是一个基于深度强化学习的蜂巢棋（Hive）AI项目，采用DQN算法实现智能体训练。**当前面临的核心问题是训练奖励震荡且2500轮后累计平均奖励几乎不变化**。

### 关键技术栈
- **强化学习算法**: DQN (Deep Q-Network)
- **深度学习框架**: PyTorch + NumPy备用实现
- **并行训练**: multiprocessing (10 workers)
- **游戏环境**: 自研Hive环境 (OpenAI Gym接口)
- **奖励塑形**: 三阶段课程学习系统

---

## 🚀 快速启动

### 1. 环境安装
```bash
pip install -r requirements.txt
```

核心依赖：
- `torch>=1.12.0` (神经网络)
- `gymnasium>=0.28.1` (强化学习环境)
- `numba>=0.56.4` (性能优化)
- `numpy>=1.21.0` + `matplotlib>=3.5.0`

### 2. 运行项目
```bash
python main.py
```

菜单选项：
- **选项3**: AI训练 (推荐使用课程学习)
- **选项4**: 评估AI并绘制统计图表
- **选项2**: 人机对战测试

---

## 🏗️ 核心架构

### 文件结构分析
```
核心训练模块：
├── ai_trainer.py          # 主训练器（多进程+课程学习）
├── ai_player.py           # DQN智能体实现
├── hive_env.py           # 游戏环境（Gym接口）
├── neural_network_torch.py # PyTorch神经网络
└── improved_reward_shaping.py # 奖励塑形系统

游戏逻辑：
├── game.py               # 游戏状态管理
├── board.py              # 棋盘逻辑
├── player.py             # 玩家基类
├── piece.py              # 棋子类型定义
└── utils.py              # 常量配置

评估工具：
├── evaluate.py           # 综合评估脚本
├── ai_evaluator.py       # 对战评估器
├── benchmark.py          # 基准测试
└── plot_*.py             # 各种绘图工具
```

### 神经网络架构
```python
# 当前网络结构 (ai_player.py:26-28)
input_dim = 820 + BOARD_SIZE * 4 + len(PIECE_TYPE_LIST)  # ~868维
hidden_dims = [1024, 512]  # 两层隐藏层
output_dim = 1  # Q值输出
```

**状态空间**: 820维
- 棋盘状态: 800维 (10×10×8种棋子)
- 手牌信息: 16维 (2玩家×8种棋子)
- 游戏状态: 4维 (当前玩家、回合数、蜂后放置状态)

**动作空间**: 离散20000种
- 放置动作: `x*1000 + y*100 + piece_type` (0-9999)
- 移动动作: `10000 + from_x*1000 + from_y*100 + to_x*10 + to_y` (10000-19999)

---

## ⚠️ 当前问题诊断

### 训练收敛问题
基于代码分析和文档，当前训练存在以下问题：

1. **奖励震荡严重**
   - 原因：epsilon衰减过快 + 奖励设计不够平滑
   - 位置：`ai_trainer.py:310-380` 的epsilon管理逻辑

2. **规则违规率高** (~46.3%)
   - 问题：`must_place_queen_violation` 比例过高
   - 影响：大量游戏提前结束，无法学习有效策略

3. **奖励塑形权重不平衡**
   - Foundation阶段：过度重视终局奖励 vs 基础规则学习
   - 代码位置：`improved_reward_shaping.py:25-50`

### 具体数据表现
```
# 训练日志显示的问题模式
Episode 5400: Epsilon=0.4147 (应该约0.885)
must_place_queen_violation: 46.3%
queen_surrounded 事件: 0% (说明AI未学会包围策略)
累计平均奖励: 2500轮后平台期
```

---

## 🔧 修复建议与代码定位

### 1. 调整Epsilon衰减策略

**问题位置**: `ai_trainer.py:330-350`
```python
# 当前问题：衰减过快
if curriculum_epsilon_config:
    # epsilon计算逻辑可能有误
    episodes_in_phase = episode  # ❌ 应该是相对阶段的episode数
```

**修复方向**:
- Foundation阶段: epsilon 0.9→0.8 (35000轮)
- Strategy阶段: epsilon 0.8→0.4 (40000轮)  
- Mastery阶段: epsilon 0.4→0.1 (25000轮)

### 2. 重新设计奖励权重

**问题位置**: `improved_reward_shaping.py:25-50`
```python
'foundation': {
    'survival_weight': 0.02,     # 太低
    'action_weight': 0.05,       # 太低  
    'strategy_weight': 0.50,     # 对Foundation阶段过高
    'terminal_weight': 0.43,     # 基础阶段不应过度重视终局
    'illegal_penalty': -8.0,     # 可能需要进一步调整
}
```

**修复思路**:
- Foundation阶段应该重视基础规则学习，降低strategy_weight
- 增加正向激励权重 (action_weight, survival_weight)

### 3. 优化经验回放机制

**问题位置**: `ai_player.py:130-160` 的 `train_on_batch`
```python
# 当前只过滤reward > -2.0的样本
legal_samples = [exp for exp in self.replay_buffer if exp[2] > -2.0]
```

**改进方向**:
- 实现优先经验回放 (PER)
- 动态调整合法样本阈值
- 添加样本重要性加权

### 4. 神经网络优化

**当前结构问题**:
- 网络可能过大（140万参数）导致过拟合
- 学习率可能不合适
- 缺少正则化

**代码位置**: `neural_network_torch.py` + `ai_player.py:26`

---

## 🧪 调试与测试工具

### 1. 训练监控
```bash
# 查看最新训练统计
python plot_reward_curve.py models/最新模型目录/前缀_reward_history.npy

# 综合评估
python evaluate.py
```

### 2. 关键调试输出
```python
# ai_trainer.py 中的调试信息
Episode XXX, Cumulative: XX.XX, Steps: XX, Reason: XXX, Epsilon: X.XXXX

# 关注指标：
- must_place_queen_violation 比例（目标 <10%）
- queen_surrounded 事件出现（目标 >5%）
- epsilon 衰减是否按预期
```

### 3. 模型对战测试
```bash
# 与基准策略对战
python benchmark.py models/某个模型目录/
```

---

## 📊 性能基准

### 训练目标
- **违规率**: must_place_queen_violation < 10%
- **策略学习**: queen_surrounded 事件 > 5%  
- **收敛速度**: 40K轮内达到稳定胜率
- **对战胜率**: vs Random > 80%, vs Heuristic > 60%

### 计算资源
- **并行度**: 10 workers (可调整)
- **训练时长**: ~3-4小时 (120K episodes)
- **内存需求**: ~2GB (经验回放缓冲区)

---

## 🛠️ 开发工作流

### 1. 分支管理
```bash
# 建议为重大修改创建分支
git checkout -b fix/reward-oscillation
git checkout -b feature/per-experience-replay
```

### 2. 实验记录
每次修改需要记录：
- 修改的超参数
- 预期效果
- 实际结果对比
- 训练曲线截图

### 3. 代码规范
- 修改超参数时注释原值和修改理由
- 添加调试输出时使用统一格式：`[DEBUG][模块名]`
- 重要修改需要更新相应的markdown文档

---

## 🔍 下一步优化方向

### 立即行动项
1. **修复epsilon衰减逻辑** (优先级: 高)
2. **重新平衡奖励权重** (优先级: 高)
3. **添加更详细的训练监控** (优先级: 中)

### 中期改进
1. **实现优先经验回放 (PER)**
2. **网络架构优化** (考虑使用更小的网络)
3. **添加对抗训练机制**

### 长期目标
1. **迁移到更先进的算法** (PPO, A3C)
2. **多智能体自对弈训练**
3. **图神经网络应用** (利用棋盘结构特性)

---

## 📞 协作指南

### 问题反馈
- **Bug报告**: 请包含完整的训练日志和复现步骤
- **性能问题**: 提供训练曲线图和具体数据
- **算法讨论**: 基于代码实现和理论分析

### 代码贡献
- 遵循现有代码风格
- 添加适当的注释和文档
- 确保向后兼容性
- 提供测试用例

---

*最后更新: 2025-07-28*
*主要贡献者: GitHub Copilot*

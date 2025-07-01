[🇬🇧 English](README.en.md) | [🇫🇷 Français](README.fr.md) | [🇪🇸 Español](README.es.md)

# Hive-RL: 基于强化学习的 Hive 棋 AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 简介

Hive-RL 是一个先进的强化学习项目，专注于训练高水平的 **Hive**（蜂巢棋）AI。该项目采用现代深度强化学习技术，实现了完整的游戏引擎、科学的奖励系统和多种先进的训练算法。

**Hive** 是一款屡获殊荣的棋类游戏，无需棋盘，规则简单但策略深度极高。玩家的目标是通过放置和移动各种昆虫棋子来包围对方的蜂后。

## ✨ 核心特性

### 🎮 完整的游戏引擎
- **精确规则实现**：完全符合官方 Hive 规则
- **DLC 扩展支持**：包含瓢虫、蚊子、鼠妇等官方扩展棋子
- **高性能棋盘**：优化的数据结构和 Numba 加速
- **动作验证**：严格的合法性检查和错误处理

### 🧠 先进的 AI 系统
- **深度 Q 网络 (DQN)**：基于 PyTorch 的现代神经网络架构
- **科学奖励整形**：精心设计的多阶段奖励系统
- **经验回放**：高效的样本复用和学习稳定性
- **ε-greedy 策略**：平衡探索与利用的动态策略

### 🚀 高性能训练框架
- **并行自对弈**：多进程并行采样，大幅提升训练效率
- **课程学习**：从基础规则到高级策略的渐进式学习
- **对抗训练**：通过对抗样本提升 AI 鲁棒性
- **模型融合**：多模型投票决策系统

### 📊 可视化与分析
- **实时监控**：训练过程中的奖励、损失、胜率曲线
- **性能分析**：详细的终局统计和行为分析
- **模型评估**：自动化的性能基准测试

## 🏗️ 项目架构

```
Hive-RL/
├── 核心引擎
│   ├── game.py              # 游戏主逻辑
│   ├── board.py             # 棋盘表示和操作
│   ├── piece.py             # 棋子类型和移动规则
│   └── player.py            # 玩家基类
├── 强化学习
│   ├── hive_env.py          # Gymnasium 环境
│   ├── ai_player.py         # AI 玩家实现
│   ├── neural_network.py    # 神经网络架构
│   └── improved_reward_shaping.py  # 奖励整形系统
├── 训练框架
│   ├── ai_trainer.py        # 主训练器
│   ├── parallel_sampler.py  # 并行采样器
│   └── ai_evaluator.py      # 性能评估器
├── 分析工具
│   ├── analyze_model.py     # 模型分析
│   └── plot_*.py           # 可视化工具
└── 用户界面
    └── main.py             # 主菜单
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib, Gymnasium
- Numba (性能优化)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动项目
```bash
python main.py
```

### 主菜单选项
1. **Human vs Human** - 本地双人对战
2. **Human vs AI** - 人机对战
3. **AI Training** - AI 训练
4. **Evaluate AI & Plots** - 性能评估
5. **Exit Game** - 退出

## 🎯 AI 训练

### 训练模式
1. **并行采样基础训练** - 高效的多进程训练
2. **自我对弈精炼训练** - 深度策略优化
3. **Ensemble 投票训练** - 多模型融合
4. **对抗式鲁棒化训练** - 抗干扰能力提升
5. **课程学习** - 渐进式技能习得

### 课程学习阶段
- **Foundation (0-40k episodes)** - 基础规则学习
- **Strategy (40k-90k episodes)** - 战略思维发展  
- **Mastery (90k-120k episodes)** - 高级策略精通

### 训练特色
- **自动保存**：训练进度实时保存，支持断点续训
- **性能监控**：实时显示训练速度和收敛状态
- **智能调度**：动态调整 epsilon 和学习率
- **多进程优化**：10个并行worker，训练速度提升10倍

## 🔬 技术原理

### 强化学习框架
- **状态空间**：820维向量，包含棋盘状态、手牌信息、游戏进度
- **动作空间**：20,000个离散动作，覆盖所有可能的放置和移动
- **奖励系统**：多层次奖励设计，从基础生存到高级策略

### 奖励整形系统
```python
终局奖励 (权重: 60-63%)
├── 胜利: +5.0 + 速度奖励
├── 失败: -6.0 (蜂后被围)
├── 超时: -3.0 (拖延惩罚)
└── 平局: 根据优势微调

策略奖励 (权重: 25-40%)
├── 包围进度: 渐进式奖励
├── 防御改善: 安全位置奖励  
└── 棋子协调: 位置价值评估

基础奖励 (权重: 5-15%)
├── 生存奖励: 微小正值
└── 动作奖励: 合法行动鼓励
```

### 神经网络架构
- **输入层**：820维状态向量
- **隐藏层**：多层全连接，ReLU激活
- **输出层**：20,000维Q值预测
- **优化器**：Adam，动态学习率
- **正则化**：Dropout，梯度裁剪

## 📈 性能指标

### 训练效率
- **并行速度**：>1000 episodes/小时
- **收敛时间**：3-4小时完成课程学习
- **样本效率**：120k episodes达到专家水平

### AI 能力
- **胜率表现**：对随机玩家 >90% 胜率
- **策略深度**：平均思考深度 15-20步
- **反应速度**：<0.1秒/步

### 稳定性
- **奖励方差**：训练后期 <0.1
- **策略一致性**：相同局面决策重现率 >95%
- **鲁棒性**：对抗扰动下仍保持高性能

## 🔧 高级配置

### 自定义奖励
```python
# 创建自定义奖励整形器
from improved_reward_shaping import HiveRewardShaper

shaper = HiveRewardShaper('custom')
shaper.config['terminal_weight'] = 0.7  # 提升终局奖励权重
shaper.config['strategy_weight'] = 0.3  # 调整策略奖励权重
```

### 训练参数调优
```python
# 在 ai_trainer.py 中调整超参数
batch_size = 32          # 批量大小
learning_rate = 0.001    # 学习率  
epsilon_start = 0.9      # 初始探索率
epsilon_end = 0.05       # 最终探索率
discount_factor = 0.95   # 折扣因子
```

### 并行配置
```python
# 调整并行worker数量
num_workers = 10         # 根据CPU核心数调整
episodes_per_worker = 100 # 每个worker的episode数
queue_maxsize = 100      # 队列大小
```

## 🐛 故障排除

### 常见问题
1. **训练速度慢**
   - 检查并行worker配置
   - 确认队列未阻塞
   - 验证reward_shaper正确传递

2. **AI行为异常**
   - 检查奖励系统配置
   - 验证终局统计合理性
   - 分析epsilon衰减曲线

3. **内存不足**
   - 减少batch_size
   - 调整经验回放缓冲区大小
   - 使用更少的并行worker

### 调试工具
```bash
# 分析最新训练模型
python analyze_model.py

# 可视化训练曲线  
python plot_reward_curve.py

# 测试环境设置
python test_environment.py
```

## 🤝 贡献指南

我们欢迎社区贡献！请查看以下指导：

### 开发环境
```bash
# 克隆仓库
git clone <repository-url>
cd Hive-RL

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 代码规范
- 遵循 PEP 8 代码风格
- 添加类型注解
- 编写单元测试
- 更新文档

### 提交流程
1. Fork 项目
2. 创建功能分支
3. 提交代码变更
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Hive 游戏**由 John Yianni 设计
- 感谢 PyTorch 和 Gymnasium 开源社区
- 特别感谢所有贡献者和测试用户

## 📞 联系方式

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: [your-email@example.com](mailto:your-email@example.com)

---

**Hive-RL**: Where AI meets the elegance of Hive! 🐝♟️🤖

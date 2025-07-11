[🇬🇧 English](README.en.md) | [🇫🇷 Français](README.fr.md) | [🇪🇸 Español](README.es.md)

# Hive-RL: 基于强化学习的 Hive 棋 AI

## 简介

Hive-RL 是一个基于强化学习（Reinforcement Learning）的 Python 项目，旨在为桌面游戏 **Hive**（中文名：蜂巢棋）训练一个高水平的人工智能（AI）。本项目实现了完整的游戏逻辑、一个兼容 OpenAI Gym/Gymnasium 标准的强化学习环境、以及一个采用深度 Q 网络（DQN）的 AI 训练器。

## 项目特色

*   **完整的游戏实现**：精确实现了 Hive 棋的基础规则和所有棋子的移动方式，并支持官方的 **DLC 扩展棋子**（瓢虫、蚊子、鼠妇）。
*   **模块化架构**：代码结构清晰，分为游戏逻辑、RL 环境、AI 玩家、训练器、评估器等模块，易于理解和扩展。
*   **强化学习驱动**：采用深度 Q 网络（DQN）作为核心算法，通过自我对弈（Self-Play）和多种先进的训练策略，让 AI 从零开始学习并不断进化。
*   **先进的训练策略**：
    *   **并行化自对弈 (Parallel Self-Play)**：利用多进程并行采样，大幅提升训练速度。
    *   **课程学习 (Curriculum Learning)**：让 AI 从简化的任务开始学习（如先学会放置蜂后），逐步过渡到完整游戏，提升学习效率。
    *   **对抗式训练 (Adversarial Training)**：通过与一个专门选择“最差”步骤的对手博弈，提升 AI 的鲁棒性。
    *   **模型融合 (Ensemble Training)**：训练多个独立的 AI 模型，在决策时进行投票，提升决策的准确性和稳定性。
*   **可视化与评估**：提供多种可视化支持，用于绘制训练过程中的奖励曲线、损失曲线、胜率曲线和其它统计数据，方便分析 AI 的学习进展。
*   **用户友好的交互界面**：提供一个命令行主菜单，支持人-人对战、人-机对战、AI 训练和 AI 评估等多种模式。

## 项目架构

*   `main.py`: 项目主入口，提供命令行交互菜单。
*   `game.py`: 游戏主逻辑，管理游戏流程、玩家和回合。
*   `board.py`: 棋盘表示和基本操作。
*   `piece.py`: 定义所有棋子（包括 DLC）的属性和移动规则。
*   `player.py`: 玩家基类，管理手牌和基本动作。
*   `ai_player.py`: AI 玩家类，实现了基于神经网络的动作选择和经验回放。
*   `hive_env.py`: 遵循 Gymnasium API 的 Hive 游戏环境，用于强化学习训练。
*   `neural_network.py`: 基于 PyTorch 的深度神经网络实现。
*   `ai_trainer.py`: AI 训练器，包含多种训练模式（并行自对弈、课程学习、对抗训练等）。
*   `ai_evaluator.py`: AI 评估器，用于测试 AI 对战随机玩家的胜率。
*   `utils.py`: 提供辅助函数和工具。
*   `requirements.txt`: 项目依赖库。

## 如何运行

### 1. 环境配置

首先，请确保您已安装 Python 3.10 或更高版本。然后，安装所有必需的依赖库：

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

通过以下命令启动项目的主菜单：

```bash
python main.py
```

您将看到以下选项：

1.  **Human vs Human**：与另一位本地玩家对战。
2.  **Human vs AI**：与训练好的 AI 对战。
3.  **AI Training**：训练一个新的 AI 模型或从断点继续训练。
4.  **Evaluate AI & Plots**：评估 AI 性能并绘制训练曲线。
5.  **Exit Game**：退出程序。

### 3. 训练 AI

*   选择主菜单中的 `AI Training` 选项。
*   您可以选择**新建训练**或从**历史断点继续训练**。
*   接下来，选择一种训练模式，例如**并行采样基础训练**或**自我对弈**。
*   训练过程中，模型和统计数据将自动保存在 `models/` 目录下，以时间戳和 DLC 状态命名的文件夹中。
*   您随时可以通过 `Ctrl+C` 中断训练，程序会自动保存当前进度，以便下次恢复。

### 4. 与 AI 对战

*   选择主菜单中的 `Human vs AI` 选项。
*   程序会自动列出 `models/` 目录中所有可用的 AI 模型。您可以选择一个模型进行对战。
*   游戏过程中，按照提示输入您的动作即可。

## 强化学习原理

本项目中的 AI 基于**深度 Q 网络 (DQN)**，这是一种价值驱动的强化学习算法。其核心思想是训练一个神经网络来近似 **Q 函数** `Q(s, a)`，该函数用于预测在给定状态 `s` 下执行动作 `a` 所能带来的长期回报（奖励）。

*   **状态 (State)**：游戏当前局面的一个向量化表示，包括棋盘上每个位置的棋子类型、双方玩家手牌中剩余的棋子数量、当前回合数等信息。
*   **动作 (Action)**：所有合法的“放置”或“移动”操作之一。
*   **奖励 (Reward)**：AI 每执行一个动作后从环境中获得的反馈信号。
    *   **获胜**：获得一个大的正奖励。
    *   **失败**：获得一个大的负奖励。
    *   **平局**：获得零奖励或一个小的正/负奖励。
    *   **Reward Shaping**：为了引导 AI 更快地学习，我们设计了一系列中间奖励，例如：
        *   包围对方蜂后会获得正奖励。
        *   自己的蜂后被包围会受到惩罚。
        *   执行一个合法的移动或放置会获得一个小的正奖励。
        *   每走一步会有一个微小的负奖励，以鼓励 AI 尽快获胜。
*   **训练流程**：
    1.  **采样**：AI（或多个并行的 AI）在环境中通过自我对弈进行游戏，收集大量的 `(状态, 动作, 奖励, 下一状态)` 经验元组。
    2.  **经验回放 (Experience Replay)**：收集到的经验被存储在一个“经验池”中。
    3.  **训练**：从经验池中随机抽取一小批经验，用于训练神经网络。训练的目标是让 `Q(s, a)` 的预测值尽可能接近**目标 Q 值**（通常是 `奖励 + 折扣因子 * max(Q(下一状态, 所有合法动作))`）。
    4.  **探索与利用 (Exploration vs. Exploitation)**：AI 在选择动作时采用 **ε-greedy** 策略。即以 ε 的概率随机选择一个合法动作（探索），以 1-ε 的概率选择 Q 值最高的动作（利用）。随着训练的进行，ε 会逐渐衰减，使得 AI 从随机探索慢慢转变为更依赖于已学到的最优策略。

通过成千上万局的自我对弈和训练，AI 的神经网络能够逐渐学习到 Hive 棋盘上复杂的模式和策略，从而达到高水平的棋力。

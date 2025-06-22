# Hive Game Python Project (English)

[中文 | English | Français | Español]


## Project Overview
This project is a Python implementation of Hive, supporting human vs human, human vs AI, AI training, and evaluation. It supports base and DLC pieces, with full board logic, player management, AI decision-making, and neural network training.

## Main Features
- **Human vs Human**: Local two-player mode, full Hive rules.
- **Human vs AI**: Play against AI, which can load a trained model or play randomly.
- **AI Training**: Self-play reinforcement learning to improve AI.
- **AI Evaluation**: Batch evaluation of AI performance.
- **Board & Rules**: Implements all base and DLC pieces, placement/movement rules, and win conditions.

## Architecture
- **Language**: Python 3
- **Main Modules**:
  - `main.py`: Entry point, menu, main loop
  - `game.py`: Game flow and state management
  - `player.py` / `ai_player.py`: Player and AI player objects
  - `board.py`: Board and piece management
  - `hive_env.py` / `game_state.py`: AI environment and state encoding
  - `neural_network.py`: Neural network implementation
  - `ai_trainer.py` / `ai_evaluator.py`: AI training and evaluation
  - `utils.py`: Constants and utilities
- **AI**: Reinforcement learning (Q-Learning/DQN), custom neural network, experience replay
- **Data Structure**: OOP, supports clone/deepcopy/state simulation

## How to Run
1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the game:

   ```bash
   python main.py
   ```

3. Select mode from the menu.

## Use Cases
- Board game development & AI research
- Reinforcement learning & game AI practice
- Python OOP & project architecture learning

## Other Notes
- Supports base and DLC pieces (Ladybug, Mosquito, Pillbug)
- Clean code structure, easy for extension
- See Q&S.md for detailed technical issues and fixes

## Module Structure

This project uses a layered, decoupled OOP architecture:

- **UI & Main Process Layer**
  - `main.py`: Menu, user interaction, main loop, module dispatch
- **Game Logic & State Layer**
  - `game.py`: Game flow, turn switching, win/lose, player management
  - `board.py`: Board data structure, piece placement/movement/validation
  - `player.py`: Human player, inventory, placement/movement
  - `piece.py`: Piece types, attributes, behaviors
- **AI & Environment Layer**
  - `ai_player.py`: AI player, inherits Player, integrates RL and neural network
  - `hive_env.py`: AI training environment, state/action/reward, OpenAI Gym style
  - `game_state.py`: State encoding and feature extraction for AI
- **AI Training & Evaluation Layer**
  - `ai_trainer.py`: Self-play training, experience collection, model update
  - `ai_evaluator.py`: AI evaluation and statistics
  - `neural_network.py`: Custom neural network for value/action prediction
- **Utils & Constants Layer**
  - `utils.py`: Piece constants, helpers, global config

---

## RL Design (Detailed)

The AI uses Q-Learning/DQN with a custom neural network. Key details:

### 1. State Encoding
- Each state is an 814-dim vector:
  - 800: 10x10 board, one-hot for top piece type (8 types)
  - 10: Both players' hand inventory (5 base pieces, normalized)
  - 4: Current player, turn, both queens placed
- See RLTechReport.md for code details.

### 2. Action Space
- All legal placements/moves are encoded as discrete ints (Action.encode_*):
  - Placement: piece type and coordinates
  - Move: from and to coordinates
- AI enumerates all possible actions and filters for legality.
- See RLTechReport.md for code details.

### 3. Neural Network
- Custom MLP:
  - Input: 814
  - Hidden: 256, ReLU
  - Output: 1 (state value, can extend to Q(s,a))
- See RLTechReport.md for code details.

### 4. Policy & Exploration
- Epsilon-greedy:
  - With probability epsilon, pick random action (explore), else pick max-value action (exploit)
  - Epsilon decays (e.g. *0.995 per game), min 0.01
- See RLTechReport.md for code details.

### 5. Experience Replay & Training
- Each step (s, a, r, s', done) goes to replay buffer
- Each training, sample batch, compute targets:
  - If not terminal: target = r + γ * V(s')
  - If terminal: target = r
- Train with MSE loss
- See RLTechReport.md for code details.

### 7. Self-Play & Model Update
- AI self-plays, accumulates experience, saves model every 100 games
- Trained model can be loaded for play/evaluation

### 8. Evaluation & Generalization
- After training, use ai_evaluator.py for batch games, win rate, avg steps
- Supports comparison under different epsilon/model params

---

For suggestions or bug reports, open an issue or PR!

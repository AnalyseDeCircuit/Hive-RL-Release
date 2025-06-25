[🇨🇳 中文](README.md) | [🇫🇷 Français](README.fr.md) | [🇪🇸 Español](README.es.md)

# Hive-RL: An AI for the board game Hive based on Reinforcement Learning

## Introduction

Hive-RL is a Python project based on Reinforcement Learning (RL) that aims to train a high-level Artificial Intelligence (AI) for the board game **Hive**. This project implements the complete game logic, a reinforcement learning environment compatible with the OpenAI Gym/Gymnasium standard, and an AI trainer using a Deep Q-Network (DQN).

## Project Features

*   **Complete Game Implementation**: Accurately implements the basic rules of Hive and the movement of all pieces, including the official **DLC expansion pieces** (Ladybug, Mosquito, Pillbug).
*   **Modular Architecture**: The code is clearly structured into modules for game logic, RL environment, AI player, trainer, evaluator, etc., making it easy to understand and extend.
*   **Reinforcement Learning Driven**: Uses a Deep Q-Network (DQN) as its core algorithm, allowing the AI to learn from scratch and continuously evolve through Self-Play and various advanced training strategies.
*   **Advanced Training Strategies**:
    *   **Parallel Self-Play**: Utilizes multiprocessing to sample in parallel, significantly speeding up training.
    *   **Curriculum Learning**: Allows the AI to start learning from simplified tasks (e.g., learning to place the Queen Bee first) and gradually transition to the full game, improving learning efficiency.
    *   **Adversarial Training**: Enhances the AI's robustness by playing against an opponent that specifically chooses the "worst" moves.
    *   **Ensemble Training**: Trains multiple independent AI models and uses voting during decision-making to improve the accuracy and stability of choices.
*   **Visualization and Evaluation**: Provides various visualization tools to plot reward curves, loss curves, win rate curves, and other statistics during the training process, making it easy to analyze the AI's learning progress.
*   **User-Friendly Interface**: Offers a command-line main menu that supports various modes, including Human vs. Human, Human vs. AI, AI Training, and AI Evaluation.

## Project Architecture

*   `main.py`: The main entry point of the project, providing a command-line interactive menu.
*   `game.py`: The main game logic, managing the game flow, players, and turns.
*   `board.py`: Board representation and basic operations.
*   `piece.py`: Defines the properties and movement rules for all pieces (including DLC).
*   `player.py`: The base class for players, managing hand and basic actions.
*   `ai_player.py`: The AI player class, implementing action selection and experience replay based on a neural network.
*   `hive_env.py`: The Hive game environment following the Gymnasium API, used for reinforcement learning training.
*   `neural_network.py`: A deep neural network implementation based on PyTorch.
*   `ai_trainer.py`: The AI trainer, including various training modes (parallel self-play, curriculum learning, adversarial training, etc.).
*   `ai_evaluator.py`: The AI evaluator, used to test the AI's win rate against a random player.
*   `utils.py`: Provides helper functions and tools.
*   `requirements.txt`: Project dependency libraries.

## How to Run

### 1. Environment Setup

First, make sure you have Python 3.10 or higher installed. Then, install all the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Main Program

Start the project's main menu with the following command:

```bash
python main.py
```

You will see the following options:

1.  **Human vs Human**: Play against another local player.
2.  **Human vs AI**: Play against a trained AI.
3.  **AI Training**: Train a new AI model or continue training from a checkpoint.
4.  **Evaluate AI & Plots**: Evaluate the AI's performance and plot training curves.
5.  **Exit Game**: Exit the program.

### 3. Train the AI

*   Select the `AI Training` option from the main menu.
*   You can choose to **start a new training** or **continue from a previous checkpoint**.
*   Next, select a training mode, such as **basic training with parallel sampling** or **self-play**.
*   During training, the model and statistics will be automatically saved in the `models/` directory, in a folder named with a timestamp and DLC status.
*   You can interrupt the training at any time with `Ctrl+C`, and the program will automatically save the current progress to be resumed later.

### 4. Play Against the AI

*   Select the `Human vs AI` option from the main menu.
*   The program will automatically list all available AI models in the `models/` directory. You can choose one to play against.
*   During the game, enter your moves as prompted.

## Reinforcement Learning Principles

The AI in this project is based on a **Deep Q-Network (DQN)**, a value-driven reinforcement learning algorithm. The core idea is to train a neural network to approximate the **Q-function** `Q(s, a)`, which predicts the long-term return (reward) of taking action `a` in a given state `s`.

*   **State**: A vectorized representation of the current game situation, including the type of piece at each position on the board, the number of pieces remaining in each player's hand, the current turn number, etc.
*   **Action**: One of all legal "place" or "move" operations.
*   **Reward**: The feedback signal the AI receives from the environment after performing an action.
    *   **Winning**: Receives a large positive reward.
    *   **Losing**: Receives a large negative reward.
    *   **Drawing**: Receives a zero or a small positive/negative reward.
    *   **Reward Shaping**: To guide the AI to learn faster, we designed a series of intermediate rewards, such as:
        *   A positive reward for surrounding the opponent's Queen Bee.
        *   A penalty for having one's own Queen Bee surrounded.
        *   A small positive reward for making a legal move or placement.
        *   A tiny negative reward for each step taken, to encourage the AI to win as quickly as possible.
*   **Training Process**:
    1.  **Sampling**: The AI (or multiple parallel AIs) plays the game through self-play in the environment, collecting a large number of `(state, action, reward, next_state)` experience tuples.
    2.  **Experience Replay**: The collected experiences are stored in an "experience pool."
    3.  **Training**: A small batch of experiences is randomly drawn from the experience pool to train the neural network. The goal of the training is to make the predicted value of `Q(s, a)` as close as possible to the **target Q-value** (usually `reward + discount_factor * max(Q(next_state, all_legal_actions))`).
    4.  **Exploration vs. Exploitation**: The AI uses an **ε-greedy** strategy to select actions. That is, with a probability of ε, it chooses a random legal action (exploration), and with a probability of 1-ε, it chooses the action with the highest Q-value (exploitation). As training progresses, ε gradually decays, causing the AI to shift from random exploration to relying more on the optimal strategies it has learned.

Through tens of thousands of games of self-play and training, the AI's neural network can gradually learn the complex patterns and strategies of the Hive board, thereby achieving a high level of play.

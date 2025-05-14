# Project 3: Reinforcement Learning Agents

## Overview
This assignment explores value-based reinforcement learning methods in Pacman:

- **ValueIterationAgent**: Batch value iteration for Markov Decision Processes.
- **QLearningAgent**: Online Q-learning with ε-greedy exploration.
- **ApproximateQAgent**: Feature-based Q-learning using state features.

## Structure
- `valueIterationAgents.py`: Implements the value iteration algorithm.
- `qlearningAgents.py`: Contains QLearningAgent and ApproximateQAgent.
- `analysis.py`: Discusses parameter choices (learning rate, discount, exploration).

## Usage Examples
```bash
# Run value iteration for 100 iterations on smallGrid
python pacman.py -p ValueIterationAgent -a iterations=100,printValues=True

# Train Q-learning agent for 2000 episodes
python pacman.py -p QLearningAgent -a episodes=2000,alpha=0.1,epsilon=0.2
```

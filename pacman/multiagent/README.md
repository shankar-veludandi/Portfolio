# Project 2: Multi-Agent Pacman

## Overview
In this assignment, you will create agents that play Pacman under adversarial and stochastic environments:

- **ReflexAgent**: A rule-based agent using an evaluation function.
- **MinimaxAgent**: Implements minimax search for perfect-information adversaries.
- **AlphaBetaAgent**: Minimax with alpha-beta pruning to cut branches.
- **ExpectimaxAgent**: Handles chance nodes by computing expected values.

## Structure
- `multiAgents.py`: Implements all agent classes and a custom evaluation function.

## Usage Examples
```bash
# Play with reflex agent on testClassic
python pacman.py -p ReflexAgent -l testClassic

# Minimax depth-2 on minimaxClassic layout
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=2

# Expectimax depth-3 for stochastic ghosts
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

# Project 1: Search

## Overview
This assignment implements fundamental graph search algorithms to navigate Pacman through various maze layouts. You will find implementations for:

- Depth‑First Search (DFS)
- Breadth‑First Search (BFS)
- Uniform Cost Search (UCS)
- A* Search with Manhattan and null heuristics

## Structure
- `search.py`: Contains the core search functions and shared utilities.
- `searchAgents.py`: Configures SearchAgent to apply your search functions to different problems.

## Usage Examples
```bash
# Solve tinyMaze with DFS
python pacman.py -l tinyMaze -p SearchAgent -a fn=depthFirstSearch
```

# Solve mediumMaze with UCS
python pacman.py -l mediumMaze -p SearchAgent -a fn=uniformCostSearch

# Solve bigMaze with A* and the Manhattan heuristic
python pacman.py -l bigMaze -p SearchAgent -a fn=aStarSearch,heuristic=manhattanHeuristic

# CS188 Pacman Projects

A suite of five assignments from UC Berkeley’s CS188, implemented in Python.  
Each folder contains my solution files alongside the original project specs.

## 📁 project1-search  
**Description:** Implemented graph-search algorithms (DFS, BFS, UCS, A*) to guide Pacman through mazes.  
**Key files edited:**  
- `search.py` – depthFirstSearch, breadthFirstSearch, uniformCostSearch, aStarSearch  
- `searchAgents.py` – SearchAgent configurations, problem definitions  

**Usage:**  
```bash
# Run DFS on the tinyMaze layout
python pacman.py -l tinyMaze -p SearchAgent -a fn=depthFirstSearch
# Run A* with Manhattan heuristic on bigMaze
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,h
```

---

## 📁 project2-multiagent  
**Description:** Built multi-agent adversarial search agents (Minimax, Alpha-Beta, Expectimax) and a reflex agent with custom evaluation.  
**Key files edited:**  
- `multiAgents.py` – ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent, evaluation function  

**Usage:**  
```bash
# Play with reflex agent
python pacman.py -p ReflexAgent -l testClassic
# Depth-2 Minimax against ghosts
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=2
# Depth-3 Expectimax for stochastic ghosts
python pacman.py -p ExpectimaxAgent -a depth=3
```

---

## 📁 project3-reinforcement  
**Description:** Implemented Value Iteration and Q-Learning agents; applied them to Gridworld, a Crawler robot, and Pacman.  
**Key files edited:**  
- `valueIterationAgents.py` – batch value iteration, policy/Q-value getters  
- `qlearningAgents.py` – Q-learning update, ε-greedy policy, feature-based approximations  
- `analysis.py` – answers to parameter-tuning questions  
**Usage:**  
```bash
# Value iteration on Gridworld
python gridworld.py -a value -i 100 -k 10
# Run Q-learning agent in Pacman
python pacman.py -p QLearningAgent -a episodes=2000,alpha=0.1,epsilon=0.2
```

---

## 📁 project4-ghostbusters  
**Description:** Designed particle-filter and exact-inference modules to track invisible ghosts via noisy distance readings.  
**Key files edited:**  
- `inference.py` – getObservationProb, observeUpdate, time-elapse updates (exact & approximate)  
- `bustersAgents.py` – agents that use inference modules to chase ghosts  
**Usage:**  
```bash
# Run Ghostbusters with exact inference
python busters.py -p BustersAgent -a inferenceType=Exact
# Run with particle filter
python busters.py -p BustersAgent -a inferenceType=Particle
```
---

## 📁 project5-classification  
**Description:** Built and tuned three classifiers (Naive Bayes, Perceptron, MIRA) on digit and face datasets.  
**Key files edited:**  
- `naiveBayes.py` – smoothing, log-joint probabilities, high-odds feature extraction  
- `perceptron.py` – training loop, weight updates  
- `mira.py` – large-margin updates, hyperparameter tuning  
- `dataClassifier.py` – pipeline integration and performance analysis  
**Usage:**  
```bash
# Naive Bayes with autotuning
python dataClassifier.py -c naiveBayes --autotune -d digits
# Train and test perceptron
python dataClassifier.py -c perceptron -d faces -t 1000 -l 0.01
```
---

### How to browse
Each subfolder contains:
- A `README.md` with assignment overview & commands
- My submitted `.py` files
- (Optionally) sample output or screenshots in `docs/`

Feel free to clone and explore each project’s code, autograder tests, and my write-ups!  

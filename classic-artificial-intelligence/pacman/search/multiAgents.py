# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = -1
        minFoodDistance = -1
        posX = newPos[0]
        posY = newPos[1]
        for food in newFood.asList():
            x = food[0]
            y = food[1]
            foodDistance = ((x - posX) ** 2 + (y - posY) ** 2) ** 0.5 
            if foodDistance < minFoodDistance or minFoodDistance == -1:
                minFoodDistance = foodDistance
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            x = ghostPosition[0] 
            y = ghostPosition[1]
            ghostDistance = ((x - posX) ** 2 + (y - posY) ** 2) ** 0.5 
            if ghostDistance < minGhostDistance or minGhostDistance == -1:
                minGhostDistance = ghostDistance
        return successorGameState.getScore() + minGhostDistance - minFoodDistance

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, gameState, depth, agentIndex):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        minmaxValue = (float("-inf"), None)
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            child = (float("-inf"), None)
            nextAgent = agentIndex + 1
            if nextAgent % gameState.getNumAgents() == 0:
                child = self.minimax(nextState, depth + 1, 0)
            else:
                child = self.minimax(nextState, depth, nextAgent)
            if minmaxValue[0] == float("-inf") or (agentIndex == 0 and child[0] > minmaxValue[0]) or (agentIndex % gameState.getNumAgents() != 0 and child[0] < minmaxValue[0] or minmaxValue == float("-inf")):
                minmaxValue = (child[0], action)
        return minmaxValue

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.minimax(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, depth, agentIndex, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        minmaxValue = (float("-inf"), None)
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            child = (float("-inf"), None)
            nextAgent = agentIndex + 1
            if nextAgent % gameState.getNumAgents() == 0:
                child = self.alphabeta(nextState, depth + 1, 0, alpha, beta)
            else:
                child = self.alphabeta(nextState, depth, nextAgent, alpha, beta)
            if minmaxValue[0] == float("-inf") or (agentIndex == 0 and child[0] > minmaxValue[0]) or (agentIndex % gameState.getNumAgents() != 0 and child[0] < minmaxValue[0] or minmaxValue == float("-inf")):
                minmaxValue = (child[0], action)
                if agentIndex == 0:
                    alpha = max(alpha, child[0])
                    if alpha > beta:
                        return (alpha, action)
                else:
                    beta = min(beta, child[0])
                    if alpha > beta:
                        return (beta, action)
        return minmaxValue

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(gameState, 0, 0, float("-inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, depth, agentIndex):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        minmaxValue = (float("-inf"), None)
        ghostSum = 0
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            child = (float("-inf"), None)
            nextAgent = agentIndex + 1
            if nextAgent % gameState.getNumAgents() == 0:
                child = self.expectimax(nextState, depth + 1, 0)
            else:
                child = self.expectimax(nextState, depth, nextAgent)
            if minmaxValue[0] == float("-inf") or (agentIndex == 0 and child[0] > minmaxValue[0]):
                minmaxValue = (child[0], action)
            if agentIndex % gameState.getNumAgents() != 0:
                ghostSum += child[0]
        if agentIndex % gameState.getNumAgents() != 0:
            minmaxValue = (ghostSum / len(gameState.getLegalActions(agentIndex)), gameState.getLegalActions(agentIndex)[-1])
        return minmaxValue

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    -found euclidean distance from current Pacman position to closest food 
    and the distance from the current Pacman position to the closest ghost
    
    -added the inverse of all of distances to current score, then subtracted
     the distance to the regular ghosts

    -changed the multiplier for the scaredGhosts since there should be
    a higher score for that
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    minGhostDistance = -1
    minFoodDistance = -1
    posX = pos[0]
    posY = pos[1]
    score = currentGameState.getScore()
    for food in food.asList():
        x = food[0]
        y = food[1]
        foodDistance = ((x - posX) ** 2 + (y - posY) ** 2) ** 0.5 
        if foodDistance > 0:
            score += 1 / foodDistance
    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        x = ghostPosition[0] 
        y = ghostPosition[1]
        ghostDistance = ((x - posX) ** 2 + (y - posY) ** 2) ** 0.5 
        if ghost.scaredTimer > 0:
            if ghostDistance > 0:
                score += 5 / ghostDistance
        else:
            if ghostDistance > 0:
                score -= 1 / ghostDistance
    return score
    

# Abbreviation
better = betterEvaluationFunction
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


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        xy = currentGameState.getPacmanPosition()
        foodList = currentGameState.getFood().asList()

        food_dists = [manhattanDistance(xy, food) for food in foodList]
        min_food_dist = min(food_dists)
        closest_food = foodList[food_dists.index(min_food_dist)]

        ghost_dists = [manhattanDistance(xy, ghost) for ghost in ghostPositions]
        min_ghost_dist = min(ghost_dists)
        closest_ghost = ghostPositions[ghost_dists.index(min_ghost_dist)]

        next_state_food = manhattanDistance(newPos, closest_food)  # want to minimize
        next_state_ghost = manhattanDistance(newPos, closest_ghost)  # want to maximize

        if next_state_ghost <= 1:
            return next_state_ghost
        else:
            return -next_state_food + scoreEvaluationFunction(childGameState)


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def mm_max(state, depth, turn):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), ""

            pacman_legal = state.getLegalActions(turn)
            best = float("-inf")
            best_move = ""
            for move in pacman_legal:
                next_state = state.getNextState(0, move)
                cmp_state, next_move = mm_min(next_state, depth, 1)
                if cmp_state > best:
                    best_move = move
                best = max(cmp_state, best)

                # best_move = pacman_legal[pacman_states.index(best_state)]
                # max_optimal = [best, best_move]
            return [best, best_move]

        def mm_min(state, depth, turn):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), ""

            worst = float("inf")
            ghost_legal = state.getLegalActions(turn)
            worst_move = ""
            if turn != state.getNumAgents() - 1:
                for move in ghost_legal:
                    next_state = state.getNextState(turn, move)
                    cmp_state, next_move = mm_min(next_state, depth, turn + 1)
                    if cmp_state < worst:
                        worst_move = move
                    worst = min(cmp_state, worst)
            else:
                for move in ghost_legal:
                    next_state = state.getNextState(turn, move)
                    cmp_state, next_move = mm_max(next_state, depth + 1, 0)
                    if cmp_state < worst:
                        worst_move = move
                    worst = min(cmp_state, worst)

            # worst_move = ghost_legal[ghost_states.index(worst_state)]
            return [worst, worst_move]

        return mm_max(gameState, 0, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")
        return self.ab_max(gameState, 0, 0, alpha, beta)[1]

    def ab_max(self, state, depth, turn, a, b):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), ""
        v = float("-inf")
        best_move = ""
        legal = state.getLegalActions(turn)
        for move in legal:
            next_state = state.getNextState(turn, move)
            cmp_state, next_move = self.ab_min(next_state, depth, 1, a, b)
            if cmp_state > v:
                best_move = move
            v = max(cmp_state, v)
            if v > b:
                return [v, best_move]
            a = max(a, v)
        return [v, best_move]

    def ab_min(self, state, depth, turn, a, b):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), ""
        v = float("inf")
        worst_move = ""
        ghost_legal = state.getLegalActions(turn)
        if turn != state.getNumAgents() - 1:
            for move in ghost_legal:
                next_state = state.getNextState(turn, move)
                cmp_state, next_move = self.ab_min(next_state, depth, turn + 1, a, b)
                if cmp_state < v:
                    worst_move = move
                v = min(cmp_state, v)
                if v < a:
                    return [v, worst_move]
                b = min(b, v)
        else:
            for move in ghost_legal:
                next_state = state.getNextState(turn, move)
                cmp_state, next_move = self.ab_max(next_state, depth + 1, 0, a, b)
                if cmp_state < v:
                    worst_move = move
                v = min(cmp_state, v)
                if v < a:
                    return [v, worst_move]
                b = min(b, v)
        return [v, worst_move]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.exp_max(gameState, 0, 0)[1]

    def avg(self, x):
        return sum(x) / len(x)

    def exp_max(self, state, depth, turn):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), ""
        v = float("-inf")
        best_move = ""
        legal = state.getLegalActions(turn)
        for move in legal:
            next_state = state.getNextState(turn, move)
            cmp_state, next_move = self.exp_min(next_state, depth, 1)
            if cmp_state > v:
                best_move = move
            v = max(cmp_state, v)
        return [v, best_move]

    def exp_min(self, state, depth, turn):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), ""
        next_states = []
        v = float("inf")
        worst_move = ""
        ghost_legal = state.getLegalActions(turn)
        if turn != state.getNumAgents() - 1:
            for move in ghost_legal:
                next_state = state.getNextState(turn, move)
                cmp_state, next_move = self.exp_min(next_state, depth, turn + 1)
                if cmp_state < v:
                    worst_move = move
                next_states.append(cmp_state)
        else:
            for move in ghost_legal:
                next_state = state.getNextState(turn, move)
                cmp_state, next_move = self.exp_max(next_state, depth + 1, 0)
                if cmp_state < v:
                    worst_move = move
                next_states.append(cmp_state)
        return [self.avg(next_states), worst_move]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <I tried using a similar function to ReflexAgent with closest food and ghost values (and remaining
    food), but apparently ghost values don't really affect Pacman's score and we can win just on food alone. I'm not
    sure why this is the case and I think my eval function is definitely suboptimal, but it passes the autograder? >
    """
    foodList = currentGameState.getFood().asList()
    ghostList = [ghost.getPosition() for ghost in currentGameState.getGhostStates()]
    xy = currentGameState.getPacmanPosition()

    remaining_food = len(foodList)
    food_dist = [manhattanDistance(xy, food) for food in foodList]
    # ghost_dist = [manhattanDistance(xy, ghost) for ghost in ghostList]
    try:
        closest_food = min(food_dist)
    except:
        closest_food = 0

    return currentGameState.getScore() - remaining_food - closest_food


# Abbreviation
better = betterEvaluationFunction

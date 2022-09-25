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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        # Successor game state of the current state after performing the action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() # New pacman position
        newFood = successorGameState.getFood() # New positions of food
        newCapsules = successorGameState.getCapsules() # New positions of capsules
        newGhostStates = successorGameState.getGhostStates() # new positions of ghosts
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # new scared times of ghosts

        "*** YOUR CODE HERE ***"

        totalScore = successorGameState.getScore() # Base score of the successor game state

        """
        If food is present adding the reciprocal of manhattan distance of the closest food with a factor of 2
        Reciprocal ensures the score be higher for smaller distances so the pacman moves to closer food
        """
        if len(newFood.asList()) != 0:
            totalScore += 2 / min([manhattanDistance(food, newPos) for food in newFood.asList()])

        """
        If capsule is present adding the reciprocal of manhattan distance of the closest capsule with a factor of 3
        Capsule has a higher preference than food with this slightly higher factor as having capsule will 
        remove the effect of ghosts for some time
        """
        if len(newCapsules) != 0:
            totalScore += 3 / min([manhattanDistance(capsule, newPos) for capsule in newCapsules])


        if len(newGhostStates) != 0:
            closeGhost = float("inf")
            for idx, ghost in enumerate(newGhostStates):
                """
                Considering only the ghosts with scared time 0. Other ghosts can't attack the pacman
                So considering those ghosts is unecessary
                """
                if(newScaredTimes[idx] == 0):
                    closeGhost = min(manhattanDistance(ghost.configuration.getPosition(), newPos), closeGhost)

            """
            Subtracting the reciprocal of closest ghost with a very high factor compared to food.
            Pacman staying alive is highly necessary than eating food.
            """
            if closeGhost != 0:
                # Penalizing very highly when the ghost is just beside
                if closeGhost <= 1:
                    totalScore -= 80 / closeGhost
                # Moderately penalizing when within manhattan distance limit of 3
                elif closeGhost <= 3:
                    totalScore -= 20 / closeGhost
                # Regular penalizing when within manhattan distance limit of 10
                elif closeGhost <= 10:
                    totalScore -= 8 / closeGhost
                # Not penalizing for larger distances as pacman can freely move when ghosts are this far

        # Avoiding the STOP action as much a possible
        if action == Directions.STOP:
            totalScore -= 200

        return totalScore

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        """
        MinmaxTree recursive function
        
        NOTE: the move taken by the inner depth nodes doesn't matter. Only the score is necessary.
        However the move of max agent is needed, so included within the function. For this reason,
        thought the base case returns STOP action it doesn't effect.
        
        Whenever current depth is a multiple of agents count, it is a max agent
        
        Inputs 
            currentGameState - game state the function is processing in current iteration
            currentDepth - the depth of the tree in the current iteration
            totalDepth - maximum possible depth of tree
            numAgents - total number of agents in the current game state
        
        Output
            (score, action) - action to be taken by the pacman, score for this action
        """
        def minmaxTree(currentGameState, currentDepth, totalDepth, numAgents):
            """
            Base case: whenever the state is win, loss or the tree depth reaches maximum, returning
            the evaluation score of the current state with STOP move
            """
            if (currentGameState.isWin() or currentGameState.isLose() or currentDepth == totalDepth):
                return (self.evaluationFunction(currentGameState), Directions.STOP)

            # All possible moves from the current state for the current agent
            legalMoves = currentGameState.getLegalActions(currentDepth % numAgents)

            # current depth is a multiple of number of agents to it is a max agent (pacman)
            # Initializing agent score and agent action values
            if (currentDepth % numAgents == 0):
                agentScore, agentAction = -float("inf"), Directions.STOP # Least score for maximizing
            else:
                agentScore, agentAction = float("inf"), Directions.STOP # Max score for minimizing

            # Iterating over the moves for this agent
            for move in legalMoves:
                # New game state for after making the move
                newGameState = currentGameState.generateSuccessor(currentDepth % numAgents, move)
                # Recursively getting scores of child game states
                score = minmaxTree(newGameState, currentDepth + 1, totalDepth, numAgents)[0]

                if(currentDepth % numAgents == 0):
                    if(score > agentScore): # Storing max score and action for max agent
                        agentScore, agentAction = score, move
                else:
                    if(score < agentScore): # Storing min score and value for min agent
                        agentScore, agentAction = score, move

            # Returning the appropriate score for the current agent and action
            return (agentScore, agentAction)

        # Main call to the recursive function
        scores = minmaxTree(gameState, 0, self.depth * gameState.getNumAgents(), gameState.getNumAgents())
        return scores[1] # Returning the action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """
        The minmax tree function from the minimax agent has been used here.
        Additional parameters of alpha and beta have been taken for pruning.
        """
        def minmaxTree(currentGameState, currentDepth, totalDepth, numAgents, alpha, beta):
            if (currentGameState.isWin() or currentGameState.isLose() or currentDepth == totalDepth):
                return (self.evaluationFunction(currentGameState), Directions.STOP)

            legalMoves = currentGameState.getLegalActions(currentDepth % numAgents)

            if (currentDepth % numAgents == 0):
                agentScore, agentAction = -float("inf"), Directions.STOP
            else:
                agentScore, agentAction = float("inf"), Directions.STOP

            for move in legalMoves:
                newGameState = currentGameState.generateSuccessor(currentDepth % numAgents, move)
                score = minmaxTree(newGameState, currentDepth + 1, totalDepth, numAgents, alpha, beta)[0]

                if(currentDepth % numAgents == 0):
                    if score > agentScore:
                        agentScore, agentAction = score, move
                    if (agentScore > beta):
                        # Returning current score if max score gets larger than beta for max agent
                        return (agentScore, agentAction)
                    # Updating alpha value with max score for max agent
                    alpha = max(alpha, agentScore)

                else:
                    if score < agentScore:
                        agentScore, agentAction = score, move
                    if(agentScore < alpha):
                        # Returning current score if min score gets lower than alpha for min agent
                        return (agentScore, agentAction)
                    # Updating beta value with min score for min agent
                    beta = min(beta, agentScore)

            return (agentScore, agentAction)

        """
        Initial values of alpha and beta of infinity and negative infinity 
        respectively for the root max agent of the tree
        """
        alpha = float("-inf")
        beta = float("inf")
        scores = minmaxTree(gameState, 0, self.depth * gameState.getNumAgents(), gameState.getNumAgents(), alpha, beta)
        return scores[1]

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
        "*** YOUR CODE HERE ***"

        # The minmax tree function from the minimax agent has been used here.
        def minmaxTree(currentGameState, currentDepth, totalDepth, numAgents):
            if (currentGameState.isWin() or currentGameState.isLose() or currentDepth == totalDepth):
                return (self.evaluationFunction(currentGameState), Directions.STOP)

            legalMoves = currentGameState.getLegalActions(currentDepth % numAgents)

            if (currentDepth % numAgents == 0):
                agentScore, agentAction = -float("inf"), Directions.STOP
            else:
                # Instead of storing the least agent score, the sum of scores is being stored for min agent
                totalAgentScore, agentAction = 0, Directions.STOP

            for move in legalMoves:
                newGameState = currentGameState.generateSuccessor(currentDepth % numAgents, move)
                score = minmaxTree(newGameState, currentDepth + 1, totalDepth, numAgents)[0]

                if (currentDepth % numAgents == 0):
                    if (score > agentScore):
                        # Storing the max score for max agent as usual
                        agentScore, agentAction = score, move
                else:
                    # Adding new score to the total score for the min agent
                    totalAgentScore += score

            """
            If agent in current iteration is min agent returning the average score as the score for this iteration. 
            Since the move doesnt matter for greater depths returning STOP action as a placeholder. 
            """
            if currentDepth % numAgents != 0:
                agentScore, agentAction = totalAgentScore / len(legalMoves), Directions.STOP

            return (agentScore, agentAction)

        scores = minmaxTree(gameState, 0, self.depth * gameState.getNumAgents(), gameState.getNumAgents())
        return scores[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


################################################
##############Stephen Blanchard   ##############
##############swb4062             ##############
##############CMPS 420 - Fall 2015##############
##############Project 3           ##############
################################################
#
# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        #a counter to track all Q values
        self.qVals = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #If we've never seen this Q node, return 0.0
        if (state, action) not in self.qVals:
            return 0.0
        #otherwise, we return the Q value of the current Q node
        else:
            return self.qVals[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qValues = []
        #Here, we run through all of the legal actions for the current state and get the Q value for
        #the (state,action) pair and then dump them into a list
        for action in self.getLegalActions(state):
            qValues.append(self.getQValue(state, action))

        #if there are no legal actions, we return 0.0
        if len(qValues) == 0:
            return 0.0
        #otherwise, we return the max of all of the legal actions for the current state
        else:
            return max(qValues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestVal = self.getValue(state)
        #look at each legal action, grab the best value for the current state and assign the corresponding
        #action to bestMove.
        for action in self.getLegalActions(state):
            if self.getQValue(state, action) == bestVal:
                bestMove = action

        #if there are no legal actions, we return none
        if len(bestMove) == 0:
            return None
        else:
            return bestMove

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #as state above, if there are no legal actions, we return None
        if len(legalActions) == 0:
            return None
        #We use util.flipCoin as suggested against the probability epsilon
        #if the number generated by flipCoin is < epsilon:
        if util.flipCoin(self.epsilon):
            #choose a random action from legalActions
            action = random.choice(legalActions)
            #otherwise, we skip randomness and compute the action with our previous function
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        qVals = self.qVals[(state, action)]
        valueFromQ = self.computeValueFromQValues(nextState)
        alpha = self.alpha
        discount = self.discount

        #update the Q values directly based on the current (state, action) pair.  No need to return anything.
        self.qVals[(state, action)] = (( reward + discount * valueFromQ) * alpha) + (qVals * (1 - alpha))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #This is basically directly from the instructions... the formula is as follows:
        qVal = 0
        #look at each feature in our pre-existing feature function
        for feature in self.featExtractor.getFeatures(state, action):
            #update the Q value based on the weight of the current feature * the current feature as the formula shows
            #  Q(state,action) = for n elements in the features list: Sum(feature[i] * (state, action) * weight[i])
            qVal = qVal + self.weights[feature] * self.featExtractor.getFeatures(state, action)[feature]
        #return Q(s,a)
        return qVal

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #The instructions give this away.  It's unbelievable.

        #difference = r + gamma(self.discount in this instance) * max(Q(s', a')) - Q(s, a)
        difference = reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        #weight[i] = weight[i] + (alpha * difference * features[i](s, a)
        for feature in self.featExtractor.getFeatures(state, action):
            self.weights[feature] = self.weights[feature] + (self.alpha * difference * self.featExtractor.getFeatures(state, action)[feature])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

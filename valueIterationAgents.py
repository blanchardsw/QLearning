################################################
##############Stephen Blanchard   ##############
##############swb4062             ##############
##############CMPS 420 - Fall 2015##############
##############Project 3           ##############
################################################

# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        #we run our iteration based on the number of times being passed in as a parameter
        for i in range(iterations):
            #we create a dictionary to store and hash each best value we find for the current state
            tempValues = util.Counter()
            #we grab the current states
            for state in self.mdp.getStates():
                #and as always, set our best value to -inf for comparison
                bestValue = float('-inf')
                #now, looking at the list of actions for the current state that we're iterating over
                for action in mdp.getPossibleActions(state):
                    #we get a value for each transition
                    value = 0
                    #each state has a transition to another state with a given action
                    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                        #we want to sum up the value of each reward and value for the transition and a given discount
                        value = value + transition[1]*(self.mdp.getReward(state, action, transition[0]) + discount*self.values[transition[0]])
                    #if the calculated value is higher than bestValue, then it's a new high
                    if value > bestValue:
                        bestValue = value
                #We want to ensure that we've found a best value before trying to assign -inf as a bestValue for the state
                if bestValue != float('-inf'):
                    tempValues[state] = bestValue
            #now we have a list of all states with their calculated values.
            #We populate the main values list for the agent.
            for state in self.mdp.getStates():
                self.values[state] = tempValues[state]

    def getValue(self, state):
        #Return the value of the state from def_init_
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #The Q-Value is just the sum of each reward plus the values of each transition
        #The discount passed into init is applied to the value as a fraction
        qValue = 0
        #So we look at each transition for the current state and action
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            #and sum up the reward + value with discount applied
            qValue = qValue + transition[1]*(self.mdp.getReward(state, action, transition[0]) + self.discount*self.values[transition[0]])
        #and return the resulting Q-Value
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #Just like in minmax, etc, we start the bestValue out at -inf for comparison
        bestValue = float('-inf')
        #and we'll find the best action (unless there are no legal moves)
        bestAction = ""
        #run through all of the possible actions based on the current state
        for action in self.mdp.getPossibleActions(state):
            #and then use the previously defined function to find the Q-value
            qValue = self.computeQValueFromValues(state, action)
            #keep track of the best value and the corresponding action that belongs to that value
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
        #if we never find a best action (because there was no legal ones)
        if bestAction == "":
            #we return None type
            return None
        else:
            #otherwise, we return what the best action was
            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

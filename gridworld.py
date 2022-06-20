# The design of the grid world experimental setup has taken inspiration from:
# https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/GridWorld/gridWorld.py

import numpy as np
import random

N = 3                  # number of rows and columns
START_STATE = (N-1, 0)  # coordinates of the starting state
GOAL_STATE = (0, N-1)   # coordinates of the goal state

class Grid:

    def __init__(self, state = START_STATE):
        self.size = N
        self.board = np.zeros([N,N])
        self.state = state
        self.goalReached = False

    def goal_reached(self):
        if self.state == GOAL_STATE:
            self.goalReached = True
        return self.goalReached
    
    def get_reward_non_terminal_bernoulli(self):
        if self.state == GOAL_STATE:
            return 5
        else:
            chance = random.randint(0, 100)
            if chance < 50:
                return 10
            else:
                return -12

    def get_reward_bernoulli(self):
        if self.state == GOAL_STATE:
            chance = random.randint(0, 100)
            if chance < 50:
                return 50
            else:
                return -40
        else:
            chance = random.randint(0, 100)
            if chance < 50:
                return 10
            else:
                return -12

    def get_reward_high_variance_gaussian(self):
        if self.state == GOAL_STATE:
            return 5
        else:
            mean = -1
            sd = 5
            reward = np.random.normal(loc = mean, scale = sd)
            return reward

    def get_reward_low_variance_gaussian(self):
        if self.state == GOAL_STATE:
            return 5
        else:
            mean = -1
            sd = 1
            reward = np.random.normal(loc = mean, scale = sd)
            return reward

    def display_board(self):
        for i in range(0, self.size):
            print('-----------')
            out = '| '
            for j in range(0, self.size):
                if (i, j) == GOAL_STATE:
                    out += 'G' + ' | '
                elif (i, j) == START_STATE:
                    out += 'S' + ' | '
                else:
                    out += ' | '
            print(out)
        print('-----------')

    


from cmath import sqrt
import numpy as np
import math
import random
from gridworld import Grid
import matplotlib.pyplot as plt
from scipy.special import softmax
import copy

class Agent:

    def __init__(self):
        self.world = Grid()
        self.Q_values = np.zeros([self.world.size, self.world.size])
        self.Q_values_2 = np.zeros([self.world.size, self.world.size])   # to be used along self.Q_values in case of double q-learning
        self.action_values = np.zeros([self.world.size, self.world.size])
        self.counter_states_visited = np.zeros([self.world.size, self.world.size])
        self.updates_value_func_A = np.zeros([self.world.size, self.world.size])
        self.updates_value_func_B = np.zeros([self.world.size, self.world.size])
        self.algorithm_no = 0
        self.succesful = 0

    def reset(self):
        self.world = Grid()

    def reset_q_values(self):
        self.Q_values = np.zeros([self.world.size, self.world.size])
        self.Q_values_2 = np.zeros([self.world.size, self.world.size])

    def reset_visit_counter(self):
        self.counter_states_visited = np.zeros([self.world.size, self.world.size])

    def reset_update_counter(self):
        self.updates_value_func_A = np.zeros([self.world.size, self.world.size])
        self.updates_value_func_B = np.zeros([self.world.size, self.world.size])        

    # increment n(s) (number of times state s has been visited)

    def increment_update_counter_A(self):
        self.updates_value_func_A[self.world.state[0]][self.world.state[1]] = self.updates_value_func_A[self.world.state[0]][self.world.state[1]] + 1

    def increment_update_counter_B(self):
        self.updates_value_func_B[self.world.state[0]][self.world.state[1]] = self.updates_value_func_B[self.world.state[0]][self.world.state[1]] + 1

    def increment_no_times_state_visited(self):
        self.counter_states_visited[self.world.state[0]][self.world.state[1]] = self.counter_states_visited[self.world.state[0]][self.world.state[1]] + 1

    # get all available actions, including the ones which lead to a state outside of the grid
    
    def get_actions(self):
        actions = []
        actions.append((self.world.state[0] - 1, self.world.state[1]))   #  for action "up"
        actions.append((self.world.state[0] + 1, self.world.state[1]))   #  for action "down"
        actions.append((self.world.state[0], self.world.state[1] + 1))   #  for action "right" 
        actions.append((self.world.state[0], self.world.state[1] - 1))   #  for action "left"
        return actions

    # get only the actions which lead to a state within the grid

    def get_legal_actions(self):
        legal_actions = []
        if self.world.state[0] - 1 >= 0:   # for action "up"
            legal_actions.append((self.world.state[0] - 1, self.world.state[1]))
        if self.world.state[0] + 1 < self.world.size:  # for action "down"
            legal_actions.append((self.world.state[0] + 1, self.world.state[1]))
        if self.world.state[1] + 1 < self.world.size:  # for action "right"
            legal_actions.append((self.world.state[0], self.world.state[1] + 1))
        if self.world.state[1] - 1 >= 0:   # for action "left"
            legal_actions.append((self.world.state[0], self.world.state[1] - 1))
        return legal_actions

    def get_legal_actions_for_state(self,state):
        legal_actions = []
        if state[0] - 1 >= 0:   # for action "up"
            legal_actions.append((state[0] - 1, state[1]))
        if state[0] + 1 < self.world.size:  # for action "down"
            legal_actions.append((state[0] + 1, state[1]))
        if state[1] + 1 < self.world.size:  # for action "right"
            legal_actions.append((state[0], state[1] + 1))
        if state[1] - 1 >= 0:   # for action "left"
            legal_actions.append((state[0], state[1] - 1))
        return legal_actions

    def generate_episode(self):          # generates an episode for monte-carlo exploration
        self.reset()
        actions_in_episode = []
        rewards = []
        while self.world.goalReached == False:
            self.world.goalReached = self.world.goal_reached()
            legal_actions = self.get_legal_actions()
            self.world.state = random.choice(legal_actions)
            #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
            #reward_for_action = self.world.get_reward_bernoulli()
            reward_for_action = self.world.get_reward_high_variance_gaussian()
            #reward_for_action = self.world.get_reward_low_variance_gaussian()
            actions_in_episode.append(self.world.state)
            rewards.append(reward_for_action)
        return actions_in_episode, rewards
        

    def take_action_greedy(self):
        legal_actions = self.get_legal_actions()
        Q_val_list = []
        for x in legal_actions:
            Q_val_list.append(self.Q_values[x[0]][x[1]])
        if len(set(Q_val_list)) == 1:
            selected_action = random.choice(legal_actions)
            return selected_action
        maximum = self.Q_values[legal_actions[0][0]][legal_actions[0][1]]
        max_i = legal_actions[0][0]
        max_j = legal_actions[0][1]
        for x in legal_actions:
            if maximum < self.Q_values[x[0]][x[1]]:
                maximum = self.Q_values[x[0]][x[1]]
                max_i = x[0]
                max_j = x[1]
        selected_action = (max_i, max_j)
        return selected_action

    def take_action_e_greedy(self):
        epsilon = 1
        if self.counter_states_visited[self.world.state[0]][self.world.state[1]] == 0:
            epsilon = 1
        else:
            epsilon  = 1 / sqrt(self.counter_states_visited[self.world.state[0]][self.world.state[1]])
        probability = random.randint(0, 1)
        available_actions = self.get_actions()
        legal_actions = self.get_legal_actions()
        if probability < epsilon.real:   #  exploration
            selected_action = random.choice(available_actions)
            if selected_action in legal_actions:
                return selected_action
            else:
                return self.world.state
        else:    #  exploitation
            maximum = self.Q_values[legal_actions[0][0]][legal_actions[0][1]]
            max_i = legal_actions[0][0] 
            max_j = legal_actions[0][1]
            for x in legal_actions:
                if maximum < self.Q_values[x[0]][x[1]]:
                    maximum = self.Q_values[x[0]][x[1]]
                    max_i = x[0]
                    max_j = x[1]
            selected_action = (max_i, max_j)
            return selected_action

           

    def take_action_softmax(self):
        legal_actions = self.get_legal_actions()
        Q_val_list = []
        for x in legal_actions:
            Q_val_list.append(self.Q_values[x[0]][x[1]])
        softmax_list = softmax(Q_val_list)
        r = random.random()
        max_i = legal_actions[0][0]
        max_j = legal_actions[0][1]
        for i in range(0, len(softmax_list)):
            if r < softmax_list[i]:
                r = softmax_list[i]
                max_i = legal_actions[i][0]
                max_j = legal_actions[i][1]
                break
            r -= softmax_list[i]
        selected_action = (max_i, max_j)
        return selected_action
    
        
    def take_action_double_q_learning_softmax(self):
        sum_matrix = np.zeros([self.world.size, self.world.size])
        for i in range(0, self.world.size):
            for j in range(0, self.world.size):
                sum_matrix[i][j] = self.Q_values[i][j] + self.Q_values_2[i][j]
        Q_val_list = []
        legal_actions = self.get_legal_actions()
        for x in legal_actions:
            Q_val_list.append(sum_matrix[x[0]][x[1]])       
        softmax_list = softmax(Q_val_list)
        r = random.random()
        max_i = legal_actions[0][0]
        max_j = legal_actions[0][1]
        for i in range(0, len(softmax_list)):
            if r < softmax_list[i]:
                r = softmax_list[i]
                max_i = legal_actions[i][0]
                max_j = legal_actions[i][1]
                break
            r -= softmax_list[i]
        selected_action = (max_i, max_j)
        return selected_action
    
    


    def take_action_double_q_learning_greedy(self):
        sum_matrix = np.zeros([self.world.size, self.world.size])
        for i in range(0, self.world.size):
            for j in range(0, self.world.size):
                sum_matrix[i][j] = self.Q_values[i][j] + self.Q_values_2[i][j]
        legal_actions = self.get_legal_actions()
        maximum_Q = sum_matrix[legal_actions[0][0]][legal_actions[0][1]]
        max_i = legal_actions[0][0]
        max_j = legal_actions[0][1]
        for x in legal_actions:
            if maximum_Q < sum_matrix[x[0]][x[1]]:
                maximum_Q = sum_matrix[x[0]][x[1]]
                max_i = x[0]
                max_j = x[1]
        selected_action = (max_i, max_j)
        return selected_action
    
    def take_action_double_q_learning_e_greedy(self):
        epsilon = 1
        if self.counter_states_visited[self.world.state[0]][self.world.state[1]] == 0:
            epsilon = 1
        else:
            epsilon  = 1 / sqrt(self.counter_states_visited[self.world.state[0]][self.world.state[1]])
        sum_matrix = np.zeros([self.world.size, self.world.size])
        for i in range(0, self.world.size):
            for j in range(0, self.world.size):
                sum_matrix[i][j] = self.Q_values[i][j] + self.Q_values_2[i][j]
        probability = random.randint(0, 1)
        available_actions = self.get_actions()
        legal_actions = self.get_legal_actions()
        if probability < epsilon.real:   #  exploration
            selected_action = random.choice(available_actions)
            if selected_action in legal_actions:
                return selected_action
            else:
                return self.world.state
        else:    #  exploitation
            maximum_Q = sum_matrix[legal_actions[0][0]][legal_actions[0][1]]
            max_i = legal_actions[0][0]
            max_j = legal_actions[0][1]
            for x in legal_actions:
                if maximum_Q < sum_matrix[x[0]][x[1]]:
                    maximum_Q = sum_matrix[x[0]][x[1]]
                    max_i = x[0]
                    max_j = x[1]
            selected_action = (max_i, max_j)
            return selected_action

    # update the Q value

    def Q_value_update(self, next_state, reward):
        #alpha = 1 / self.counter_states_visited[self.world.state[0]][self.world.state[1]]     #  linear learning rate
        alpha = 1 / pow(self.counter_states_visited[self.world.state[0]][self.world.state[1]], 0.8)  # polynomial learning rate
        gamma  = 0.95
        coord_list = self.get_legal_actions_for_state(next_state)
        maximum = self.Q_values[coord_list[0][0]][coord_list[0][1]]
        for x in coord_list:
            if maximum < self.Q_values[x[0]][x[1]]:
                maximum = self.Q_values[x[0]][x[1]]
        self.Q_values[self.world.state[0]][self.world.state[1]] = self.Q_values[self.world.state[0]][self.world.state[1]] + alpha * (reward + gamma * maximum - self.Q_values[self.world.state[0]][self.world.state[1]])
        return self.Q_values

    # Q value update for double Q-learning

    

    def Q_value_update_double(self, next_state, reward, update_QA):
        #alpha = 1 / self.counter_states_visited[previous_state[0]][previous_state[1]]     #  linear learning rate
        alpha = 1 / pow(self.counter_states_visited[self.world.state[0]][self.world.state[1]], 0.8)  # polynomial learning rate
        gamma  = 0.95
        coord_list = self.get_legal_actions_for_state(next_state)
        if update_QA == True:
            max_QA = self.Q_values[coord_list[0][0]][coord_list[0][1]]
            max_i = coord_list[0][0]
            max_j = coord_list[0][1]
            for x in coord_list:
                if max_QA < self.Q_values[x[0]][x[1]]:
                    max_QA = self.Q_values[x[0]][x[1]]
                    max_i = x[0]
                    max_j = x[1]
            maximum = self.Q_values_2[max_i][max_j]
            self.Q_values[self.world.state[0]][self.world.state[1]] = self.Q_values[self.world.state[0]][self.world.state[1]] + alpha * (reward + gamma * maximum - self.Q_values[self.world.state[0]][self.world.state[1]])
            return self.Q_values
        else:
            max_QB = self.Q_values_2[coord_list[0][0]][coord_list[0][1]]
            max_i = coord_list[0][0]
            max_j = coord_list[0][1]
            for x in coord_list:
                if max_QB < self.Q_values_2[x[0]][x[1]]:
                    max_QB = self.Q_values_2[x[0]][x[1]]
                    max_i = x[0]
                    max_j = x[1]
            maximum = self.Q_values[max_i][max_j]
            self.Q_values_2[self.world.state[0]][self.world.state[1]] = self.Q_values_2[self.world.state[0]][self.world.state[1]] + alpha * (reward + gamma * maximum - self.Q_values_2[self.world.state[0]][self.world.state[1]])
            return self.Q_values_2


    def Q_value_update_SARSA(self, next_state, reward):
        #alpha = 1 / self.counter_states_visited[self.world.state[0]][self.world.state[1]]     #  linear learning rate
        alpha = 1 / pow(self.counter_states_visited[self.world.state[0]][self.world.state[1]], 0.8)  # polynomial learning rate
        gamma  = 0.95
        self.Q_values[self.world.state[0]][self.world.state[1]] = self.Q_values[self.world.state[0]][self.world.state[1]] + alpha * (reward + gamma * self.Q_values[next_state[0]][next_state[1]] - self.Q_values[self.world.state[0]][self.world.state[1]])
        return self.Q_values

    # The update formula for speedy q-learning was adapted from:
    # https://github.com/indujohniisc/GSQL

    def Q_value_update_SQL(self, previous_state, reward, alpha, Q_previous_estimate, Q_current_estimate):
        gamma = 0.95
        legal_actions_current_state = self.get_legal_actions()
        maximum_q_current_estimate = Q_current_estimate[legal_actions_current_state[0][0]][legal_actions_current_state[0][1]]
        maximum_q_previous_estimate = Q_previous_estimate[legal_actions_current_state[0][0]][legal_actions_current_state[0][1]]
        for x in legal_actions_current_state:
            if maximum_q_previous_estimate < Q_previous_estimate[x[0]][x[1]]:
                maximum_q_previous_estimate = Q_previous_estimate[x[0]][x[1]]
        for x in legal_actions_current_state:
            if maximum_q_current_estimate < Q_current_estimate[x[0]][x[1]]:
                maximum_q_current_estimate = Q_current_estimate[x[0]][x[1]] 
        self.Q_values[previous_state[0]][previous_state[1]] = Q_current_estimate[previous_state[0]][previous_state[1]] + alpha * (reward + gamma * maximum_q_previous_estimate - Q_current_estimate[previous_state[0]][previous_state[1]]) + (1 - alpha) * (reward + gamma * maximum_q_current_estimate - reward - gamma * maximum_q_previous_estimate)
        Q_previous_estimate = copy.deepcopy(Q_current_estimate)
        Q_current_estimate = copy.deepcopy(self.Q_values)
        return self.Q_values, Q_previous_estimate, Q_current_estimate

    def get_action_value_starting_state(self):
        self.algorithm_no = self.algorithm_no + 1
        self.world = Grid()
        gamma = 0.95
        action_value = 1
        action_no = 0
        i = self.world.size - 1
        j = 0
        end = False
        while end == False:
            
            if i == 0 and j == self.world.size - 1:
                #self.succesful = self.succesful + 1
                #print("optimum policy found for ", self.succesful, " algorithms")
                end = True
            if action_no > 5:
                '''
                if self.algorithm_no == 1:
                    print("stuck in loop at Q-learning/e-greedy")
                elif self.algorithm_no == 2:
                    print("stuck in loop at Q-learning/greedy")
                elif self.algorithm_no == 3:
                    print("stuck in loop at Q-learning/softmax")
                elif self.algorithm_no == 7:
                    print("stuck in loop at SARSA/e-greedy")
                elif self.algorithm_no == 8:
                    print("stuck in loop at SARSA/greedy")
                elif self.algorithm_no == 9:
                    print("stuck in loop at SARSA/softmax")
                elif self.algorithm_no == 10:
                    print("stuck in loop at SQL/e-greedy")
                elif self.algorithm_no == 11:
                    print("stuck in loop at SQL/greedy")
                elif self.algorithm_no == 12:
                    print("stuck in loop at SQL/softmax")
                elif self.algorithm_no == 13:
                    print("stuck in loop at Monte-Carlo")
                '''
                #print("Stuck in loop")
                break
            action_no = action_no + 1
            legal_actions = self.get_legal_actions()
            maximum = self.Q_values[legal_actions[0][0]][legal_actions[0][1]]
            max_i = legal_actions[0][0]
            max_j = legal_actions[0][1]
            for x in legal_actions:
                if maximum < self.Q_values[x[0]][x[1]]:
                    maximum = self.Q_values[x[0]][x[1]]
                    max_i = x[0]
                    max_j = x[1]
            i = max_i
            j = max_j
            self.world.state = (i, j)
            reward = 0
            if end == True:
                #print("final reward obtained")
                reward = self.world.get_reward_non_terminal_bernoulli()
            else:
                reward = self.world.get_reward_non_terminal_bernoulli()
            action_value = action_value + pow(gamma, action_no) * reward
        return action_value

    def get_action_value_starting_state_double_q_learning(self):
        self.algorithm_no = self.algorithm_no + 1
        self.world = Grid()
        gamma = 0.95
        sum_matrix = np.zeros([self.world.size, self.world.size])
        for i in range(0, self.world.size):
            for j in range(0, self.world.size):
                sum_matrix[i][j] = self.Q_values[i][j] + self.Q_values_2[i][j]
        action_value = 1
        action_no = 0
        i = self.world.size - 1
        j = 0
        end = False
        while end == False:
            if i == 0 and j == self.world.size - 1:
                #self.succesful = self.succesful + 1
                #print("optimum policy found for ", self.succesful, " algorithms")
                end = True
            if action_no > 5:
                '''
                if self.algorithm_no == 4:
                    print("stuck in loop at Double Q-learning/e-greedy")
                elif self.algorithm_no == 5:
                    print("stuck in loop at Double Q-learning/greedy")
                elif self.algorithm_no == 6:
                    print("stuck in loop at Double Q-learning/softmax")
                '''
                break
            action_no = action_no + 1
            legal_actions = self.get_legal_actions()
            maximum = sum_matrix[legal_actions[0][0]][legal_actions[0][1]]
            max_i = legal_actions[0][0]
            max_j = legal_actions[0][1]
            for x in legal_actions:
                if maximum < sum_matrix[x[0]][x[1]]:
                    maximum = sum_matrix[x[0]][x[1]]
                    max_i = x[0]
                    max_j = x[1]
            i = max_i
            j = max_j
            self.world.state = (i, j)
            reward = 0
            if end == True:
                #print("final reward obtained")
                reward = self.world.get_reward_non_terminal_bernoulli()
            else:
                reward = self.world.get_reward_non_terminal_bernoulli()
            action_value = action_value + pow(gamma, action_no) * reward
        return action_value

    def Q_Learning_Experiment_e_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()        
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                next_state = self.take_action_e_greedy()
                #print("previous state: ", previous_state)
                #print("current state: ", self.world.state)
                if self.world.state != next_state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values = self.Q_value_update(next_state, reward_for_action)
                    self.world.state = next_state
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value

    def Q_Learning_Experiment_softmax(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()        
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                next_state = self.take_action_softmax()
                if self.world.state != next_state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values = self.Q_value_update(next_state, reward_for_action)
                    self.world.state = next_state
            i = i + 1
            self.reset()    #  reset the enviroment
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value    

    def Q_Learning_Experiment_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                next_state = self.take_action_greedy()
                #if self.world.state != next_state:
                #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                #reward_for_action = self.world.get_reward_bernoulli()
                reward_for_action = self.world.get_reward_high_variance_gaussian()
                #reward_for_action = self.world.get_reward_low_variance_gaussian()
                #self.increment_no_times_state_visited()
                reward_for_experiment = reward_for_experiment + reward_for_action
                self.Q_values = self.Q_value_update(next_state, reward_for_action)
                self.world.state = next_state
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value

    def Double_Q_Learning_Experiment_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()          
        i = 0
        max_q_estimate_start = []
        max_q_2_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            max_q_2_estimate_start.append(max(self.Q_values_2[self.world.size - 2][0], self.Q_values_2[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                update_QA = False
                chance = random.randint(0, 100)
                if chance < 50:
                    update_QA = True
                next_state = self.take_action_double_q_learning_greedy()
                if self.world.state != next_state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    if update_QA == True:
                        self.Q_values = self.Q_value_update_double(next_state, reward_for_action, update_QA)
                    else:
                        self.Q_values_2 = self.Q_value_update_double(next_state, reward_for_action, update_QA)
                    self.world.state = next_state
            i = i + 1
            self.reset()

        action_value = self.get_action_value_starting_state_double_q_learning()

        max_q_estimate = []

        for i in range(0, len(max_q_estimate_start)):
            max_q_estimate.append(max_q_estimate_start[i] + max_q_2_estimate_start[i])

        return avg_reward, max_q_estimate, action_value

    def Double_Q_Learning_Experiment_softmax(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()         
        i = 0
        max_q_estimate_start = []
        max_q_2_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            max_q_2_estimate_start.append(max(self.Q_values_2[self.world.size - 2][0], self.Q_values_2[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                update_QA = False
                chance = random.randint(0, 100)
                if chance < 50:
                    update_QA = True
                next_state = self.take_action_double_q_learning_softmax()
                #if self.world.state != next_state:
                #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                #reward_for_action = self.world.get_reward_bernoulli()
                reward_for_action = self.world.get_reward_high_variance_gaussian()
                #reward_for_action = self.world.get_reward_low_variance_gaussian()
                #self.increment_no_times_state_visited()
                reward_for_experiment = reward_for_experiment + reward_for_action
                if update_QA == True:
                    self.Q_values = self.Q_value_update_double(next_state, reward_for_action, update_QA)
                else:
                    self.Q_values_2 = self.Q_value_update_double(next_state, reward_for_action, update_QA)
                self.world.state = next_state
            i = i + 1
            self.reset()

        action_value = self.get_action_value_starting_state_double_q_learning()

        max_q_estimate = []

        for i in range(0, len(max_q_estimate_start)):
            max_q_estimate.append(max_q_estimate_start[i] + max_q_2_estimate_start[i])

        return avg_reward, max_q_estimate, action_value

    def Double_Q_Learning_Experiment_e_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()         
        i = 0
        max_q_estimate_start = []
        max_q_2_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            max_q_2_estimate_start.append(max(self.Q_values_2[self.world.size - 2][0], self.Q_values_2[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                update_QA = False
                chance = random.randint(0, 100)
                if chance < 50:
                    update_QA = True
                next_state = self.take_action_double_q_learning_e_greedy()
                if next_state != self.world.state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    if update_QA == True:
                        self.Q_values = self.Q_value_update_double(next_state, reward_for_action, update_QA)
                    else:
                        self.Q_values_2 = self.Q_value_update_double(next_state, reward_for_action, update_QA)
                    self.world.state = next_state
            i = i + 1
            self.reset()

        action_value = self.get_action_value_starting_state_double_q_learning()

        max_q_estimate = []

        for i in range(0, len(max_q_estimate_start)):
            max_q_estimate.append(max_q_estimate_start[i] + max_q_2_estimate_start[i])

        return avg_reward, max_q_estimate, action_value
        

    def SARSA_experiment_e_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()         
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if (i > 0):
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                next_state = self.take_action_e_greedy()
                if self.world.state != next_state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values = self.Q_value_update_SARSA(next_state, reward_for_action)
                    self.world.state = next_state
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value



    def SARSA_experiment_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()         
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if (i > 0):
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                next_state = self.take_action_greedy()
                if self.world.state != next_state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values = self.Q_value_update_SARSA(next_state, reward_for_action)
                    self.world.state = next_state
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value


    def SARSA_experiment_softmax(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()         
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            #self.increment_no_times_state_visited()
            if (i > 0):
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.increment_no_times_state_visited()
                self.world.goalReached = self.world.goal_reached()
                #previous_state = self.world.state
                next_state = self.take_action_softmax()
                if self.world.state != next_state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    #self.increment_no_times_state_visited()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values = self.Q_value_update_SARSA(next_state, reward_for_action)
                    self.world.state = next_state
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value


    def SQL_Experiment_e_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()        
        Q_previous_estimate = np.zeros([self.world.size, self.world.size])
        Q_current_estimate = np.zeros([self.world.size, self.world.size])
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            alpha = 1 / (1 + i)
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.world.goalReached = self.world.goal_reached()
                previous_state = self.world.state
                self.world.state = self.take_action_e_greedy()
                if previous_state != self.world.state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values, Q_previous_estimate, Q_current_estimate = self.Q_value_update_SQL(previous_state, reward_for_action, alpha, Q_previous_estimate, Q_current_estimate)
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value


    def SQL_Experiment_greedy(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()        
        Q_previous_estimate = np.zeros([self.world.size, self.world.size])
        Q_current_estimate = np.zeros([self.world.size, self.world.size])
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            alpha = 1 / (1 + i)
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.world.goalReached = self.world.goal_reached()
                previous_state = self.world.state
                self.world.state = self.take_action_greedy()
                if previous_state != self.world.state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values, Q_previous_estimate, Q_current_estimate = self.Q_value_update_SQL(previous_state, reward_for_action, alpha, Q_previous_estimate, Q_current_estimate)
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value


    def SQL_Experiment_softmax(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()        
        Q_previous_estimate = np.zeros([self.world.size, self.world.size])
        Q_current_estimate = np.zeros([self.world.size, self.world.size])
        i = 0
        max_q_estimate_start = []
        avg_reward = []     # for plotting
        total_reward = 0
        action_value = 0
        while i < epochs:
            alpha = 1 / (1 + i)
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            if i > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / i)
            reward_for_experiment = 0
            while self.world.goalReached == False:
                self.world.goalReached = self.world.goal_reached()
                previous_state = self.world.state
                self.world.state = self.take_action_softmax()
                if previous_state != self.world.state:
                    #reward_for_action = self.world.get_reward_non_terminal_bernoulli()
                    #reward_for_action = self.world.get_reward_bernoulli()
                    reward_for_action = self.world.get_reward_high_variance_gaussian()
                    #reward_for_action = self.world.get_reward_low_variance_gaussian()
                    reward_for_experiment = reward_for_experiment + reward_for_action
                    self.Q_values, Q_previous_estimate, Q_current_estimate = self.Q_value_update_SQL(previous_state, reward_for_action, alpha, Q_previous_estimate, Q_current_estimate)
            i = i + 1
            self.reset()    #  reset the enviroment
        
        action_value = self.get_action_value_starting_state()

        return avg_reward, max_q_estimate_start, action_value

    
    def monte_carlo_experiment(self, epochs = 10):
        self.reset()
        self.reset_q_values()
        self.reset_visit_counter()        
        episode = 0
        max_q_estimate_start = []
        avg_reward = []
        total_reward = 0
        action_value = 0
        keys = []
        for i in range(0, self.world.size):
            for j in range(0, self.world.size):
                keys.append((i, j))
        returns = {key: [] for key in keys}
        gamma  = 0.95
        while episode < epochs:
            G_state = 0
            max_q_estimate_start.append(max(self.Q_values[self.world.size - 2][0], self.Q_values[self.world.size - 1][1]))
            if episode > 0:
                total_reward = total_reward + reward_for_experiment
                avg_reward.append(total_reward / episode)
            states, rewards = self.generate_episode()
            reward_counter = len(rewards)
            reward_for_experiment = sum(rewards)
            for state in reversed(states):
                G_state = gamma * G_state + rewards[reward_counter - 1]
                reward_counter = reward_counter - 1
                returns[state].append(G_state)
            episode = episode + 1
            for i in range(0, self.world.size):
                for j in range(0, self.world.size):
                    if len(returns[(i, j)]) > 0:
                        self.Q_values[i][j] = sum(returns[(i, j)]) / len(returns[(i, j)])
        action_value = self.get_action_value_starting_state()
        self.reset_q_values()
        self.reset()
        return avg_reward, max_q_estimate_start, action_value







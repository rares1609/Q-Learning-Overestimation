from cProfile import label
from turtle import color
from gridworld import Grid
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
from scipy.stats import sem
import seaborn as sns



def plot_reward_Q_learning(reward_Q_e_greedy, reward_Q_softmax, std_q_e_greedy, std_q_softmax, std_array_q_e, std_array_q_softmax):
    x = [0, 10000]
    #std_e = [std_q_e_greedy, std_q_e_greedy] 
    #std_greedy = [std_q_greedy, std_q_greedy]
    #std_softmax = [std_q_softmax, std_q_softmax]
    x_list = np.arange(len(reward_Q_e_greedy))
    #for i in range(0, 9999):
    #    x_list.append(i)
    plt.figure(figsize=(12,8))
    plt.plot(reward_Q_e_greedy, 'b-', label = "Q-learning / e-greedy reward")
    plt.fill_between(x_list, reward_Q_e_greedy - std_array_q_e, reward_Q_e_greedy + std_array_q_e, color = "b", alpha = 0.2, label = "Q-learning / e-greedy standard deviation")
    #plt.plot(x, std_e, color = "red", linestyle = "dotted", label = "Q-learning/e-greedy standard deviation")
    #plt.plot(reward_Q_greedy, 'r-', label = "Q-learning / greedy reward")
    #plt.fill_between(x_list, reward_Q_greedy - std_array_q_greedy, reward_Q_greedy + std_array_q_greedy, color = "r", alpha = 0.2, label = "Q-learning / greedy standard deviation")
    #plt.plot(x, std_greedy, color = "green", linestyle = "dotted", label = "Q-learning/greedy standard deviation")
    plt.plot(reward_Q_softmax, 'g-', label = "Q-learning / softmax reward")
    plt.fill_between(x_list, reward_Q_softmax - std_array_q_softmax, reward_Q_softmax + std_array_q_softmax, color = "g", alpha = 0.2, label = "Q-learning / softmax standard deviation")
    #plt.plot(x, std_softmax, color = "blue", linestyle = "dotted", label = "Q-learning/softmax standard deviation")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xticks([0, 2500, 5000, 7500, 10000])
    plt.ylim(-15, 5)
    plt.xlabel("Games") 
    plt.ylabel("Mean reward")
    plt.title("Mean reward trend and standard deviation for Q-Learning algorithms after 50 trials (High-Variance Gaussian rewards)")
    plt.show()


def plot_maxq_actionval_Q_learning(x, y_Q_e_greedy, y_Q_softmax, max_val_Q_e_greedy, max_val_Q_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_Q_e_greedy, color = "blue", label = "Q-learning / e-greedy maximum Q value in starting state")
    plt.plot(x, y_Q_e_greedy, color = "blue", linestyle = "dashed", label = "Q-learning / e-greedy action value in starting state")
    #plt.plot(max_val_Q_greedy, color = "red", label = "Q-learning / greedy maximum Q value in starting state")
    #plt.plot(x, y_Q_greedy, color = "red", linestyle = "dashed", label = "Q-learning / greedy action value in starting state")
    plt.plot(max_val_Q_softmax, color = "green", label = "Q-learning / softmax maximum Q value in starting state")
    plt.plot(x, y_Q_softmax, color = "green", linestyle = "dashed", label = "Q-learning / softmax action value in starting state")
    plt.grid()
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state for Q-Learning after 50 trials (High-Variance Gaussian rewards)")
    plt.show()

def run_q_greedy():
    rewards_Q_greedy_final = []
    max_val_Q_greedy_final = []
    actionval_list_Q_e_greedy = []
    reward_Q_greedy, max_val_Q_greedy, action_Q_greedy = agent.Q_Learning_Experiment_greedy(10000)
    pass


def multiple_runs_q_learning():

    rewards_Q_e_greedy_final = []
    #rewards_Q_greedy_final = []
    rewards_Q_softmax_final = []

    max_val_Q_e_greedy_final = []
    #max_val_Q_greedy_final = []
    max_val_Q_softmax_final = []

    actionval_list_Q_e_greedy = []
    #actionval_list_Q_greedy = []
    actionval_list_Q_softmax = []

    for i in range(0, 50):
        reward_Q_e_greedy, max_val_Q_e_greedy, action_Q_e_greedy= agent.Q_Learning_Experiment_e_greedy(10000)
        #reward_Q_greedy, max_val_Q_greedy, action_Q_greedy = agent.Q_Learning_Experiment_greedy(10000) 
        reward_Q_softmax, max_val_Q_softmax, action_Q_softmax = agent.Q_Learning_Experiment_softmax(10000)  

        rewards_Q_e_greedy_final.append(reward_Q_e_greedy)
        #rewards_Q_greedy_final.append(reward_Q_greedy)
        rewards_Q_softmax_final.append(reward_Q_softmax)  

        max_val_Q_e_greedy_final.append(max_val_Q_e_greedy)
        #max_val_Q_greedy_final.append(max_val_Q_greedy)
        max_val_Q_softmax_final.append(max_val_Q_softmax)

        actionval_list_Q_e_greedy.append(action_Q_e_greedy)
        #actionval_list_Q_greedy.append(action_Q_greedy)
        actionval_list_Q_softmax.append(action_Q_softmax)

        print("Run: ", i + 1, " complete")

    
    avg_rewards_Q_e_greedy = np.average(np.array(rewards_Q_e_greedy_final), axis = 0)
    #avg_rewards_Q_greedy = np.average(np.array(rewards_Q_greedy_final), axis = 0) 
    avg_rewards_Q_softmax = np.average(np.array(rewards_Q_softmax_final), axis = 0)

    std_q_e_greedy = np.std(avg_rewards_Q_e_greedy)
    #std_q_greedy = np.std(avg_rewards_Q_greedy)
    std_q_softmax = np.std(avg_rewards_Q_softmax)

    std_array_q_e = np.std(np.array(rewards_Q_e_greedy_final), axis = 0)
    #std_array_q_greedy = np.std(np.array(rewards_Q_greedy_final), axis = 0)
    std_array_q_softmax = np.std(np.array(rewards_Q_softmax_final), axis = 0)

    print(std_q_e_greedy)

    #print(std_q_greedy)

    print(std_q_softmax)

    avg_maxq_Q_e_greedy = np.average(np.array(max_val_Q_e_greedy_final), axis = 0)
    #avg_maxq_Q_greedy = np.average(np.array(max_val_Q_greedy_final), axis = 0)
    avg_maxq_Q_softmax = np.average(np.array(max_val_Q_softmax_final), axis = 0)

    avg_actionval_Q_e_greedy = sum(actionval_list_Q_e_greedy) / len(actionval_list_Q_e_greedy)
    #avg_actionval_Q_greedy = sum(actionval_list_Q_greedy) / len(actionval_list_Q_greedy)
    avg_actionval_Q_softmax = sum(actionval_list_Q_softmax) / len(actionval_list_Q_softmax)

    x = [1, 10000]

    y_Q_e_greedy = [avg_actionval_Q_e_greedy, avg_actionval_Q_e_greedy]
    #y_Q_greedy = [avg_actionval_Q_greedy, avg_actionval_Q_greedy]
    y_Q_softmax = [avg_actionval_Q_softmax, avg_actionval_Q_softmax]

    print("Q-learning with e-greedy exploration average reward: ", sum(avg_rewards_Q_e_greedy) / len(avg_rewards_Q_e_greedy))
    #print("Q-learning with greedy exploration average reward: ", sum(avg_rewards_Q_greedy) / len(avg_rewards_Q_greedy))
    print("Q-learning with softmax exploration average reward: ", sum(avg_rewards_Q_softmax) / len(avg_rewards_Q_softmax))
    print("Standard deviation of Q-learning with e-greedy: ", stats.stdev(avg_rewards_Q_e_greedy))
    #print("Standard deviation of Q-learning with greedy: ", stats.stdev(avg_rewards_Q_greedy))
    print("Standard deviation of Q-learning with softmax: ", stats.stdev(avg_rewards_Q_softmax))
    print("Difference between maximum Q-value and action value in the starting state for Q-learning with e-greedy exploration: ", avg_maxq_Q_e_greedy[len(avg_maxq_Q_e_greedy) - 1] - avg_actionval_Q_e_greedy)
    #print("Difference between maximum Q-value and action value in the starting state for Q-learning with greedy exploration: ", avg_maxq_Q_greedy[len(avg_maxq_Q_greedy) - 1] - avg_actionval_Q_greedy)
    print("Difference between maximum Q-value and action value in the starting state for Q-learning with softmax exploration: ", avg_maxq_Q_softmax[len(avg_maxq_Q_softmax) - 1] - avg_actionval_Q_softmax)


    plot_reward_Q_learning(avg_rewards_Q_e_greedy, avg_rewards_Q_softmax, std_q_e_greedy, std_q_softmax, std_array_q_e, std_array_q_softmax)

    plot_maxq_actionval_Q_learning(x, y_Q_e_greedy, y_Q_softmax, avg_maxq_Q_e_greedy, avg_maxq_Q_softmax)

if __name__ == "__main__":
    board = Grid()
    agent = Agent()   

    multiple_runs_q_learning()
from gridworld import Grid
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats


def plot_reward_SARSA(reward_SARSA_e_greedy,reward_SARSA_greedy, reward_SARSA_softmax, std_SARSA_e_greedy, std_SARSA_softmax, std_array_SARSA_e, std_array_SARSA_greedy ,std_array_SARSA_softmax):   
    x = [0, 10000]
    std_e = [std_SARSA_e_greedy, std_SARSA_e_greedy] 
    #std_greedy = [std_SARSA_greedy, std_SARSA_greedy]
    std_softmax = [std_SARSA_softmax, std_SARSA_softmax]   
    x_list = np.arange(len(reward_SARSA_e_greedy))  
    plt.figure(figsize=(12,8))
    plt.plot(reward_SARSA_e_greedy, 'b-', label = "SARSA / e-greedy reward")
    plt.fill_between(x_list, reward_SARSA_e_greedy - std_array_SARSA_e, reward_SARSA_e_greedy + std_array_SARSA_e, color = "b", alpha = 0.2, label = "SARSA / e-greedy standard deviation")
    #plt.plot(x, std_e, color = "forestgreen", linestyle = "dotted", label = "SARSA/e-greedy standard deviation")
    plt.plot(reward_SARSA_greedy, 'r-', label = "SARSA / greedy reward")
    plt.fill_between(x_list, reward_SARSA_greedy - std_array_SARSA_greedy, reward_SARSA_greedy + std_array_SARSA_greedy, color = "r", alpha = 0.2, label = "SARSA / greedy standard deviation")
    #plt.plot(x, std_greedy, color = "magenta", linestyle = "dotted", label = "SARSA/greedy standard deviation")
    plt.plot(reward_SARSA_softmax, 'g-', label = "SARSA / softmax reward")
    plt.fill_between(x_list, reward_SARSA_softmax - std_array_SARSA_softmax, reward_SARSA_softmax + std_array_SARSA_softmax, color = "g", alpha = 0.2, label = "SARSA / softmax standard deviation")  
    #plt.plot(x, std_softmax, color = "deeppink", linestyle = "dotted", label = "SARSA/softmax standard deviation")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xticks([0, 2500, 5000, 7500, 10000])
    plt.ylim(-15, 5)
    plt.xlabel("Games") 
    plt.ylabel("Mean reward")
    plt.title("Mean reward trend and standard deviation for SARSA algorithms after 50 trials (High-Variance Gaussian rewards)")
    plt.show()

def plot_maxq_actionval_SARSA(x, y_SARSA_e_greedy, y_SARSA_greedy ,y_SARSA_softmax, max_val_SARSA_e_greedy, max_val_SARSA_greedy ,max_val_SARSA_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_SARSA_e_greedy, color = "blue", label = "SARSA / e-greedy maximum Q value in starting state")
    plt.plot(x, y_SARSA_e_greedy, color = "blue", linestyle = "dashed", label = "SARSA / e-greedy action value in starting state")
    plt.plot(max_val_SARSA_greedy, color = "red", label = "SARSA / greedy maximum Q value in starting state")
    plt.plot(x, y_SARSA_greedy, color = "red", linestyle = "dashed", label = "SARSA / greedy action value in starting state")
    plt.plot(max_val_SARSA_softmax, color = "green", label = "SARSA / softmax maximum Q value in starting state")
    plt.plot(x, y_SARSA_softmax, color = "green", linestyle = "dashed", label = "SARSA / softmax action value in starting state")    
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state for SARSA after 50 trials (High-Variance Gaussian rewards)")
    plt.show()  

def multiple_runs_SARSA():
    rewards_SARSA_e_greedy_final = []
    rewards_SARSA_greedy_final = []
    rewards_SARSA_softmax_final = []

    max_val_SARSA_e_greedy_final = []
    max_val_SARSA_greedy_final = []
    max_val_SARSA_softmax_final = []

    actionval_list_SARSA_e_greedy = []
    actionval_list_SARSA_greedy = []
    actionval_list_SARSA_softmax = []

    for i in range(0, 50):
        reward_SARSA_e_greedy, max_val_SARSA_e_greedy, action_value_SARSA_e_greedy = agent.SARSA_experiment_e_greedy(10000)        
        reward_SARSA_greedy, max_val_SARSA_greedy, action_value_SARSA_greedy = agent.SARSA_experiment_greedy(10000)      
        reward_SARSA_softmax, max_val_SARSA_softmax, action_value_SARSA_softmax = agent.SARSA_experiment_softmax(10000)

        rewards_SARSA_e_greedy_final.append(reward_SARSA_e_greedy)
        rewards_SARSA_greedy_final.append(reward_SARSA_greedy)
        rewards_SARSA_softmax_final.append(reward_SARSA_softmax)   

        max_val_SARSA_e_greedy_final.append(max_val_SARSA_e_greedy)
        max_val_SARSA_greedy_final.append(max_val_SARSA_greedy)
        max_val_SARSA_softmax_final.append(max_val_SARSA_softmax)

        actionval_list_SARSA_e_greedy.append(action_value_SARSA_e_greedy)
        actionval_list_SARSA_greedy.append(action_value_SARSA_greedy)
        actionval_list_SARSA_softmax.append(action_value_SARSA_softmax)

        print("Run: ", i + 1, " complete")

    avg_rewards_SARSA_e_greedy = np.average(np.array(rewards_SARSA_e_greedy_final), axis = 0)
    avg_rewards_SARSA_greedy = np.average(np.array(rewards_SARSA_greedy_final), axis = 0)
    avg_rewards_SARSA_softmax = np.average(np.array(rewards_SARSA_softmax_final), axis = 0) 

    std_SARSA_e_greedy = np.std(avg_rewards_SARSA_e_greedy)
    std_SARSA_greedy = np.std(avg_rewards_SARSA_greedy)
    std_SARSA_softmax = np.std(avg_rewards_SARSA_softmax)

    std_array_SARSA_e = np.std(np.array(rewards_SARSA_e_greedy_final), axis = 0)
    std_array_SARSA_greedy = np.std(np.array(rewards_SARSA_greedy_final), axis = 0)
    std_array_SARSA_softmax = np.std(np.array(rewards_SARSA_softmax_final), axis = 0)

    avg_maxq_SARSA_e_greedy = np.average(np.array(max_val_SARSA_e_greedy_final), axis = 0)
    avg_maxq_SARSA_greedy = np.average(np.array(max_val_SARSA_greedy_final), axis = 0)
    avg_maxq_SARSA_softmax = np.average(np.array(max_val_SARSA_softmax_final), axis = 0)  

    avg_actionval_SARSA_e_greedy = sum(actionval_list_SARSA_e_greedy) / len(actionval_list_SARSA_e_greedy)
    avg_actionval_SARSA_greedy = sum(actionval_list_SARSA_greedy) / len(actionval_list_SARSA_greedy)
    avg_actionval_SARSA_softmax = sum(actionval_list_SARSA_softmax) / len(actionval_list_SARSA_softmax)     

    x = [1, 10000]

    y_SARSA_e_greedy = [avg_actionval_SARSA_e_greedy, avg_actionval_SARSA_e_greedy]

    y_SARSA_greedy = [avg_actionval_SARSA_greedy, avg_actionval_SARSA_greedy]

    y_SARSA_softmax = [avg_actionval_SARSA_softmax, avg_actionval_SARSA_softmax]

    print("SARSA with e-greedy exploration average reward: ", sum(avg_rewards_SARSA_e_greedy) / len(avg_rewards_SARSA_e_greedy))
    print("SARSA with greedy exploration average reward: ", sum(avg_rewards_SARSA_greedy) / len(avg_rewards_SARSA_greedy))
    print("SARSA with softmax exploration average reward: ", sum(avg_rewards_SARSA_softmax) / len(avg_rewards_SARSA_softmax))
    print("Standard deviation of SARSA with e-greedy: ", stats.stdev(avg_rewards_SARSA_e_greedy))
    print("Standard deviation of SARSA with greedy: ", stats.stdev(avg_rewards_SARSA_greedy))
    print("Standard deviation of SARSA with softmax: ", stats.stdev(avg_rewards_SARSA_softmax))
    print("Difference between maximum Q-value and action value in the starting state for SARSA with e-greedy exploration: ", avg_maxq_SARSA_e_greedy[len(avg_maxq_SARSA_e_greedy) - 1] - avg_actionval_SARSA_e_greedy)
    print("Difference between maximum Q-value and action value in the starting state for SARSA with greedy exploration: ", avg_maxq_SARSA_greedy[len(avg_maxq_SARSA_greedy) - 1] - avg_actionval_SARSA_greedy)
    print("Difference between maximum Q-value and action value in the starting state for SARSA with softmax exploration: ", avg_maxq_SARSA_softmax[len(avg_maxq_SARSA_softmax) - 1] - avg_actionval_SARSA_softmax)  

    plot_reward_SARSA(avg_rewards_SARSA_e_greedy, avg_rewards_SARSA_greedy ,avg_rewards_SARSA_softmax, std_SARSA_e_greedy, std_SARSA_softmax, std_array_SARSA_e, std_array_SARSA_greedy ,std_array_SARSA_softmax)

    plot_maxq_actionval_SARSA(x, y_SARSA_e_greedy, y_SARSA_greedy ,y_SARSA_softmax, avg_maxq_SARSA_e_greedy, avg_maxq_SARSA_greedy ,avg_maxq_SARSA_softmax)                                   





if __name__ == "__main__":
    board = Grid()
    agent = Agent()  

    multiple_runs_SARSA()    
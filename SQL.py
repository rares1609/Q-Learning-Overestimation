from gridworld import Grid
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats


def plot_reward_SQL(reward_SQL_e_greedy, reward_SQL_greedy ,reward_SQL_softmax, std_SQL_e_greedy, std_SQL_softmax, std_array_SQL_e, std_array_SQL_greedy ,std_array_SQL_softmax):
    x = [0, 10000]
    std_e = [std_SQL_e_greedy, std_SQL_e_greedy] 
    #std_greedy = [std_SQL_greedy, std_SQL_greedy]
    std_softmax = [std_SQL_softmax, std_SQL_softmax]  
    x_list = np.arange(len(reward_SQL_e_greedy))
    plt.figure(figsize=(12,8))
    plt.plot(reward_SQL_e_greedy, 'b-', label = "Speedy Q-learning / e-greedy reward")
    plt.fill_between(x_list, reward_SQL_e_greedy - std_array_SQL_e, reward_SQL_e_greedy + std_array_SQL_e, color = "b", alpha = 0.2, label = "Speedy Q-learning / e-greedy standard deviation")
    #plt.plot(x, std_e, color = "darkgoldenrod", linestyle = "dotted", label = "Speedy Q-learning/e-greedy standard deviation")
    plt.plot(reward_SQL_greedy, 'r-', label = "Speedy Q-learning / greedy reward")
    plt.fill_between(x_list, reward_SQL_greedy - std_array_SQL_greedy, reward_SQL_greedy + std_array_SQL_greedy, color = "r", alpha = 0.2, label = "Speedy Q-learning / greedy standard deviation")
    #plt.plot(x, std_greedy, color = "steelblue", linestyle = "dotted", label = "Speedy Q-learning/greedy standard deviation")
    plt.plot(reward_SQL_softmax, 'g-', label = "Speedy Q-learning / softmax reward")
    plt.fill_between(x_list, reward_SQL_softmax - std_array_SQL_softmax, reward_SQL_softmax + std_array_SQL_softmax, color = "g", alpha = 0.2, label = "Speedy Q-learning / softmax standard deviation")
    #plt.plot(x, std_softmax, color = "sienna", linestyle = "dotted", label = "Speedy Q-learning/softmax standard deviation")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xticks([0, 2500, 5000, 7500, 10000])
    plt.ylim(-15, 5)
    plt.xlabel("Games") 
    plt.ylabel("Mean reward")
    plt.title("Mean reward trend and standard deviation for Speedy Q-Learning algorithms after 50 trials (High-Variance Gaussian rewards)")
    plt.show()

def plot_maxq_actionval_SQL(x, y_SQL_e_greedy, y_SQL_greedy ,y_SQL_softmax, max_val_SQL_e_greedy, max_val_SQL_greedy ,max_val_SQL_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_SQL_e_greedy, color = "blue", label = "Speedy Q-learning / e-greedy maximum Q value in starting state")
    plt.plot(x, y_SQL_e_greedy, color = "blue", linestyle = "dashed", label = "Speedy Q-learning / e-greedy action value in starting state")
    plt.plot(max_val_SQL_greedy, color = "red", label = "Speedy Q-learning / greedy maximum Q value in starting state")
    plt.plot(x, y_SQL_greedy, color = "red", linestyle = "dashed", label = "Speedy Q-learning / greedy action value in starting state")
    plt.plot(max_val_SQL_softmax, color = "green", label = "Speedy Q-learning / softmax maximum Q value in starting state")
    plt.plot(x, y_SQL_softmax, color = "green", linestyle = "dashed", label = "Speedy Q-learning / softmax action value in starting state")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state for Speedy Q-Learning after 50 trials (High-Variance Gaussian rewards)")
    plt.show() 

def multiple_runs_SQL():
    rewards_SQL_e_greedy_final = []
    rewards_SQL_greedy_final = []
    rewards_SQL_softmax_final = []

    max_val_SQL_e_greedy_final = []
    max_val_SQL_greedy_final = []
    max_val_SQL_softmax_final = []

    actionval_list_SQL_e_greedy = []
    actionval_list_SQL_greedy = []
    actionval_list_SQL_softmax = []    

    for i in range(0, 50):

        reward_SQL_e_greedy, max_val_SQL_e_greedy, action_value_SQL_e_greedy = agent.SQL_Experiment_e_greedy(10000)     
        reward_SQL_greedy, max_val_SQL_greedy, action_value_SQL_greedy = agent.SQL_Experiment_greedy(10000)
        reward_SQL_softmax, max_val_SQL_softmax, action_value_SQL_softmax = agent.SQL_Experiment_softmax(10000) 

        rewards_SQL_e_greedy_final.append(reward_SQL_e_greedy)
        rewards_SQL_greedy_final.append(reward_SQL_greedy)
        rewards_SQL_softmax_final.append(reward_SQL_softmax) 

        max_val_SQL_e_greedy_final.append(max_val_SQL_e_greedy)
        max_val_SQL_greedy_final.append(max_val_SQL_greedy)
        max_val_SQL_softmax_final.append(max_val_SQL_softmax)

        actionval_list_SQL_e_greedy.append(action_value_SQL_e_greedy)
        actionval_list_SQL_greedy.append(action_value_SQL_greedy)
        actionval_list_SQL_softmax.append(action_value_SQL_softmax)

        print("Run: ", i + 1, " complete")


    avg_rewards_SQL_e_greedy = np.average(np.array(rewards_SQL_e_greedy_final), axis = 0)
    avg_rewards_SQL_greedy = np.average(np.array(rewards_SQL_greedy_final), axis = 0)
    avg_rewards_SQL_softmax = np.average(np.array(rewards_SQL_softmax_final), axis = 0)

    std_SQL_e_greedy = np.std(avg_rewards_SQL_e_greedy)
    std_SQL_greedy = np.std(avg_rewards_SQL_greedy)
    std_SQL_softmax = np.std(avg_rewards_SQL_softmax)

    std_array_SQL_e = np.std(np.array(rewards_SQL_e_greedy_final), axis = 0)
    std_array_SQL_greedy = np.std(np.array(rewards_SQL_greedy_final), axis = 0)
    std_array_SQL_softmax = np.std(np.array(rewards_SQL_softmax_final), axis = 0)

    avg_maxq_SQL_e_greedy = np.average(np.array(max_val_SQL_e_greedy_final), axis = 0)
    avg_maxq_SQL_greedy = np.average(np.array(max_val_SQL_greedy_final), axis = 0)
    avg_maxq_SQL_softmax = np.average(np.array(max_val_SQL_softmax_final), axis = 0)

    avg_actionval_SQL_e_greedy = sum(actionval_list_SQL_e_greedy) / len(actionval_list_SQL_e_greedy)
    avg_actionval_SQL_greedy = sum(actionval_list_SQL_greedy) / len(actionval_list_SQL_greedy)
    avg_actionval_SQL_softmax = sum(actionval_list_SQL_softmax) / len(actionval_list_SQL_softmax)

    x = [1, 10000]

    y_SQL_e_greedy = [avg_actionval_SQL_e_greedy, avg_actionval_SQL_e_greedy]

    y_SQL_greedy = [avg_actionval_SQL_greedy, avg_actionval_SQL_greedy]

    y_SQL_softmax = [avg_actionval_SQL_softmax, avg_actionval_SQL_softmax]  

    print("Speedy Q-learning with e-greedy exploration average reward: ", sum(avg_rewards_SQL_e_greedy) / len(avg_rewards_SQL_e_greedy))
    print("Speedy Q-learning with greedy exploration average reward: ", sum(avg_rewards_SQL_greedy) / len(avg_rewards_SQL_greedy))
    print("Speedy Q-learning with softmax exploration average reward: ", sum(avg_rewards_SQL_softmax) / len(avg_rewards_SQL_softmax))
    print("Standard deviation of SQL with e-greedy: ", stats.stdev(avg_rewards_SQL_e_greedy))
    print("Standard deviation of SQL with greedy: ", stats.stdev(avg_rewards_SQL_greedy))
    print("Standard deviation of SQL with softmax: ", stats.stdev(avg_rewards_SQL_softmax))
    print("Difference between maximum Q-value and action value in the starting state for Speedy Q-learning with e-greedy exploration: ", avg_maxq_SQL_e_greedy[len(avg_maxq_SQL_e_greedy) - 1] - avg_actionval_SQL_e_greedy)
    print("Difference between maximum Q-value and action value in the starting state for Speedy Q-learning with greedy exploration: ", avg_maxq_SQL_greedy[len(avg_maxq_SQL_greedy) - 1] - avg_actionval_SQL_greedy)
    print("Difference between maximum Q-value and action value in the starting state for Speedy Q-learning with softmax exploration: ", avg_maxq_SQL_softmax[len(avg_maxq_SQL_softmax) - 1] - avg_actionval_SQL_softmax)    
    
    plot_reward_SQL(avg_rewards_SQL_e_greedy, avg_rewards_SQL_greedy ,avg_rewards_SQL_softmax, std_SQL_e_greedy, std_SQL_softmax, std_array_SQL_e, std_array_SQL_greedy ,std_array_SQL_softmax)

    plot_maxq_actionval_SQL(x, y_SQL_e_greedy, y_SQL_greedy , y_SQL_softmax, avg_maxq_SQL_e_greedy, avg_maxq_SQL_greedy ,avg_maxq_SQL_softmax)




if __name__ == "__main__":
    board = Grid()
    agent = Agent()    

    multiple_runs_SQL()    
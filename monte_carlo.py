from gridworld import Grid
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats


def plot_reward_monte_carlo(reward_monte_carlo, std_monte_carlo, std_array_monte_carlo):
    x = [0, 10000]
    y = [std_monte_carlo, std_monte_carlo]
    x_list = np.arange(len(reward_monte_carlo))
    plt.figure(figsize=(12,8))
    plt.plot(reward_monte_carlo, 'b-', label = "Monte-Carlo reward")
    plt.fill_between(x_list, reward_monte_carlo - std_array_monte_carlo, reward_monte_carlo + std_array_monte_carlo, color = "b", alpha = 0.2, label = "Monte-Carlo standard deviation")
    #plt.plot(x, y, color = "gold", linestyle = "dotted", label = "Monte-Carlo standard deviation")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xticks([0, 2500, 5000, 7500, 10000])
    plt.ylim(-20, 5)
    plt.xlabel("Games") 
    plt.ylabel("Mean reward")
    plt.title("Mean reward trend and standard deviation for Monte Carlo after 50 trials (High-Variance Gaussian rewards)")
    plt.show()

def plot_maxq_actionval_monte_carlo(x, y_monte_carlo, max_val_monte_carlo):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_monte_carlo, color = "blue", label = "Monte-Carlo maximum Q value in starting state")
    plt.plot(x, y_monte_carlo, color = "blue", linestyle = "dashed", label = "Monte-Carlo action value in starting state")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state for Monte Carlo after 50 trials (High-Variance Gaussian rewards)")
    plt.show() 


def multiple_runs_monte_carlo():

    rewards_monte_carlo_final = []

    max_val_monte_carlo_final = []

    actionval_list_monte_carlo = []

    for i in range(0, 50):
        reward_monte_carlo, max_val_monte_carlo, action_value_monte_carlo = agent.monte_carlo_experiment(10000)

        rewards_monte_carlo_final.append(reward_monte_carlo)

        max_val_monte_carlo_final.append(max_val_monte_carlo)

        actionval_list_monte_carlo.append(action_value_monte_carlo)

        print("Run: ", i + 1, " complete")

    avg_rewards_monte_carlo = np.average(np.array(rewards_monte_carlo_final), axis = 0)

    std_monte_carlo = np.std(avg_rewards_monte_carlo)

    std_array_monte_carlo = np.std(np.array(rewards_monte_carlo_final), axis = 0)

    avg_maxq_monte_carlo = np.average(np.array(max_val_monte_carlo_final), axis = 0)

    avg_actionval_monte_carlo = sum(actionval_list_monte_carlo) / len(actionval_list_monte_carlo)

    x = [1, 10000]

    y_monte_carlo = [avg_actionval_monte_carlo, avg_actionval_monte_carlo]

    print("Monte-Carlo average reward: ", sum(avg_rewards_monte_carlo) / len(avg_rewards_monte_carlo))
    print("Standard deviation of Monte-Carlo: ", stats.stdev(avg_rewards_monte_carlo))
    print("Difference between maximum Q-value and action value in the starting state for Monte-Carlo: ", avg_maxq_monte_carlo[len(avg_maxq_monte_carlo) - 1] - avg_actionval_monte_carlo)   
   
    plot_reward_monte_carlo(avg_rewards_monte_carlo, std_monte_carlo, std_array_monte_carlo)

    plot_maxq_actionval_monte_carlo(x, y_monte_carlo, avg_maxq_monte_carlo)

    


if __name__ == "__main__":
    board = Grid()
    agent = Agent() 

    multiple_runs_monte_carlo()
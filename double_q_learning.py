from gridworld import Grid
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats


def plot_reward_double_Q_learning(reward_double_Q_e_greedy,reward_double_Q_softmax, std_double_q_e_greedy, std_double_q_softmax, std_array_double_q_e,std_array_double_q_softmax):        
    x = [0, 10000]
    std_e = [std_double_q_e_greedy, std_double_q_e_greedy] 
    #std_greedy = [std_double_q_greedy, std_double_q_greedy]
    std_softmax = [std_double_q_softmax, std_double_q_softmax] 
    x_list = np.arange(len(reward_double_Q_e_greedy))   
    plt.figure(figsize=(12,8))
    plt.plot(reward_double_Q_e_greedy, 'b-', label = "Double Q-learning / e-greedy reward")
    plt.fill_between(x_list, reward_double_Q_e_greedy - std_array_double_q_e, reward_double_Q_e_greedy + std_array_double_q_e, color = "b", alpha = 0.2, label = "Double Q-learning / e-greedy standard deviation")
    #plt.plot(x, std_e, color = "blue", linestyle = "dotted", label = "Double Q-learning/e-greedy standard deviation")
    #plt.plot(reward_double_Q_greedy, 'r-' , label = "Double Q-learning / greedy reward")
    #plt.fill_between(x_list, reward_double_Q_greedy - std_array_double_q_greedy, reward_double_Q_greedy + std_array_double_q_greedy, color = "r", alpha = 0.2, label = "Double Q-learning / greedy standard deviation")
    #plt.plot(x, std_greedy, color = "darkturquoise", linestyle = "dotted", label = "Double Q-learning/greedy standard deviation")
    plt.plot(reward_double_Q_softmax, 'g-' , label = "Double Q-learning / softmax reward")   
    plt.fill_between(x_list, reward_double_Q_softmax - std_array_double_q_softmax, reward_double_Q_softmax + std_array_double_q_softmax, color = "g", alpha = 0.2, label = "Double Q-learning / softmax standard deviation")
    #plt.plot(x, std_softmax, color = "darkorange", linestyle = "dotted", label = "Double Q-learning/softmax standard deviation")
    plt.legend(loc = 'best')
    plt.grid()
    plt.xticks([0, 2500, 5000, 7500, 10000])
    plt.ylim(-15, 5)
    plt.ylabel("Mean reward")
    plt.xlabel("Games") 
    plt.title("Mean reward trend and standard deviation for Double Q-Learning algorithms after 50 trials (High-Variance Gaussian rewards)")
    plt.show()

def plot_maxq_actionval_double_Q_learning(x, y_double_e_greedy, y_double_softmax, max_val_double_Q_e_greedy, max_val_double_Q_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_double_Q_e_greedy, color = "blue", label = "Double Q-learning / e-greedy maximum Q value in starting state")
    plt.plot(x, y_double_e_greedy, color = "blue", linestyle = "dashed", label = "Double Q-learning / e-greedy action value in starting state")
    #plt.plot(max_val_double_Q_greedy, color = "red", label = "Double Q-learning / greedy maximum Q value in starting state")
    #plt.plot(x, y_double_greedy, color = "red", linestyle = "dashed", label = "Double Q-learning / greedy action value in starting state")
    plt.plot(max_val_double_Q_softmax, color = "green", label = "Double Q-learning / softmax maximum Q value in starting state")
    plt.plot(x, y_double_softmax, color = "green", linestyle = "dashed", label = "Double Q-learning / softmax action value in starting state")    
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state for Double Q-Learning after 50 trials (High-Variance Gaussian rewards)")
    plt.show()    

def multiple_runs_double_q_learning():
    rewards_double_Q_e_greedy_final = []
    #rewards_double_Q_greedy_final = []
    rewards_double_Q_softmax_final = []

    max_val_double_Q_e_greedy_final = []
    #max_val_double_Q_greedy_final = []
    max_val_double_Q_softmax_final = []

    actionval_list_double_Q_e_greedy = []
    #actionval_list_double_Q_greedy = []
    actionval_list_double_Q_softmax = []   

    for i in range(0, 50):
        reward_double_Q_e_greedy, max_val_double_Q_e_greedy, action_double_Q_e_greedy = agent.Double_Q_Learning_Experiment_e_greedy(10000)    
        #reward_double_Q_greedy, max_val_double_Q_greedy, action_value_double_Q_greedy = agent.Double_Q_Learning_Experiment_greedy(10000)
        reward_double_Q_softmax, max_val_double_Q_softmax, action_value_double_Q_softmax = agent.Double_Q_Learning_Experiment_softmax(10000)

        rewards_double_Q_e_greedy_final.append(reward_double_Q_e_greedy)
        #rewards_double_Q_greedy_final.append(reward_double_Q_greedy)
        rewards_double_Q_softmax_final.append(reward_double_Q_softmax)

        max_val_double_Q_e_greedy_final.append(max_val_double_Q_e_greedy)
        #max_val_double_Q_greedy_final.append(max_val_double_Q_greedy)
        max_val_double_Q_softmax_final.append(max_val_double_Q_softmax)

        actionval_list_double_Q_e_greedy.append(action_double_Q_e_greedy)
        #actionval_list_double_Q_greedy.append(action_value_double_Q_greedy)
        actionval_list_double_Q_softmax.append(action_value_double_Q_softmax)

        print("Run: ", i + 1, " complete")

    avg_rewards_double_Q_e_greedy = np.average(np.array(rewards_double_Q_e_greedy_final), axis = 0)
    #avg_rewards_double_Q_greedy = np.average(np.array(rewards_double_Q_greedy_final), axis = 0)
    avg_rewards_double_Q_softmax = np.average(np.array(rewards_double_Q_softmax_final), axis = 0) 

    std_double_q_e_greedy = np.std(avg_rewards_double_Q_e_greedy) 
   # std_double_q_greedy = np.std(avg_rewards_double_Q_greedy)
    std_double_q_softmax = np.std(avg_rewards_double_Q_softmax)     

    std_array_double_q_e = np.std(np.array(rewards_double_Q_e_greedy_final), axis = 0)
    #std_array_double_q_greedy = np.std(np.array(rewards_double_Q_greedy_final), axis = 0)
    std_array_double_q_softmax = np.std(np.average(np.array(rewards_double_Q_softmax_final), axis = 0))

    avg_maxq_double_Q_e_greedy = np.average(np.array(max_val_double_Q_e_greedy_final), axis = 0)
    #avg_maxq_double_Q_greedy = np.average(np.array(max_val_double_Q_greedy_final), axis = 0)
    avg_maxq_double_Q_softmax = np.average(np.array(max_val_double_Q_softmax_final), axis = 0)

    avg_actionval_double_Q_e_greedy = sum(actionval_list_double_Q_e_greedy) / len(actionval_list_double_Q_e_greedy)
    #avg_actionval_double_Q_greedy = sum(actionval_list_double_Q_greedy) / len(actionval_list_double_Q_greedy)
    avg_actionval_double_Q_softmax = sum(actionval_list_double_Q_softmax) / len(actionval_list_double_Q_softmax) 

    x = [1, 10000]

    y_double_e_greedy = [avg_actionval_double_Q_e_greedy, avg_actionval_double_Q_e_greedy]

    #y_double_greedy = [avg_actionval_double_Q_greedy, avg_actionval_double_Q_greedy]

    y_double_softmax = [avg_actionval_double_Q_softmax, avg_actionval_double_Q_softmax]   

    print("Double Q-learning with e-greedy exploration average reward: ", sum(avg_rewards_double_Q_e_greedy) / len(avg_rewards_double_Q_e_greedy))
    #print("Double Q-learning with greedy exploration average reward: ", sum(avg_rewards_double_Q_greedy) / len(avg_rewards_double_Q_greedy))
    print("Double Q-learning with softmax exploration average reward: ", sum(avg_rewards_double_Q_softmax) / len(avg_rewards_double_Q_softmax))
    print("Standard deviation of Double Q-learning with e-greedy: ", stats.stdev(avg_rewards_double_Q_e_greedy))
    #print("Standard deviation of Double Q-learning with greedy: ", stats.stdev(avg_rewards_double_Q_greedy))
    print("Standard deviation of Double Q-learning with softmax: ", stats.stdev(avg_rewards_double_Q_softmax))
    print("Difference between maximum Q-value and action value in the starting state for Double Q-learning with e-greedy exploration: ", avg_maxq_double_Q_e_greedy[len(avg_maxq_double_Q_e_greedy) - 1] - avg_actionval_double_Q_e_greedy) 
    #print("Difference between maximum Q-value and action value in the starting state for Double Q-learning with greedy exploration: ", avg_maxq_double_Q_greedy[len(avg_maxq_double_Q_greedy) - 1] - avg_actionval_double_Q_greedy) 
    print("Difference between maximum Q-value and action value in the starting state for Double Q-learning with softmax exploration: ", avg_maxq_double_Q_softmax[len(avg_maxq_double_Q_softmax) - 1] - avg_actionval_double_Q_softmax) 

    plot_reward_double_Q_learning(avg_rewards_double_Q_e_greedy ,avg_rewards_double_Q_softmax, std_double_q_e_greedy, std_double_q_softmax, std_array_double_q_e, std_array_double_q_softmax)  

    plot_maxq_actionval_double_Q_learning(x, y_double_e_greedy ,y_double_softmax, avg_maxq_double_Q_e_greedy,avg_maxq_double_Q_softmax)   

if __name__ == "__main__":
    board = Grid()
    agent = Agent() 

    multiple_runs_double_q_learning()
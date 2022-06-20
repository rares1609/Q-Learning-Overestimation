from gridworld import Grid
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def plot_reward_all(reward_Q_e_greedy, reward_Q_greedy, reward_Q_softmax, reward_double_Q_e_greedy, reward_double_Q_greedy, reward_double_Q_softmax, reward_SARSA_e_greedy, reward_SARSA_greedy, reward_SARSA_softmax, reward_SQL_e_greedy, reward_SQL_greedy, reward_SQL_softmax, reward_monte_carlo):
    plt.figure(figsize=(12,8))
    plt.plot(reward_Q_e_greedy, color = "red", label = "Q-learning/e-greedy")
    plt.plot(reward_Q_greedy, color = "olive", label = "Q-learning/greedy")
    plt.plot(reward_Q_softmax, color = "black", label = "Q-learning/softmax")
    plt.plot(reward_double_Q_e_greedy, color = "blue", label = "Double Q-learning/e-greedy")
    plt.plot(reward_double_Q_greedy, color = "darkturquoise", label = "Double Q-learning/greedy")
    plt.plot(reward_double_Q_softmax, color = "darkorange", label = "Double Q-learning/softmax")
    plt.plot(reward_SARSA_e_greedy, color = "forestgreen", label = "SARSA/e-greedy")
    plt.plot(reward_SARSA_greedy, color = "magenta", label = "SARSA/greedy")
    plt.plot(reward_SARSA_softmax, color = "deeppink", label = "SARSA/softmax")
    plt.plot(reward_SQL_e_greedy, color = "darkgoldenrod", label = "SQL/e-greedy")
    plt.plot(reward_SQL_greedy, color = "steelblue", label = "SQL/greedy")
    plt.plot(reward_SQL_softmax, color = "sienna", label = "SQL/softmax")
    plt.plot(reward_monte_carlo, color = "gold", label = "monte-carlo")
    plt.legend(loc = 'best')
    plt.ylim(-20, 20)
    plt.xlabel("Games") 
    plt.ylabel("Average reward")
    plt.title("Reward trend after 10000 epochs for all algorithms (Low variance Gaussian rewards)")
    plt.show()

def plot_reward_Q_learning(reward_Q_e_greedy, reward_Q_greedy, reward_Q_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(reward_Q_e_greedy, color = "red", label = "Q-learning/e-greedy")
    plt.plot(reward_Q_greedy, color = "olive", label = "Q-learning/greedy")
    plt.plot(reward_Q_softmax, color = "black", label = "Q-learning/softmax")
    plt.legend(loc = 'best')
    plt.ylim(-10, 10)
    plt.xlabel("Games") 
    plt.ylabel("Average reward")
    plt.title("Reward trend after 10000 epochs for Q-Learning algorithms (Low variance Gaussian rewards)")
    plt.show()

def plot_reward_double_Q_learning(reward_double_Q_e_greedy, reward_double_Q_greedy, reward_double_Q_softmax):        
    plt.figure(figsize=(12,8))
    plt.plot(reward_double_Q_e_greedy, color = "blue", label = "Double Q-learning/e-greedy")
    plt.plot(reward_double_Q_greedy, color = "darkturquoise", label = "Double Q-learning/greedy")
    plt.plot(reward_double_Q_softmax, color = "darkorange", label = "Double Q-learning/softmax")   
    plt.legend(loc = 'best')
    plt.ylim(-15, 5)
    plt.xlabel("Games") 
    plt.ylabel("Average reward")
    plt.title("Reward trend after 10000 epochs for Double Q-Learning algorithms (Low variance Gaussian rewards)")
    plt.show()

def plot_reward_SARSA(reward_SARSA_e_greedy, reward_SARSA_greedy, reward_SARSA_softmax):   
    plt.figure(figsize=(12,8))
    plt.plot(reward_SARSA_e_greedy, color = "forestgreen", label = "SARSA/e-greedy")
    plt.plot(reward_SARSA_greedy, color = "magenta", label = "SARSA/greedy")
    plt.plot(reward_SARSA_softmax, color = "deeppink", label = "SARSA/softmax")  
    plt.legend(loc = 'best')
    plt.ylim(-10, 10)
    plt.xlabel("Games") 
    plt.ylabel("Average reward")
    plt.title("Reward trend after 10000 epochs for SARSA algorithms (Low variance Gaussian rewards)")
    plt.show()

def plot_reward_SQL(reward_SQL_e_greedy, reward_SQL_greedy, reward_SQL_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(reward_SQL_e_greedy, color = "darkgoldenrod", label = "SQL/e-greedy")
    plt.plot(reward_SQL_greedy, color = "steelblue", label = "SQL/greedy")
    plt.plot(reward_SQL_softmax, color = "sienna", label = "SQL/softmax")
    plt.legend(loc = 'best')
    plt.ylim(-10, 10)
    plt.xlabel("Games") 
    plt.ylabel("Average reward")
    plt.title("Reward trend after 10000 epochs for Speedy Q-learning algorithms (Low variance Gaussian rewards)")
    plt.show()

def plot_reward_monte_carlo(reward_monte_carlo):
    plt.figure(figsize=(12,8))
    plt.plot(reward_monte_carlo, color = "gold", label = "monte-carlo")
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Average reward")
    plt.title("Reward trend after 10000 epochs for the Monte-Carlo algorithms (Low variance Gaussian rewards)")
    plt.show()

def plot_maxq_actionval_Q_learning(x, y_Q_e_greedy, y_Q_greedy, y_Q_softmax, max_val_Q_e_greedy, max_val_Q_greedy, max_val_Q_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_Q_e_greedy, color = "red", label = "Q-learning/e-greedy")
    plt.plot(x, y_Q_e_greedy, color = "red", linestyle = "dashed")
    plt.plot(max_val_Q_greedy, color = "olive", label = "Q-learning/greedy")
    plt.plot(x, y_Q_greedy, color = "olive", linestyle = "dashed")
    plt.plot(max_val_Q_softmax, color = "black", label = "Q-learning/softmax")
    plt.plot(x, y_Q_softmax, color = "black", linestyle = "dashed")
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state after 10000 epochs for Q-Learning (Low variance Gaussian rewards)")
    plt.show()


def plot_maxq_actionval_double_Q_learning(x, y_double_e_greedy, y_double_greedy, y_double_softmax, max_val_double_Q_e_greedy, max_val_double_Q_greedy, max_val_double_Q_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_double_Q_e_greedy, color = "blue", label = "Double Q-learning/e-greedy")
    plt.plot(x, y_double_e_greedy, color = "blue", linestyle = "dashed")
    plt.plot(max_val_double_Q_greedy, color = "darkturquoise", label = "Double Q-learning/greedy")
    plt.plot(x, y_double_greedy, color = "darkturquoise", linestyle = "dashed")
    plt.plot(max_val_double_Q_softmax, color = "darkorange", label = "Double Q-learning/softmax")
    plt.plot(x, y_double_softmax, color = "darkorange", linestyle = "dashed")    
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state after 10000 epochs for Double Q-Learning (Low variance Gaussian rewards)")
    plt.show()    

def plot_maxq_actionval_SARSA(x, y_SARSA_e_greedy, y_SARSA_greedy, y_SARSA_softmax, max_val_SARSA_e_greedy, max_val_SARSA_greedy, max_val_SARSA_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_SARSA_e_greedy, color = "forestgreen", label = "SARSA/e-greedy")
    plt.plot(x, y_SARSA_e_greedy, color = "forestgreen", linestyle = "dashed")
    plt.plot(max_val_SARSA_greedy, color = "magenta", label = "SARSA/greedy")
    plt.plot(x, y_SARSA_greedy, color = "magenta", linestyle = "dashed")
    plt.plot(max_val_SARSA_softmax, color = "deeppink", label = "SARSA/softmax")
    plt.plot(x, y_SARSA_softmax, color = "deeppink", linestyle = "dashed")    
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state after 10000 epochs for SARSA (Low variance Gaussian rewards)")
    plt.show()    


def plot_maxq_actionval_SQL(x, y_SQL_e_greedy, y_SQL_greedy, y_SQL_softmax, max_val_SQL_e_greedy, max_val_SQL_greedy, max_val_SQL_softmax):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_SQL_e_greedy, color = "darkgoldenrod", label = "SQL/e-greedy")
    plt.plot(x, y_SQL_e_greedy, color = "darkgoldenrod", linestyle = "dashed")
    plt.plot(max_val_SQL_greedy, color = "steelblue", label = "SQL/greedy")
    plt.plot(x, y_SQL_greedy, color = "steelblue", linestyle = "dashed")
    plt.plot(max_val_SQL_softmax, color = "sienna", label = "SQL/softmax")
    plt.plot(x, y_SQL_softmax, color = "sienna", linestyle = "dashed")
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state after 10000 epochs for Speedy Q-Learning (Low variance Gaussian rewards)")
    plt.show()    

def plot_maxq_actionval_monte_carlo(x, y_monte_carlo, max_val_monte_carlo):
    plt.figure(figsize=(12,8))
    plt.plot(max_val_monte_carlo, color = "gold", label = "monte carlo")
    plt.plot(x, y_monte_carlo, color = "gold", linestyle = "dashed")
    plt.legend(loc = 'best')
    plt.xlabel("Games") 
    plt.ylabel("Maximum Q value")
    plt.title("Maximum Q value vs. action value in the starting state after 10000 epochs for Monte-Carlo (Low variance Gaussian rewards)")
    plt.show() 


if __name__ == "__main__":
    board = Grid()
    agent = Agent()    

    '''
     

    reward_double_Q_e_greedy, max_val_double_Q_e_greedy, action_double_Q_e_greedy = agent.Double_Q_Learning_Experiment_e_greedy(10000)

    reward_double_Q_greedy, max_val_double_Q_greedy, action_value_double_Q_greedy = agent.Double_Q_Learning_Experiment_greedy(10000)

    reward_double_Q_softmax, max_val_double_Q_softmax, action_value_double_Q_softmax =agent.Double_Q_Learning_Experiment_softmax(10000)

    print(max_val_double_Q_e_greedy[-1] - action_double_Q_e_greedy)
    print(max_val_double_Q_greedy[-1] - action_value_double_Q_greedy)
    print(max_val_double_Q_softmax[-1] - action_value_double_Q_softmax)

    x = [0, 10000]

    y_double_e_greedy = [action_double_Q_e_greedy, action_double_Q_e_greedy]

    y_double_greedy = [action_value_double_Q_greedy, action_value_double_Q_greedy]

    y_double_softmax = [action_value_double_Q_softmax, action_value_double_Q_softmax]

    plot_reward_double_Q_learning(reward_double_Q_e_greedy, reward_double_Q_greedy, reward_double_Q_softmax)

    plot_maxq_actionval_double_Q_learning(x, y_double_e_greedy, y_double_greedy, y_double_softmax, max_val_double_Q_e_greedy, max_val_double_Q_greedy, max_val_double_Q_softmax)

    
    #for i in range(0, 50):
    #    agent.Q_Learning_Experiment_greedy(10000)
    #    print("Run, ", i + 1, "complete")

    '''

    x = [0, 10000]
    
    reward_e, maxq_e, actionval_e = agent.Q_Learning_Experiment_e_greedy(10000)

    print("finished e-greedy")

    reward_greedy, maxq_greedy, actionval_greedy = agent.Q_Learning_Experiment_greedy(10000)

    print("finished greedy")

    reward_softmax, maxq_softmax, actionval_softmax = agent.Q_Learning_Experiment_softmax(10000)

    print("finished softmax")

    y_Q_e_greedy = [actionval_e, actionval_e]

    y_Q_greedy = [actionval_greedy, actionval_greedy]

    y_Q_softmax = [actionval_softmax, actionval_softmax]

    plot_reward_Q_learning(reward_e, reward_greedy, reward_softmax)

    plot_maxq_actionval_Q_learning(x, y_Q_e_greedy, y_Q_greedy, y_Q_softmax, maxq_e, maxq_greedy, maxq_softmax)



    


    

    

    




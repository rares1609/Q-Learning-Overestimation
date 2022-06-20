# Studying-the-overestimation-bias-of-Q-Learning

This project aims to study the implications that several tabular reinforcement learning algorithm have on the overestimation bias introduced by Q-Learning.

The setting is a 3x3 grid world in which the agent starts in the bottom left corner (starting state) and has to work its way up to the upper right corner (goal state).

The algorithms which were tested are:

- Q-Learning
- Double Q-Learning
- SARSA
- Speedy Q-Learning
- Monte Carlo

The temporal difference algorithms from the previous list are accompanied by 3 different exploration strategies:

- Greedy
- e-greedy
- Softmax

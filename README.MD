# Project Title: Reinforcement Learning for Snake Game

## Project Overview

The goal of this project is to leverage RL techniques to train an intelligent agent that can navigate and thrive in the dynamic environment of the Snake Game. As the agent evolves, it will learn to make strategic decisions to maximize its score by consuming food while avoiding collisions with its own tail and the boundaries of the game.

## Key Components

1. **Environment Setup:**
    - Set up the Snake Game environment using the PyGame library.
    - Define the game rules and reward structure for the RL agent.

   __Reward:__  
   - Eat food: +10  
   - Game over: -10  
   - else: 0  

2. **Reinforcement Learning Algorithm:**
   - Choose a suitable RL algorithm (Deep Q Network) for training the agent.
   - Fine-tune hyperparameters to optimize the learning process.

3. **State Representation:**
    - Define the state space for the agent, considering factors such as the position of the food, the agent's position, and the agent's direction.
    - Experiment with different state representations to improve the agent's performance.

 
   - [danger straight, danger right, danger left,  
 direction left, direction right, direction up, direction down,
 food left, food right, food up]

4. **Action Space and Policies:**
   - Define the action space for the agent, considering possible moves such as turning left, right, or advancing.

   -  [1,0,0] -> Straight  
   -   [0,1,0] -> Right  
   -   [0,0,1] -> Left  

5. **Training and Evaluation**

6. **Visualization**


## Notes
   - If the weights are loaded and the model is already good, reduce epsilon to better performance(reducing exploration)
## Learning Outcomes

- Gain hands-on experience with RL algorithms and their applications in game environments.
- Understand the challenges of defining suitable state representations and reward structures.
- Explore the impact of hyperparameter tuning on the performance of RL agents.
- Analyze the learning process through visualization tools and performance metrics.

## Prerequisites

- Basic understanding of Python programming.
- Familiarity with fundamental concepts of Reinforcement Learning.









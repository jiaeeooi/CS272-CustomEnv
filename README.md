# CS272-FinalProject
Final Project - CS 272, Reinforcement Learning

File explanation

## Specify file name at the top of each file that requires it

## custom_roundabout.py: 
Our custom environment

## train_custom_agent.py: 
Trains an agent on the custom environment. Can set the amount of timesteps.
- Currently set to use PPO
- Saves to a specified file name (Make sure to check before running)

## continue_train_custom_agent.py: 
Takes an existing trained data set, and continues training it with more timesteps
- Loads and saves to specified file names (Make sure to check before running)

## our_custom_agent.py: 
Visually runs a trained agent against the environment over 20 episodes. Loads a trained model from a given file name
- Loads a model from a specified file name

## random_agent.py: 
Runs a random choice agent against the environemnt over 20 episodes.

## roundabout.py: 
Runs a random choice agent against the original base roundabout env given in highway-env

## test_custom_roundabout.py: 
Evaluates a trained agent/model and compares it to a random choice agent.
- Loads a model from a specified file name

## visualize_steps.py: 
Visually runs a trained agent agains the environment. This file specifally allows the user to advance from step to step, instead of playing the whole episode at once
- Loads a model from a specified file name
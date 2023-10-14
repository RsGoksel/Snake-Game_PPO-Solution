# Snake-Game_PPO-Solution
Snake game environment integrated with OpenAI Gym. Proximal Policy Optimization (PPO) implementation for training. Visualization of training progress and agent performance. Easy to understand code.

# What is PPO?
  PPO stands for Proximal Policy Optimization. PPO uses a "proximal" objective function to ensure stable policy updates. 
  It uses a clipping mechanism to prevent large policy updates, which makes training more stable.
  
![image](https://github.com/RsGoksel/Snake-Game_PPO-Solution/assets/80707238/732e1a20-d920-48f9-b6cd-a045cf1c00e8)

# Problem
The classic snake game was discussed for the PPO solution example as problem. 
* _**Reward:** Reward was the distance between the apple and agent. The closer get to the apple, the more rewards get._
* _**Observation Space:** Observation array was "Snake_X, Snake_Y, Apple_X, Apple_Y, Euclid_Distance_Apple_to_Snake"._
* _**Action Space:** Action space was discrete. Because the screen of game was tiled, and we have 4 directions for move."._

# Usage

```
$ pip install -r requirements.txt
```
* test.py   :  Testing for the environment. You can display how game screen is.
* train.py  :  Trains the Agent for learning the game. 
* Agent.py  :  Snake agent's class.
* env_PPO.py:  Main game source code. You can alter the game if you want add some hardness to game.
* Snake_PPO.ipynb :  Comprehensive file that involves whole code and attributes.

For testing,
```
$ python test.py
```

For training,
```
$ python train.py
```

Also, you can alter the constant variables from Constants.py file Especially PPO hyperparameters.


# AgentPlay
![Agent](https://github.com/RsGoksel/Snake-Game_PPO-Solution/assets/80707238/429fb406-f7fb-45a6-bf85-f9bea9606df2)







 

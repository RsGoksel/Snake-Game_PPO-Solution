from env_PPO import SnakeEnv
import Constants
import pygame

HEIGHT = Constants.HEIGHT
WIDTH = Constants.WIDTH

 
env = SnakeEnv()

episode = 10
level = 0

while level <= episode and env.run:
    state = env.reset()
    env.done = False
    score = 0
    
    level += 1
    while not env.done and env.run:
    
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
        
    print('Episode: {} Score:{}'.format(level, score))

pygame.display.quit()
pygame.quit()
print("Game Over")
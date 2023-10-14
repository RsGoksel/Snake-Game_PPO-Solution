import gym
from gym import spaces

import pygame
import random
import numpy as np

import traceback

from Agent import Snake
import Constants


WIDTH, HEIGHT = Constants.WIDTH, Constants.HEIGHT
get_random_apple = lambda: [random.randrange(1,int(WIDTH/15))*15,random.randrange(1,int(HEIGHT/15))*15]


class SnakeEnv(gym.Env):
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.done = False #For episode 
        self.run = True # For game window
        
        self.action_space = spaces.Discrete(4) #For left right up and down
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=float) 
        """
        returns;
            Snake_X, 
            Snake_Y, 
            Apple_X, 
            Apple_Y, 
            Euclid_Distance_Apple_to_Snake
        """
        
        self.info = {}
        
        self.Agent = Snake()
        self.Apple = get_random_apple()
        
        #Euclidian distance between snake's head and the Apple
        self.distance = np.linalg.norm(np.array([self.Agent.head[0], self.Agent.head[1]]) - np.array([self.Apple[0], self.Apple[1]]))
    
    
    def step(self, action):
    
        self.render()
        
        reward = 0
        
        if action == 0: # left
            self.Agent.head[0] -= 15
    
        if action == 1: # Up
            self.Agent.head[1] -= 15
    
        if action == 2: # Right
            self.Agent.head[0] += 15
    
        if action == 3: # Down
            self.Agent.head[1] += 15
    
    
        # Reward if current distance between apple and snake, is lover than old distance. 
        # Briefly, rewarding if the snake over close
        if np.linalg.norm(np.array([self.Agent.head[0], self.Agent.head[1]]) - np.array([self.Apple[0], self.Apple[1]])) < self.distance:
            reward += 15
        
        else:
            reward -= 5
        
        # Refreshing the distance
        self.distance = np.linalg.norm(np.array([self.Agent.head[0], self.Agent.head[1]]) - np.array([self.Apple[0], self.Apple[1]]))
        
        # Getting observation
        self.observation = np.array([self.Agent.head[0], self.Agent.head[1], self.Apple[0], self.Apple[1], self.distance])
        
        # Controlling if agent in game arena, otherwise finish the round
        if self.Agent.head[1] < 0 or self.Agent.head[0] < 0 or self.Agent.head[1] > HEIGHT or self.Agent.head[0] > WIDTH:
            self.done = True
       
        # Move the apple if the snake eats the apple.
        if self.Apple == self.Agent.head:
            
            self.Agent.tail.insert(0,list(self.Agent.head))
            self.Apple = get_random_apple()
            self.distance = np.linalg.norm(np.array([self.Agent.head[0], self.Agent.head[1]]) - np.array([self.Apple[0], self.Apple[1]]))
                
        
        return self.observation, reward, self.done, self.info
        
    
    def reset(self):
    
        self.Agent = Snake()
        self.Apple = get_random_apple()
        
        self.distance = np.linalg.norm(np.array([self.Agent.head[0], self.Agent.head[1]]) - np.array([self.Apple[0], self.Apple[1]]))
    
        self.observation = np.array([self.Agent.head[0], self.Agent.head[1], self.Apple[0], self.Apple[1], self.distance])
        self.done = False
        
        return self.observation
    
    
    def render(self):
        
        self.screen.fill((0,0,0))
        pygame.draw.rect(self.screen, (255,0,0), [self.Apple[0],self.Apple[1], 15,15])

        for pos in self.Agent.tail:
            pygame.draw.rect(self.screen, (0,0,120), [pos[0], pos[1], 15,15])

        self.Agent.tail.insert(0,list(self.Agent.head))
        self.Agent.tail.pop()

        blockSize = 15 
        for x in range(0, 1200, blockSize):
            for y in range(75, 800, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(self.screen, (25,25,25), rect, 1)
        
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
                self.close()
                
                
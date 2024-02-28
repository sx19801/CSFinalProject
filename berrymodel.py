from mesa import Model
from agent import PreyAgent, Obstacle, BerryAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from server import NUMBER_OF_CELLS

import numpy as np
import random



env_rows = NUMBER_OF_CELLS    #state space
env_columns = NUMBER_OF_CELLS #actions
num_actions = 4
            

class BerryCollectionModel(Model):

    def __init__(self, number_of_prey, number_of_berries, width, height):
        self.number_of_prey = number_of_prey
        self.number_of_berries = number_of_berries
        self.grid = MultiGrid(width, height, False, False)
        self.schedule = RandomActivation(self)
        
        self.running = True
        
        self.rl_env = RLEnvironment(self)
        
        # Create agents
        for i in range(self.number_of_prey):
            a = PreyAgent(i, self)
            self.schedule.add(a)
            print(f"agent number {a.unique_id}")
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

            
        #placing berries
        for i in range(self.number_of_berries):
            a = BerryAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x,y = self.get_unique_rand_location()
            print(f"berry x,y {x,y}")
            self.rl_env.berry_reward(x,y)
            self.grid.place_agent(a, (x, y))
            
    def get_unique_rand_location(self):
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        while not self.grid.is_cell_empty((x,y)):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
        return x,y
        

    def create_obstacles(self):
        obstacles = [(5,1),(5,2),(5,4),(5,5),(5,6),(5,7),(5,9),(5,10),(5,11)]
        
        for obstacle in obstacles:
            x,y = obstacle
            self.grid.place_agent(Obstacle(), (x,y))
    
    def step(self):
        self.schedule.step()  
        self.rl_env.get_state_agent()
        self.rl_env.berry_reward_location_check()
        #do rl shit here
        

class RLEnvironment:
    def __init__(self, model):
        self.model = model
        
        #state and action space
        #action codes: up = 0, right = 1, down = 2, left = 3
        self.action_space = ['up', 'right', 'down', 'left']
        
        #initialise state
        self.state = self.get_state_agent()
    
        #rows then columns
        #initialise all values to 0 to begin
        self.Q = np.zeros([env_rows, env_columns, num_actions])
        
        #setting all locations to have reward -0.1
        self.rewards = np.full((env_rows,env_columns), -0.1)
        
    def berry_reward(self, x, y):
        #setting the berry location to have reward +10
        self.rewards[y,x] = 10
        print(f"{x} {y}")
        
    
    def berry_reward_location_check(self):
        for i in range(len(self.rewards)):
            print(f"{self.rewards[len(self.rewards)-1-i]}")
            
    
    def get_state_agent(self):
        #state is just position for our example
        walker = self.model.schedule.agents[0]         
        #print(f"the walker agents position is {walker.pos}")
        #return walker.pos
        pass
    
    def epsilon_greedy_policy(Qtable, state, epsilon):
        rand_int = np.random.uniform(0,1)
        if rand_int > epsilon:
            action = np.argmax(Qtable[state])
        else:
            
            action =  
            
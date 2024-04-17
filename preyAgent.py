import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import math
import pygame as pg
import time 

from gymnasium.envs.registration import register

#GLOBAL VARIABLES
BUFFER_SIZE = 10_000
STEPS_DONE = 0
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) #named tuple for the ease of using the torch methods
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(device)
class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNModel, self).__init__() # This line ensures the nn.Module class' init func is called 
        self.n_observations = n_observations
        self.n_actions = n_actions
        #print(f"n_observations is: {n_observations} adn type is: {type(n_observations)}")
        #print(f"n_actions is: {n_actions} and type is: {type(n_actions)}")
        self.layer1 = nn.Linear(n_observations, 128, device=device)
        self.layer2 = nn.Linear(128,128,device=device)
        self.layer3 = nn.Linear(128, n_actions,device=device)
        
    # def forward1(self,x):
    #     print(f"in forward of model: {device}")
    #     return self.network(x)
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNPreyAgent:
    def __init__(self, model, action_space):
        self.policy_net = model
        self.target_net = model
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #print(f"the target-net with state dict loaded: {self.target_net.load_state_dict(self.policy_net.state_dict())}")
        #print(f"policy net: {self.policy_net}")
        #print(f"target net: {self.target_net}")
        
        #time.sleep(5)
        self.action_space = action_space
        
        self.replay_buffer = MemoryReplay(BUFFER_SIZE)
        
    #HEADS UP!!! the forward method in the model gets called implicitly if you pass the observations parameter when calling the model(state)
    # def get_actions(self, observations):
    #     q_vals = self.model(observations)
        
    #     return q_vals.max(-1) # returns the last element in the array
    
    def select_action(self, state):
        global STEPS_DONE
        sample = random.random()
        
        #epsilon greedy algo
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * STEPS_DONE / EPS_DECAY)
        STEPS_DONE += 1
        if sample > eps_threshold:
            with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                # print(f"type of state: {type(state)}")
                # print(f"policy net with state returns: {self.policy_net(state)}")
                # print(f"policy net with state with max(1) returns: {self.policy_net(state).max(1)}")
                # print(f"policy net with state with max(1) returns: {self.policy_net(state).max(1).indices}")
                
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            #print(f"the else statement: {[[self.action_space.sample()]]} and its type: {type([[self.action_space.sample()]])}")
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)
        
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return                                  #if the length of the buffer isnt >= to the batch size then do nothing
        transitions = self.replay_buffer.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        #print(f"the batch: {batch}")
        
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()
        

# class DQNPredAgent:
#     def _init_(self, model, action_space):
#         #self.policy_net = model
#         #self.target_net = model
#         #self.optimiser = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
#         #self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.action_space = action_space
        
#         self.replay_buffer = MemoryReplay(BUFFER_SIZE)
        
#     def move(self, state):
        
        
#Memory replay buffer used to sample from 
class MemoryReplay(object):
    def __init__(self, BUFFER_SIZE):
        #double ended queue for the memory replay buffer
        self.memory_replay_buffer = deque([], maxlen=BUFFER_SIZE)
        
    def insert(self, *args): # * means it accepts variable length arguments
        self.memory_replay_buffer.append(Transition(*args))
    
    def sample(self,num_samples):
        #check if number of samples is less than size of buffer
        assert num_samples <= len(self.memory_replay_buffer)
        return random.sample(self.memory_replay_buffer, num_samples)  
    
    def __len__(self):
        return len(self.memory_replay_buffer)  
        
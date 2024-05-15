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
BUFFER_SIZE = 1000
STEPS_DONE = 0
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 25000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) #named tuple for the ease of using the torch methods
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class DQNConvModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        #print(f"input shape: {input_shape}")
        #print(f"input shape: {input_shape[1]}")
        #print(f"n_actions: {n_actions}")
        
        super(DQNConvModel, self).__init__()
        # Define convolutional layers
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, device=device)


        print(f"weights of conv1: {self.conv1.weight.data}")
        time.sleep(10)
        dummy_input = torch.zeros(1, *input_shape, device=self.conv1.weight.device)
        # Compute the size of the output from the last Conv layer to properly connect to the Linear layer
        #print(f"input_shape: {input_shape}")
        dummy_output = self._forward_convs(dummy_input)
        self.fc_input_dim = dummy_output.view(-1).shape[0]
        #print(f"self.fc_input_dim: {self.fc_input_dim}")
        #time.sleep(5)
        # Define linear layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512, device=device)
        self.out = nn.Linear(512, n_actions, device=device)

        self._initialize_weights()

    def _forward_convs(self, x):
        #print(f"shape: {shape}"), x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(f"torch.numel(o): {int(torch.numel(o))}")
        return x
    
    def forward(self, x):
        #print("WE IN IT RBO 1")
        #x = x.unsqueeze(0)  #adds another dimension
        #print(f"x is : {x} and has size: {x.size()}")
        x = self._forward_convs(x)

        #print(f"Output of conv layers shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the output for the Linear layer
        #print(f"Shape after flattening: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.out(x)
        #print(f"the output or QVALS for each action: {x}")
        #time.sleep(3)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class DQNPreyAgent:
    def __init__(self, model, action_space, index, alive):
        self.index = index
        self.alive = alive
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
        #print(f"eps threshold: {eps_threshold}")
        if sample > eps_threshold:
        #if sample < eps_threshold:

            with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                # print(f"type of state: {type(state)}")
                # print(f"policy net with state returns: {self.policy_net(state)}")
                # print(f"policy net with state with max(1) returns: {self.policy_net(state).max(1)}")
                # print(f"policy net with state with max(1) returns: {self.policy_net(state).max(1).indices}")
                #print("we in the thing")
                #print(state.size())
                state = state.unsqueeze(0)
                #print(state.size())
                #print(f"state in select action: {state}")

                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            #print(state)
            #print(f"the else statement: {[[self.action_space.sample()]]} and its type: {type([[self.action_space.sample()]])}")
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)
        
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return                                  #if the length of the buffer isnt >= to the batch size then do nothing
        transitions = self.replay_buffer.sample(BATCH_SIZE)
        #print("-------------------------------")
        #print(f"transitions of hell: {transitions}")
        #print("-------------------------------")
        batch = Transition(*zip(*transitions))
        #print(f"the batch: {batch}")
        #print(f"next state batch: {batch.next_state}")
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        
        #print(f"non final mask: {non_final_mask}")

        non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
        for s in batch.next_state:
            if s is None:
                pass
                #print("yo there a next state with none in the batch")
        
        #print(f"final next state in the batch: {final_next_states}")
        #print(f"state batch: {batch.state}")
        #print(f"non final nexxt states: {non_final_next_states}")
        
        #print(f"SIZE OF BATCH STATE: {batch.state[0].size()}")
        
        state_batch = torch.stack(batch.state, axis=0)
        #print(f"state batch after stack: {state_batch}")
        #print(f"state batch after stack [0]: {state_batch[0]}")
        #print(f"SIZE OF BATCH STATE: {state_batch.size()}")
        action_batch = torch.cat(batch.action)
        #print(f"action batch: {action_batch}")
        reward_batch = torch.cat(batch.reward)
        #print(f"reward batch: {reward_batch}")

        #print(f"non final nexxt states: {non_final_next_states.size()}")

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        #print("-----------------------TRAIN-----------------------")
        #print(f"next_state_values: {next_state_values}")
        #print("---------------------END TRAIN---------------------")
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
        #print(Transition(*args))
        #time.sleep(0.2)
    
    def sample(self,num_samples):
        #check if number of samples is less than size of buffer
        assert num_samples <= len(self.memory_replay_buffer)
        return random.sample(self.memory_replay_buffer, num_samples)  
    
    def __len__(self):
        return len(self.memory_replay_buffer)  
        
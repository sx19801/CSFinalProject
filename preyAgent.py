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
BUFFER_SIZE = 10000
STEPS_DONE = 0
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 35000
TAU = 0.005
LR = 1e-3
BETASTART = 0.4
BETAFRAMES = 40000
BETADECAY = 40000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) #named tuple for the ease of using the torch methods
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 34
#set_seed(SEED)


class DQNConvModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNConvModel, self).__init__()
        # Define convolutional layers
        self.set_seed(SEED)
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, device=device)


        dummy_input = torch.zeros(1, *input_shape, device=self.conv1.weight.device)
        # Compute the size of the output from the last Conv layer to properly connect to the Linear layer
        
        dummy_output = self._forward_convs(dummy_input)
        self.fc_input_dim = dummy_output.view(-1).shape[0]
        
        # Define linear layers
        self.fc1 = nn.Linear(self.fc_input_dim, 512, device=device)
        self.out = nn.Linear(512, n_actions, device=device)

        self._initialize_weights()
        #print(f"weights of conv1: {self.conv1.weight.data}")
        #time.sleep(10)

    def _forward_convs(self, x):
        #print(f"shape: {shape}"), x
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        #print(f"torch.numel(o): {int(torch.numel(o))}")
        return x
    
    def forward(self, x):
        x = self._forward_convs(x)


        x = x.view(x.size(0), -1)  # Flatten the output for the Linear layer
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.out(x)
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

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DQNPreyAgent:
    def __init__(self, model, action_space, index, alive):
        self.index = index
        self.alive = alive
        self.policy_net = model
        self.target_net = model
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.action_space = action_space
        self.epsilon_threshold = 0
        self.beta = 0
        self.replay_buffer = MemoryReplay(BUFFER_SIZE)
    
        
    #HEADS UP!!! the forward method in the model gets called implicitly if you pass the observations parameter when calling the model(state)
    
    def select_action(self, state):
        global STEPS_DONE
        sample = random.random()
        
        #epsilon greedy algo
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * STEPS_DONE / EPS_DECAY)
        #eps_threshold = 10
        STEPS_DONE += 1
        #print(f"eps threshold: {eps_threshold}")
        self.epsilon_threshold = eps_threshold
        if sample > eps_threshold:
        #if sample < eps_threshold:

            with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
                state = state.unsqueeze(0)
               
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)
        
    def return_eps_thresh(self):
        return self.epsilon_threshold
    
    def return_beta(self):
        return self.beta
    

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return                                  #if the length of the buffer isnt >= to the batch size then do nothing
        
        #Linear Increase of BETa
        beta = min(1.0, BETASTART + STEPS_DONE * (1.0 - BETASTART) / BETAFRAMES)

        #Exponential to match Eps Thresh
        #beta = BETASTART + (1.0 - BETASTART) * (1 - math.exp(-STEPS_DONE / BETADECAY))
        #sigmoid
        #beta = BETASTART + (1.0 - BETASTART) * (1 - math.exp(-((STEPS_DONE / BETADECAY) ** 2)))        #print(f"beta : {beta}")
        self.beta = beta
        
        transitions, indices, weights = self.replay_buffer.sample(BATCH_SIZE, beta=beta)
        
        batch = Transition(*zip(*transitions))
        
        #print(f"rewards from the batch of transitions{batch.reward}")
        #print(batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        
        #print(f"non final mask: {non_final_mask}")
        

        non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
        
        state_batch = torch.stack(batch.state, axis=0)
        print(f"batch.action: {batch.action} and shape ")
        time.sleep(5)
        action_batch = torch.cat(batch.action)
        
        reward_batch = torch.cat(batch.reward)

        print(f"action batch: {action_batch.shape} and state batch: {state_batch.shape}")

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        #next state values are 0 for next_state = None 
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        #Huber loss: less sensitive to outliers than standard MSE, behaves like MSE for small errors but MAE for large, improving stability
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        weighted_loss = (loss*torch.tensor(weights, device=device)).mean()
        #print(f"the loss : {loss} and weighted loss : {weighted_loss} of agent {self.index}")
        # Optimize the model
        self.optimiser.zero_grad()
        #gradients calculated through backprop
        weighted_loss.backward()
        #gradient clipping to mitigate exploding gradients, gradients > 100 clipped to 100, gradients <-100 clipped to -100
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()

        td_errors = expected_state_action_values - state_action_values.squeeze(1)
        
        new_priorities = abs(td_errors).detach() + 1e-6  # Add a small epsilon to avoid zero priority
        self.replay_buffer.update_priorities(indices, new_priorities.cpu().numpy())

        return loss

        


        
#Memory replay buffer used to sample from 

class MemoryReplay(object):
    def __init__(self, BUFFER_SIZE, alpha=0.6):
        self.memory_replay_buffer = deque([], maxlen=BUFFER_SIZE)
        self.priorities = deque([], maxlen=BUFFER_SIZE)
        self.alpha = alpha  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
        
    def insert(self, *args): 
        #print(f"self.priorities: {self.priorities}")
        max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0 # Assign max priority to new experience
        self.memory_replay_buffer.append(Transition(*args))
        self.priorities.append(max_priority)
    
    def sample(self, num_samples, beta=0.4):
        if len(self.memory_replay_buffer) < num_samples:
            raise ValueError("Not enough samples in buffer to sample the required number of samples.")
        
        priorities = np.array(self.priorities, dtype=np.float32)
        scaled_priorities = priorities ** self.alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        #print(f"sample probs: {sample_probs} and num samples: {num_samples}")
        indices = np.random.choice(len(self.memory_replay_buffer), num_samples, p=sample_probs)
        samples = [self.memory_replay_buffer[idx] for idx in indices]
        
        # Importance-sampling weights
        total = len(self.memory_replay_buffer)
        weights = (total * sample_probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory_replay_buffer)
        
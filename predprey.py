import pygame as pg
import gymnasium as gym
import time 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

from PredPreyEnvi import PredPreyEnv
from preyAgent import DQNPreyAgent, DQNModel, DQNConvModel
import torch


from itertools import count

from gymnasium.envs.registration import register

register(
    id="PreyPredEnv-v1",
    entry_point="PredPreyEnvi:PredPreyEnv",
    max_episode_steps=100
)

env = gym.make("PreyPredEnv-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

GRIDSIZE = 9
WIDTH,HEIGHT = 720, 560
ROWS, COLUMNS = GRIDSIZE,GRIDSIZE
FPS = 10
TAU = 0.008
#variables here
num_prey = 5
channels = 4    #represents the features i.e. should always be 4 unless change to features

def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    rewards_t = None    #just preventing rewards from being printed

    turns_t = torch.tensor(turns_in_episode, dtype=torch.float)

    
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Number of Turns')
    #plt.plot(rewards_t.numpy())
    plt.plot(turns_t.numpy())
    # Take 100 episode averages and plot them too
    if rewards_t == None:
        pass
    # elif len(rewards_t) >= 100:
    #     means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    if len(turns_t) >= 100:
        means = turns_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())



    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

prey_pos = np.array([])
pred_pos = np.array([])
berry_pos = np.array([])



#Testing//Main loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
#env = PredatorPreyEnv()
initial_state, info = env.reset()
input_shape = (channels,ROWS,COLUMNS)
prey_agents = []
alive = True

def get_correct_state(state,i):      
    correct_state = state[:3]
    #print(f"state[i:i+1]: {state[i:i+1]}")
    correct_state = np.append(correct_state, state[i:i+1], axis=0)
    return correct_state
         

#prey_agent = DQNPreyAgent(DQNModel(n_observations,env.action_space.n).to(device=device), env.action_space)

for prey_index in range(num_prey):
    prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, prey_index, alive))
    
#prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, 0, alive=True))
#prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, 1, alive=True))
#prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, 2, alive=True))

#pg.init()
turns_in_episode = []
episode_rewards = []
running = True
while running:
          
    if torch.cuda.is_available():
        num_episodes = 1800
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        print(f"episode: {i_episode}")
        if not running: break
        
        # Initialize the environment and get its state
        if i_episode!=0:
            initial_state, info = env.reset()
        #print(f"state rn: {initial_state}")
        print(f"info['prey']: {info['prey']}")
        for i in range(num_prey):
            prey_agents[i].alive = True     #make all agents alive again


        for t in count(): # indefinite loop with count()
            #print("------------------")
            if env.render_mode == "human" or "off":
                running = env.key_presses(running)
            if not running: break

            

            actions = {}
            self_identifier_initial_states_list = [0]*num_prey  # Assuming 4 channels; use more if needed

            for i in range(num_prey):
                if prey_agents[i].alive == True:         #ONLY SELECT ACTION IF PREY AGENT ALIVE
                    index = i+3
                    #print(f"index in select action: {index}")
                    #print(i)
                    #print(f"initial_state: {initial_state}")
                    #print(f"shape of sstate: {initial_state.shape}")
                    
                    self_identifier_initial_state = get_correct_state(initial_state, index)

                    #print(f"self_identifier_state: {self_identifier_initial_state}")
                    
                    #time.sleep(4)

                    self_identifier_initial_state_tensor = torch.from_numpy(self_identifier_initial_state).float().to(device)

                    self_identifier_initial_states_list[i] = self_identifier_initial_state_tensor
                    #print(f"self identifier states[i]: {self_identifier_initial_states_list[i]}")

                    #print(f"unique state {i}: {self_identifier_initial_state}")

                    #print(f"self_identifier_initial_state_tensor THIS ONE:  {self_identifier_initial_state_tensor}")
                    
                    actions[i] = prey_agents[i].select_action(self_identifier_initial_state_tensor)
                    #CNN
                else:
                    actions[i] = -1
                    self_identifier_initial_states_list[i] = None       #if dead from the getgo then set initial state to None
                
            
            
            observation, rewards, active_prey, terminated, info = env.step(actions) # .item() returns a scalar from a tensor with one value
            #print(f"info: {info}")
            #print(f"observation after step : {observation}")
            print(f"info['prey']: {info['prey']}")
            #print(f"prey agent alive before step in ep {i_episode}: {active_prey}")
           #       NEED TO MAKE OBSERVATION INTO THE CORRECT FORM

            
            #print(f"rewards: {rewards}")
            rewards = torch.tensor(rewards, device=device) # creates a tensor, [reward] converts it into a list with one element ensuring new tensor has one element
            #print(f"rewards: {rewards}")
            #print(f"rewards: {torch.tensor(rewards[i], device=device)}")
            
            #self_identifier_states = torch.tensor(self_identifier_state, device=device)
            #print(f"rewards: {rewards}")
            #print(f"rewards[0]: {rewards[0]}")

            
            done = terminated 

            for i in range(num_prey):                           #after the step, setting the dead agents' .aliv variable to false
                if active_prey[i] == False and prey_agents[i].alive:                     #just for ease
                    #print(f"inside alive prey thing!")
                    prey_agents[i].alive = False
            
            #print(f"prey agent alive after step in ep {i_episode}: {active_prey}")


            for i in range(num_prey):       
                if self_identifier_initial_states_list[i] == None:  #skip adding transition to buffer if dead from initial state
                    continue
                    #next_state = torch.tensor(observation_list, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0) adds a dimension of size 1 at desired location
                index = 3+i
                        #print(f"index in select action: {index}")
                next_state_i = get_correct_state(observation, index)
                next_state_i_tensor = torch.from_numpy(next_state_i).float().to(device)

                if prey_agents[i].alive == False:   #overwrite next state to none if dead
                    next_state_i_tensor = None
                        #print(f"next state {i}: {next_state_i_tensor}")
                #print(f"-------------------\n replay buffer inserts of agent {i}; initial state: {self_identifier_initial_states_list[i]}\n actions: {actions[i]}\n next state: {next_state_i_tensor}\n rewards: {torch.tensor(rewards[i], device=device)}\n ----------------")
                if actions[i] == -1:
                    print("this shit is fucked")
                prey_agents[i].replay_buffer.insert(self_identifier_initial_states_list[i], actions[i], next_state_i_tensor, torch.tensor([rewards[i]], device=device))

            #####################
            #IF AGENT DEAD THEN DONT DO ANY OPTIMISATION OR TRAINING 
            #####################
            
            #print("YOOO")
            # Store the transition in memory
            
            #prey_agent.replay_buffer.insert(state, action, next_state, reward) #REGULAR
            # Move to the next state
            
            #state = next_state     #REGULAR    
            initial_state = observation
            #print(f"the grid after assigning the next state to it: {grid}")
            # Perform one step of the optimization/training (on the policy network)
            
            for i in range(num_prey):
                if prey_agents[i].alive == True:            #only train if alive!
                    prey_agents[i].train()
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = prey_agents[i].target_net.state_dict()
                    policy_net_state_dict = prey_agents[i].policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    prey_agents[i].target_net.load_state_dict(target_net_state_dict)
            
            if env.render_mode == "human":
                env.render_frame()
            
            if done:
                current_ep_reward = env.get_inf()
                episode_rewards.append(current_ep_reward)
                turns_in_episode.append(t)
                #print(f"total reward this episode: {current_ep_reward}")
                # if len(episode_rewards) == 550:
                #     env.render_frame()
                plot_durations()
                break                                   #exits the episode
            print(f"current turn in episode: {t}")            
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()            

    
    #ACTION SET TO SMTH HERE THEN STEP TAKEN THEN RENDER
    
    
    
    #draw
pg.quit()

import pygame as pg
import gymnasium as gym
import time 
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

from PredPreyEnvi import PredPreyEnv
from preyAgent import DQNPreyAgent, DQNConvModel
import torch


from itertools import count

from gymnasium.envs.registration import register

register(
    id="PreyPredEnv-v1",
    entry_point="PredPreyEnvi:PredPreyEnv",
    max_episode_steps=150
)

env = gym.make("PreyPredEnv-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

GRIDSIZE = 5
WIDTH,HEIGHT = 720, 560
ROWS, COLUMNS = GRIDSIZE,GRIDSIZE
FPS = 10
TAU = 0.005     #up is learning is slow which it is so up it !
#variables here
num_prey = 2
channels = 6    #represents the features i.e. should always be 6 unless change to features

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 34
env.seed(SEED)
set_seed(SEED)


# def plot_durations(show_result=False):
#     plt.figure(1)
#     rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
#     rewards_t = None    #just preventing rewards from being printed

#     turns_t = torch.tensor(turns_in_episode, dtype=torch.float)

    
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Number of Turns')
#     #plt.plot(rewards_t.numpy())
#     plt.plot(turns_t.numpy())
#     # Take 100 episode averages and plot them too
#     if rewards_t == None:
#         pass
#     # elif len(rewards_t) >= 100:
#     #     means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
#     #     means = torch.cat((torch.zeros(99), means))
#     #     plt.plot(means.numpy())

#     if len(turns_t) >= 100:
#         means = turns_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())



#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())


colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
def plot_durations(show_result=False):
    plt.figure(1, figsize=(10, 10))
    
    rewards_t_agent0 = torch.tensor(episode_rewards0, dtype=torch.float)
    rewards_t_agent1 = torch.tensor(episode_rewards1, dtype=torch.float)
    turns_t = torch.tensor(turns_in_episode, dtype=torch.float)
    altruism0_t = torch.tensor(altruism0,dtype=torch.float)
    altruism1_t = torch.tensor(altruism1,dtype=torch.float)
    berries_held_avg0_t = torch.tensor(berries_held_avg0, dtype=torch.float)
    berries_held_avg1_t = torch.tensor(berries_held_avg1, dtype=torch.float)
    
    
    # eps_thresholds_t = torch.tensor(epsilon_thresholds, dtype=torch.float)
    # betas_t = torch.tensor(betas, dtype=torch.float)

    plt.clf()

    if show_result:
        plt.suptitle('Epsilon Threshold Decay and Beta - Importance Sampling Corrections')
    else:
        plt.suptitle('Training...')

    # First subplot for rewards

    # plt.xlabel('Episode')
    # plt.ylabel('Value')
    # #plt.plot(eps_thresholds_t.numpy(), label='Epsilon Threshold Decay', color=(0,0.4,1))
    # #plt.plot(betas_t.numpy(), label='Beta - Importance Sampling Corrections', color=(1,0.6,0.3))
    # plt.legend()
    
    #---------------------------------
    #GOOD STUFF HERE \/\/\/\/\/\/
    #---------------------------------
    plt.subplot(2, 2, 1)
    plt.plot(rewards_t_agent0.numpy(), label='Rewards Agent 0', color=(0.7,0.7,1))
    plt.plot(rewards_t_agent1.numpy(), label='Rewards Agent 1', color=(1,0.7,0.7))
    plt.legend()
   
    if len(rewards_t_agent0) >= 100:
        means = rewards_t_agent0.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average Agent 0', color=(0.4,0.4,1))
    plt.legend()
   
   
    if len(rewards_t_agent1) >= 100:
        means = rewards_t_agent1.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average Agent 1', color=(1,0.4,0.4))
    plt.legend()
    # Second subplot for number of turns
    plt.subplot(2, 2, 2)
    plt.xlabel('Episode')
    plt.ylabel('Number of Turns')
    plt.plot(turns_t.numpy(), label='Turns', color='orange')
    if len(turns_t) >= 100:
        means = turns_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average', color='green')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.xlabel('Episode')
    plt.ylabel('Number of Altruistic Acts')
    plt.plot(altruism0_t.numpy(), label='Altruisitic Act Agent 0', color=(1,0.8,0.4))
    plt.plot(altruism1_t.numpy(), label='Altruisitic Act Agent 1', color=(1,0.4,0.1))
    if len(altruism0_t) >= 100:
        means = altruism0_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average Agent 0', color=(0.6,0.8,0.6))
    if len(altruism1_t) >= 100:
        means = altruism1_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average Agent 0', color=(0.4,0.6,0.4))
    plt.legend()

    plt.subplot(2,2,4)
    plt.xlabel('Episode')
    plt.ylabel('Number of Berries')
    plt.plot(berries_held_avg0_t.numpy(), label='Avg. Num. Berries Held Agent 0', color=(0.8, 0.33, 0.6))
    plt.plot(berries_held_avg1_t.numpy(), label='Avg. Num. Berries Held Agent 1', color=(1, 0.75, 0.9))
    if len(berries_held_avg0_t) >= 100:
        means = berries_held_avg0_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average Agent 0', color=(0.6, 0.15, 0.4))
    if len(berries_held_avg1_t) >= 100:
        means = berries_held_avg1_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-episode average Agent 1', color=(0.4, 0.15, 0.2))
    plt.legend()
    plt.tight_layout()
 
    #---------------------------------
    #---------------------------------
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
initial_state, info = env.reset()
input_shape = (channels,ROWS,COLUMNS)
prey_agents = []
alive = True

def get_correct_state(state,i):      
    correct_state = state[:5]
    #print(f"state[i:i+1]: {state[i:i+1]}")
    correct_state = np.append(correct_state, state[i:i+1], axis=0)
    return correct_state
         

# for prey_index in range(num_prey):
#     prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, prey_index, alive))

prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, 0, alive))
prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, 1, alive))


turns_in_episode = []
episode_rewards0 = []
episode_rewards1 = []
epsilon_thresholds = []
betas = []
losses = []
berries_held_avg0 = []
berries_held_avg1 = []
altruism0 = []
altruism1 = []
running = True
while running:
          
    if torch.cuda.is_available():
        num_episodes = 2500
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        print(f"episode: {i_episode}")
        if not running: break
        
        # Initialize the environment and get its state
        if i_episode!=0:
            initial_state, info = env.reset()
        for i in range(num_prey):
            prey_agents[i].alive = True     #make all agents alive again

        berry_running_total0 = 0
        berry_running_total1 = 0
        total_altruism_acts0 = 0
        total_altruism_acts1 = 0
        
        for t in count(): # indefinite loop with count()
            if env.render_mode == "human" or "off":
                running = env.key_presses(running)
            if not running: break


            actions = {}
            self_identifier_initial_states_list = [0]*num_prey  # Assuming 4 channels; use more if needed

            for i in range(num_prey):
                if prey_agents[i].alive == True:         #ONLY SELECT ACTION IF PREY AGENT ALIVE
                    index = i+5
                   
                    self_identifier_initial_state = get_correct_state(initial_state, index)

                    #print(f"self identifier initial state is :{self_identifier_initial_state}")
                    #time.sleep(5)
                    self_identifier_initial_state_tensor = torch.from_numpy(self_identifier_initial_state).float().to(device)
                    
                    self_identifier_initial_states_list[i] = self_identifier_initial_state_tensor
                  
                    actions[i] = prey_agents[i].select_action(self_identifier_initial_state_tensor)
                    #CNN
                else:
                    actions[i] = -1
                    self_identifier_initial_states_list[i] = None       #if dead from the getgo then set initial state to None
                
            
            
            observation, rewards, active_prey, terminated, info = env.step(actions) # .item() returns a scalar from a tensor with one value
            print(f"altrusim:{info['altruism']}")
            #print(f"rewards : {rewards}")
            done = terminated 

            for i in range(num_prey):                           #after the step, setting the dead agents' .aliv variable to false
                if active_prey[i] == False and prey_agents[i].alive:                     #just for ease
                    prey_agents[i].alive = False
            


            for i in range(num_prey):       
                if self_identifier_initial_states_list[i] == None:  #skip adding transition to buffer if dead from initial state
                    continue
                index = 5+i
                next_state_i = get_correct_state(observation, index)
                next_state_i_tensor = torch.from_numpy(next_state_i).float().to(device)

                if prey_agents[i].alive == False:   #overwrite next state to none if dead
                    next_state_i_tensor = None
                #print(f"-------------------\n replay buffer inserts of agent {i}; initial state: {self_identifier_initial_states_list[i]}\n actions: {actions[i]}\n next state: {next_state_i_tensor}\n rewards: {torch.tensor(rewards[i], device=device)}\n ----------------")
                if actions[i] == -1:
                    print("bug!")

                print(actions)
                prey_agents[i].replay_buffer.insert(self_identifier_initial_states_list[i], actions[i], next_state_i_tensor, torch.tensor([rewards[i]], device=device))

            
            #state = next_state     #REGULAR    
            initial_state = observation
            
            for i in range(num_prey):
                if prey_agents[i].alive == True:            #only train if alive!
                    loss = prey_agents[i].train()
                    #soft update of target networks weights
                    losses.append(loss)
                    target_net_state_dict = prey_agents[i].target_net.state_dict()
                    policy_net_state_dict = prey_agents[i].policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    prey_agents[i].target_net.load_state_dict(target_net_state_dict)
            
            if env.render_mode == "human":
                env.render_frame()
            
            #print(f"berries held after info: {info['number of berries held'][0]} and {info['number of berries held'][1]}")
            berry_running_total0 += info['number of berries held'][0]
            berry_running_total1 += info['number of berries held'][1]

            #print(f"berry running total: {berry_running_total0}")
            if done:
                current_ep_rewards = env.get_inf()

                berry_average0 = berry_running_total0/t+1
                berry_average1 = berry_running_total1/t+1
                

                total_altruism_acts0 = info['altruism'][0]
                total_altruism_acts1 = info['altruism'][1]
                altruism0.append(total_altruism_acts0)
                altruism1.append(total_altruism_acts1)
                
                #print(f"current ep rewards: {current_ep_rewards}")
                current_ep_rewards_0 = current_ep_rewards[0]
                current_ep_rewards_1 = current_ep_rewards[1]
                episode_rewards0.append(current_ep_rewards_0)
                episode_rewards1.append(current_ep_rewards_1)
                berries_held_avg0.append(berry_average0)
                berries_held_avg1.append(berry_average1)
                print(f"t: {t}")
                print(f"altrusim:{info['altruism']}")
                #print(f"berries: {info['number of berries held']}")

                
                # betas.append((prey_agents[0].return_beta()+prey_agents[1].return_beta())/2)
                # epsilon_thresholds.append((prey_agents[0].return_eps_thresh()+prey_agents[1].return_eps_thresh())/2)
                
                
                #print(f"episode rewards of 0 and 1: {episode_rewards0} and {episode_rewards1}")
                #time.sleep(3)
                turns_in_episode.append(t)
                plot_durations()
                break                                   #exits the episode
            #print(f"turn {t} completed in episode {i_episode}")  
        #time.sleep(2)          
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()            
    
    #ACTION SET TO SMTH HERE THEN STEP TAKEN THEN RENDER
    
    #draw
pg.quit()

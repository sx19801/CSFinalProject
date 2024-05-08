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
    max_episode_steps=300
)

env = gym.make("PreyPredEnv-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


WIDTH,HEIGHT = 480, 360
ROWS, COLUMNS = 5,5
FPS = 10
TAU = 0.005
#variables here

def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
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
num_prey = 2


#Testing//Main loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#env = PredatorPreyEnv()
state, info = env.reset()
#print(f"state looks like: {state}")
#state = list(state.values())       #THIS IS FOR WHEN STATE RETURNS A DICT
#print(f"state looks like: {state}")

state = np.concatenate(state) # xy of prey, pred, berry in the form [X_PREY, Y_PREY, X_PRED, Y_PRED, X_BERRY, Y_BERRY]
#print(f"state now looks like: {state}")

n_observations = len(state)
#print(f"observations is now: {n_observations}")
#print(f'action space .n: {env.action_space.n}')
#time.sleep(5)

#prey_agent = DQNPreyAgent(DQNModel(n_observations,env.action_space.n).to(device=device), env.action_space)

input_shape = (4,5,5)
prey_agents = []
alive = True
for prey_index in range(num_prey):
    prey_agents.append(DQNPreyAgent(DQNConvModel(input_shape, env.action_space.n).to(device), env.action_space, prey_index, alive))
    


#pg.init()
running = True
episode_rewards = []
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
            break
        
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False
                break
            
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        #print(f"state rn: {state}")
        #state_prey = state["prey"]     #DEPRECATED
        #state_pred = state["pred"]     #DEPRECATED
        #state_berry = state["berry"]   #DEPRECATED
        #print(f"state_prey before concatenate: {state_prey}")
        #state = np.concatenate([state_prey, state_pred, state_berry])  #DEPRECATED
        #print(f"the state values in the num episodes loop: {state}")
        
        grid = torch.zeros((input_shape[0], ROWS, COLUMNS))  # 4 channels for prey, predator, berry and self identification


        #THE FOLLOWING FOR WHEN STATE REPRESENTED AS COORDINATES NOT GRID
        # for i in range(0, len(state), 2):
        #     x, y = state[i], state[i + 1]
            
        #     # Check if coordinates are within bounds
        #     if x >= 0 and x < COLUMNS and y >= 0 and y < ROWS:
        #         grid[i // 2, x, y] = 1  # Set the corresponding cell to 1 for each agent
        #     else:
        #         print(f"Invalid coordinates ({x}, {y}) for agent {i // 2 + 1}. Ignoring.")

        # Add a batch dimension
        #print(f"grid is: {grid}")
        grid = grid.unsqueeze(0)
        grid = grid.to(device)
        # Convert the grid to a PyTorch tensor
        grid = torch.tensor(grid, dtype=torch.float32)
        #time.sleep(5)
        print(f"grid is: {grid}")
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #print(f"state size is: {state.size()}")
        #print(f"state is: {state}")
        #print(f"the state as a tensor after concatenating: {state}")
        #time.sleep(5)
        
        for t in count(): # indefinite loop with count()
            # print("------------------")
            # print(f"THE CURRENT STEP IN EPISODE {i_episode}: {t}")
            #print(f"the state going into the select_action method: {state}")
            
            #action = prey_agent.select_action(state)    #REGULAR
            actions = {}
            for i in range(num_prey):
                index = i
                if prey_agents[i].alive == True:         #ONLY SELECT ACTION IF PREY AGENT ALIVE
                    actions[index] = prey_agents[i].select_action(grid)
                    #CNN
                else:
                    actions[index] = -1
                
            #print(f"action is: {actions}")  
            #actions_scalar = [action.item() for action in actions]
            #print(f"actions_scalar: {actions_scalar}")
            #time.sleep(5)
            observation, reward, terminated, truncated, _ = env.step(actions) # .item() returns a scalar from a tensor with one value
            #print(f"observation : {observation}")
            
            observation_list = list(observation.values())
            observation_list = np.concatenate(observation_list)
            reward = torch.tensor([reward], device=device) # creates a tensor, [reward] converts it into a list with one element ensuring new tensor has one element
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation_list, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0) adds a dimension of size 1 at desired location



            #####################
            #IF AGENT DEAD THEN DONT DO ANY OPTIMISATION OR TRAINING 
            #####################
            
            #print("YOOO")
            # Store the transition in memory
            prey_agent.replay_buffer.insert(grid, action, next_state, reward)
            #prey_agent.replay_buffer.insert(state, action, next_state, reward) #REGULAR
            # Move to the next state
            
            #state = next_state     #REGULAR    
            grid = next_state
            #print(f"the grid after assigning the next state to it: {grid}")
            # Perform one step of the optimization/training (on the policy network)
            prey_agent.train()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = prey_agent.target_net.state_dict()
            policy_net_state_dict = prey_agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            prey_agent.target_net.load_state_dict(target_net_state_dict)

            env.render_frame()
            
            if done:
                current_ep_reward = env.get_inf()
                episode_rewards.append(current_ep_reward)
                print(f"total reward this episode: {current_ep_reward}")
                # if len(episode_rewards) == 550:
                #     env.render_frame()
                plot_durations()
                break
            print(f"current turn in episode: {t}")            
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()            

    
    #ACTION SET TO SMTH HERE THEN STEP TAKEN THEN RENDER
    
    
    
    #draw
pg.quit()

import numpy as np
import pygame as pg
import random
import time
import gymnasium as gym
from gymnasium import spaces

GRIDSIZE = 5
SEED = 34
class PredPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}
    
    def __init__(self, render_mode="human", grid_height=GRIDSIZE, grid_width=GRIDSIZE, num_prey=2, num_pred=0, num_berry=2, num_channels=4):
        self.grid_size = (grid_width, grid_height)
        self.window_size = [720,560]
        self.num_prey = num_prey
        self.num_pred = num_pred
        self.num_berry = num_berry
        self.num_channels = num_channels
        self.num_channels = 5 + num_prey    #the three channels being all prey,pred,berry and others being self identifiers
        self.current_ep_total_reward = np.zeros(self.num_prey)
        self.num_agents = self.num_berry+self.num_pred+self.num_prey
        self.current_turn_rewards = np.zeros(self.num_prey)
        
        #standard implementation ONLY WALKING
        #self.action_space = spaces.Discrete(4)  # Example: 4 actions (up, down, left, right)
        self.action_space = spaces.Discrete(5)
        
        #berry stuff       
        self.conflict = False
        self.throw_successful = [False]*self.num_prey
        self.died_this_turn = [False]*self.num_prey
        self.berry_energy_amount = 5
        self.starting_energy = 10
        self.starting_berry_amount = 1
        self.prey_energy = [self.starting_energy]*self.num_prey
        self.prey_berry_amount = [self.starting_berry_amount]*self.num_prey
        self.collected_berry = [False]*self.num_prey
        #print(f"self.prey_berry_amount in init:{self.prey_berry_amount}")
        #print(f"prey energy init {self.prey_energy}")
        #time.sleep(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_channels, self.grid_size[0], self.grid_size[1]), dtype=int)
        
        #pred stuff
        self.moving_out_of_grid = [False]*self.num_prey
        self.prey_died_this_turn = False
        self.target_prey = None
        
        self.currently_pursuing_locations = np.zeros((self.num_pred, 2))
        self.currently_pursuing = np.zeros((self.num_pred, 1))

        #print(f"currently_pursuing shape: {self.currently_pursuing}")
        #print(f"currently_pursuinh shape: {self.currently_pursuing[0]}")
        

        self.active_prey = np.ones(self.num_prey, dtype=bool)
        
        self.prey_locations = np.array([[0,0],[4,4]])
        spawn_pool_pred = self.spawning()
        self.pred_locations = np.array([spawn_pool_pred.pop() for _ in range(self.num_pred)])
        spawn_pool_other = self.spawning_other(spawn_pool_pred)
        #self.prey_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_prey)])
        self.berry_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_berry)])
        
        self.number_of_altruistic_acts = [0]*self.num_prey
        #print(f"berry locations: {self.berry_locations} and shape {self.berry_locations.shape}")
       
        self.active_prey_locations = self.prey_locations
        #print(self.prey_locations.shape)
                                                                              
        #print(f"prey_locations form: {self.observation_space}")
        self._action_to_direction = {
            0: np.array([0,1]),   #up
            1: np.array([1,0]),   #right
            2: np.array([0,-1]),  #down
            3: np.array([-1,0]),  #left
            #4: np.array([0,0]),   #stay still
            4: "THROW",               #throw berry
            -1: np.array([0,0])   #no change
        }
            
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def spawning_other(self, spawn_pool):
        excluded_positions = set()
        random.seed(SEED)
        for pos in self.prey_locations:
            # Add the position itself and all adjacent positions
            x, y = pos
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                        excluded_positions.add((nx, ny))

        # Filter the spawn pool to remove excluded positions
        spawn_pool = [pos for pos in spawn_pool if tuple(pos) not in excluded_positions]
        random.shuffle(spawn_pool)

        return spawn_pool

    def spawn_berry(self, prey_locations, berry_number):
        #np.random.seed(SEED)
        #print(f"prey locations: {prey_locations}  pred locations: {self.pred_locations}  berry locations: {self.berry_locations}")
        #print(f"size of pbrry locations: {self.berry_locations.shape} and prey shape: {prey_locations.shape} sie of pred: {self.pred_locations.shape}")
        if self.num_pred == 0:
            occupied_locations = np.concatenate((prey_locations, self.berry_locations))
        else:
            occupied_locations = np.concatenate((prey_locations, self.pred_locations, self.berry_locations), axis=0)

        #print(f"pred sie: {self.pred_locations.shape}")
        occupied_locations = set(map(tuple, occupied_locations))
        #print(f"array occupied: {occupied_locations}")
        spawn_pool = np.array([[x, y] for x in range(self.grid_size[0]) for y in range(self.grid_size[1])])
            
        mask = np.array([tuple(loc) not in occupied_locations for loc in spawn_pool])
        #print(f"spawnpool: {spawn_pool}")
        
        available_locations = spawn_pool[mask]
        #print(f"available locations: {available_locations}")
        if available_locations.size > 0:
            random_location = available_locations[np.random.choice(available_locations.shape[0])]
        #print(f"available spawns: {random_location}")
        random_location = random_location.reshape(1,2)
        #print(f"random location: {random_location}")
        #print(f"berry locations before adding: {self.berry_locations}")
        #print(f"berry number: {berry_number}")
        self.berry_locations = np.delete(self.berry_locations, berry_number, axis=0)
        self.berry_locations = np.vstack([self.berry_locations, random_location])
        #print(f"berry locations after adding: {self.berry_locations}")
        #time.sleep(2)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def spawning(self):
        # Create a full grid of possible positions
        spawn_pool = [np.array([x, y]) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        random.shuffle(spawn_pool)  # Shuffle the spawn pool to randomize agent placement
        return spawn_pool
     
    def find_agent_position(grid, channel):
        # Locate the agent in the grid (where the agent is represented by 1)
        positions = np.where(grid[channel] == 1)
        if positions[0].size == 0:
            return None  # Agent not found
        return positions[0][0], positions[1][0]  # Return the first found position (y, x)

    def move_agent(current_position, direction, grid_size):
        y, x = current_position
        if direction == 'up':
            y = max(0, y - 1)
        elif direction == 'down':
            y = min(grid_size[0] - 1, y + 1)
        elif direction == 'left':
            x = max(0, x - 1)
        elif direction == 'right':
            x = min(grid_size[1] - 1, x + 1)
        return (y, x)   
    
    def _get_obs(self, prey_locations):      #adds all the stuff to the grid
        grid = np.zeros((self.num_channels, self.grid_size[0], self.grid_size[1]), dtype=int)  # Assuming 1 channel; use more if needed
        
        #print(f"the grid in _get_obs(): {grid}")
        #print(f"the prey locations in _get_obs(): {prey_locations}")
        # self.prey_locations = self.prey_locations[0]          #making the list of arrays into just an array i.e. from [[x,y]] to [x,y]
        # self.pred_locations = self.pred_locations[0]
        # self.berry_locations = self.berry_locations[0]
        #print(f"self.prey_berry_amount:{self.prey_berry_amount}")
        #print(len(self.prey_locations))
        #print(self.prey_location[1])
        #print(self.prey_location[0])
        for k, active in enumerate(self.active_prey):
            if active:  
                location = prey_locations[k]
                grid[0, location[0], location[1]] = 1  # Mark prey

        #print(f"grid 0: {grid[0]}")
        for location in self.berry_locations:
            grid[1, location[0], location[1]] = 1  # Mark berry

        for l, active_prey in enumerate(self.active_prey):
            if active_prey:
                prey_location = prey_locations[l]
                grid[2, prey_location[0],prey_location[1]] = self.prey_berry_amount[l] #Mark berry amount of each prey

        for m, location in enumerate(self.prey_locations):
            grid[3, location[0], location[1]] = self.prey_energy[m] #prey energy channels

        for n in range(self.num_prey):
            if self.throw_successful[n]:
                grid[4, prey_locations[n][0], prey_locations[n][1]] = 1 #mark if agent has thrown successfully this turn
        
        #print(grid[2])        
        for i in range(self.num_prey):
            if self.active_prey[i] == True:
                grid[i+5, prey_locations[i][0], prey_locations[i][1]] = 1     # Mark self

        #print(f"grid: {grid}")
        return grid
    
    def _get_info(self): # gets some kind of info i.e. could be distances from berry and pred etc.
        return {"prey": self.prey_locations.squeeze(), "pred": self.pred_locations.squeeze(), "berry": self.berry_locations.squeeze(), "alive_prey": self.active_prey, "altruism": self.number_of_altruistic_acts, "number of berries held": self.prey_berry_amount}
    
    def get_inf(self): # gets some kind of info i.e. could be distances from berry and pred etc.
        return self.current_ep_total_reward
    
    def step(self, actions):
        self.died_this_turn = [False]*self.num_prey
        self.throw_successful = [False]*self.num_prey
        self.collected_berry = [False]*self.num_prey
        #actions is a dict of the form: PreyIndexNumber: Action or False
        #need to check whether action is false or true 
        #print("-----------")
        #print(f"self.prey_locations: {self.prey_locations}")
        #print(f"self.pred_locations: {self.pred_locations}")
        assert len(actions) == self.num_prey, "each agent must have an action"
        
        #print(actions)
        self.scalar_actions = []
        episode_done = []
        #print("-----------")
        #print(f"prey energy at beginning of step {self.prey_energy}")
        #print("-----------")
        for key in actions:
            #print(f"actions[key]: {actions[key]}")
            if actions[key] == -1:        #check for if the agent is dead
                self.scalar_actions.append(-1)           #assign -1 to action to represent dead
                self.active_prey[key] = False
            else:
                self.scalar_actions.append(actions[key].item())
        
        #print(f"active prey: {self.active_prey}")
        #print(f"self.prey energy: {self.prey_energy}")
        new_prey_locations = np.array(self.prey_locations, copy=True)
        #print(f"actions are : {self.scalar_actions}")
        for index, scalar_action in enumerate(self.scalar_actions):
            #print(f"index is : {index}")
            if scalar_action != -1:        #dead check
                if scalar_action == 0 and self.prey_locations[index][1] < self.grid_size[1]:  # Move up
                    #print(new_prey_locations[index])
                    new_prey_locations[index][1] += 1
                    if new_prey_locations[index][1] == self.grid_size[1]:
                        new_prey_locations[index][1] = self.grid_size[1]-1
                        self.prey_energy[index] = self.prey_energy[index]-1
                        self.moving_out_of_grid[index] = True
                    else:
                        self.prey_energy[index] = self.prey_energy[index]-1
                    #print(new_prey_locations[index])
                elif scalar_action == 1 and self.prey_locations[index][0] < self.grid_size[0]:  # Move right
                    #print(new_prey_locations[index])
                    new_prey_locations[index][0] += 1
                    #print(f"prey energy before {self.prey_energy[index]}")
                    if new_prey_locations[index][0] == self.grid_size[0]:
                        new_prey_locations[index][0] = self.grid_size[0]-1
                        self.prey_energy[index] = self.prey_energy[index]-1
                        self.moving_out_of_grid[index] = True
                    else:
                        self.prey_energy[index] = self.prey_energy[index]-1
                    #print(new_prey_locations[index])
                    #print(f"prey energy after {self.prey_energy[index]}")
                elif scalar_action == 2 and self.prey_locations[index][1] > -1:  # Move down
                   # print(new_prey_locations[index])
                    new_prey_locations[index][1] -= 1
                    if new_prey_locations[index][1] == -1:
                        new_prey_locations[index][1] = 0
                        self.prey_energy[index] = self.prey_energy[index]-1
                        self.moving_out_of_grid[index] = True
                    else:
                        self.prey_energy[index] = self.prey_energy[index]-1
                   # print(new_prey_locations[index])
                elif scalar_action == 3 and self.prey_locations[index][0] > -1:  # Move right
                    #print(new_prey_locations[index])
                    new_prey_locations[index][0] -= 1
                    if new_prey_locations[index][0] == -1:
                        new_prey_locations[index][0] = 0
                        self.prey_energy[index] = self.prey_energy[index]-1
                        self.moving_out_of_grid[index] = True
                    else:
                        self.prey_energy[index] = self.prey_energy[index]-1
                        
                    #print(new_prey_locations[index])
                
                elif scalar_action == 4: #Throw berry to other agent
                    #print("throwing")
                    if index == 0:
                        target_agent = 1
                    else:
                        target_agent = 0

                    if self.active_prey[target_agent] == False: #if target agent dead then dont throw adn successful throw remains false, prey rests
                        new_prey_locations[index] = self.prey_locations[index]
                        self.prey_energy[index] = self.prey_energy[index]-1
                        
                    elif self.prey_berry_amount[index] > 0:     #if more than 1 berry throw the berry
                        self.prey_berry_amount[index] = self.prey_berry_amount[index]-1
                        self.prey_berry_amount[target_agent] = self.prey_berry_amount[target_agent] + 1
                        self.throw_successful[index] = True
                        self.prey_energy[index] = self.prey_energy[index]-1
                    else:
                        new_prey_locations[index] = self.prey_locations[index]
                        self.prey_energy[index] = self.prey_energy[index]-1

                else:
                    new_prey_locations[index] = self.prey_locations[index] #not dead but moving out of border
                    self.moving_out_of_grid[index] = True
                    self.prey_energy[index] -= 1

            elif self.scalar_actions[index] == -1:      #dead check
                #print("dead index shit working")
                new_prey_locations[index] = self.prey_locations[index]
        
        new_prey_locations_resolved = self.resolve_conflicts(new_prey_locations, self.current_ep_total_reward)
       
        #print(f"prey energy before {self.prey_energy}")
        if self.active_prey[0] and self.active_prey[1]:
            if np.array_equal(new_prey_locations_resolved[0], new_prey_locations_resolved[1]):
                print("we got a problem")
                time.sleep(5)

        for i in range(self.num_prey):
            if self.active_prey[i] == True:
        #         print(f"self.scalar actions: {self.scalar_actions} and unqieu {self.scalar_actions[i]} ")
                
                #print(f"new prey locations i : {self.prey_locations[i]} and new prey locations resolved: {new_prey_locations_resolved[i]} and conflict? : {self.conflict}")
                
                if np.array_equal(self.prey_locations[i], new_prey_locations_resolved[i]) and self.moving_out_of_grid[i] == False and self.scalar_actions[i] != 4:
                    self.prey_energy[i]+=1

        #print(f"prey energy after {self.prey_energy}")

        #         if self.throw_successful[i] == False:
        #             self.prey_energy[i] = self.prey_energy[i]+1
        #         elif np.array_equal(new_prey_locations[i], new_prey_locations_resolved[i]) and self.conflict:   #attempt move but cant cause the guy is trying to throw or smth
        #             if self.moving_out_of_grid[i] == False:
        #                 print(f"prey energy {i} restored due to conflict resolution")
        #                 self.prey_energy[i] = self.prey_energy[i]+1                 #if agent has not moved as a result of a conflict then give energy back, only penalising movement that is on edge of grid
        #                 #print(f"prey energy: {self.prey_energy}")
        #                 #time.sleep(3)
                
        # for i in range(self.num_prey):
        #     if self.active_prey[i]:
        #         if np.array_equal(new_prey_locations[i], new_prey_locations_resolved[i]):
                    

        # self.conflict = False
       #check whether berry collected
        for i in range(self.num_prey):  
            if self.active_prey[i] == True:
                for j in range(self.num_berry):
                    if np.array_equal(new_prey_locations_resolved[i], self.berry_locations[j]):
                        self.prey_berry_amount[i] = self.prey_berry_amount[i]+1
                        self.spawn_berry(new_prey_locations_resolved, j)
                        self.collected_berry[i] = True 

        #check whether agent has 0 energy
        for i in range(self.num_prey):
            if self.prey_energy[i]<1:
                if self.prey_berry_amount[i] > 0:
                    self.prey_berry_amount[i] = self.prey_berry_amount[i]-1         #eat a berry
                    self.prey_energy[i] += self.berry_energy_amount
                else:
                    self.died_this_turn[i]=True                                     #dead if no berries

        

        #ASSIGNING REWARDS
        self.current_turn_rewards = np.zeros(self.num_prey)
        for i in range(self.num_prey):
            if self.active_prey[i] == True:
                if self.died_this_turn[i]:
                    self.active_prey[i]=False
                    self.current_turn_rewards[i] = -20   
                elif self.collected_berry[i]:
                    self.current_turn_rewards[i] = 10
                elif self.throw_successful[i]:
                    if i == 0:
                        if self.prey_berry_amount[1] < 1:
                            self.current_turn_rewards[i] = 6
                            self.number_of_altruistic_acts[0] +=1
                            #time.sleep(4)
                    elif i == 1:
                        if self.prey_berry_amount[0] < 1:
                            self.current_turn_rewards[i] = 6
                            self.number_of_altruistic_acts[1] +=1
                            #time.sleep(4)
                #DEATH
                                
               
                elif self.current_turn_rewards[i] == 0:
                    if self.throw_successful[i] == False and self.scalar_actions[i] == 4:
                        if np.any(self.active_prey == False):
                            self.current_turn_rewards[i] = -0.2
                        else:
                            self.current_turn_rewards[i]  = -0.1     #selfishness PENALISING GIVINg


                    elif np.array_equal(self.prey_locations[i], new_prey_locations_resolved[i]) and self.moving_out_of_grid[i] == False:  #no penalty if staying in same position as a result of conflict resolution
                        self.current_turn_rewards[i] = 0
                    elif np.array_equal(self.prey_locations[i], new_prey_locations_resolved[i]) and self.moving_out_of_grid[i] == True:
                        self.current_turn_rewards[i] = -0.5
                    else:
                        self.current_turn_rewards[i] = 0.3
                    
                        #     if new_prey_locations_resolved[index] == self.prey_locations[index]:
                #         self.current_ep_total_reward = 0
            
        self.prey_locations = new_prey_locations_resolved #setting old positions to new positions

        episode_done = not any(self.active_prey)    #finish ep if no prey agents alive

        observation = self._get_obs(new_prey_locations_resolved)
        #print(observation)
        #time.sleep(5)
        info = self._get_info()
        #print(f"current ep total reward before adding: {self.current_ep_total_reward}")
        print(f"current turn reward : {self.current_turn_rewards}")
        if self.current_ep_total_reward[1] < -200:
            time.sleep(60)
        self.current_ep_total_reward = self.current_ep_total_reward+self.current_turn_rewards
        #print(f"current ep total reward after adding: {self.current_ep_total_reward}")
        #time.sleep(1)

        if self.render_mode == "human":
            pass
            #self.render_frame()
      
        #print(f"self.prey_berry_amount after step:{self.prey_berry_amount}")
        self.prey_shouting_this_turn = False    #end of turn no longer shouting
        self.moving_out_of_grid = [False]*self.num_prey
        self.turns_until_move_counter+=1
        return observation, self.current_turn_rewards, self.active_prey, episode_done, info    

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        # and return the initial observation
        #print("------RESET CALLED------")
        #following line to seed np.random
        super().reset(seed=seed)
        self.active_prey = np.ones(self.num_prey, dtype=bool)
        self.prey_energy = [self.starting_energy]*self.num_prey
        self.prey_berry_amount = [self.starting_berry_amount]*self.num_prey
        #print(f"self.prey_berry_amount in reset:{self.prey_berry_amount}")
        self.turns_until_move_counter = 0
        self.target_prey = None
        spawn_pool_pred = self.spawning()
        self.pred_locations = np.array([spawn_pool_pred.pop() for _ in range(self.num_pred)])
        #self.prey_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_prey)])
        self.prey_locations = np.array([[0,0],[4,4]])
        spawn_pool_other = self.spawning_other(spawn_pool_pred)
        self.throw_successful = [False]*self.num_prey
        self.berry_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_berry)])
        self.collected_berry = [False]*self.num_prey
        self.number_of_altruistic_acts = [0]*self.num_prey
        self.currently_pursuing_locations = np.zeros((self.num_pred, 2))
        self.currently_pursuing = np.zeros((self.num_pred, 1))

        self.current_ep_total_reward = np.zeros(self.num_prey)
        self.active_prey_locations = self.prey_locations
        observation = self._get_obs(self.prey_locations)
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render_frame()
        #time.sleep(1)
        return observation, info 

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.render_frame()
                   
    def render_frame(self):
        if self.window is None and self.render_mode == "human" :
        #initialise pygame and the display window
            pg.init()
            pg.display.init()
            pg.display.set_caption('Predator, Prey and Berries')
            self.window = pg.display.set_mode(
                (self.window_size)
            )
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()
        
        
        self.window.fill((43,43,43))
        canvas = pg.Surface((500, 500))
        canvas.fill((54, 54, 54))
        pix_square_size = (
            canvas.get_size()[0] / self.grid_size[0]
        )  # The size of a single grid square in pixels

        pix_square_size = int(pix_square_size)
        
        
        #draw pred
        for pred in range(self.num_pred):
            pg.draw.rect(
                canvas,
                (255, 0, 0),
                pg.Rect(
                    tuple(pix_square_size * self.pred_locations[pred]), (pix_square_size,pix_square_size),
                ),
            )
            pg.draw.rect(
                canvas,
                (255, 130, 99),
                pg.Rect(
                    tuple(pix_square_size * (self.pred_locations[pred]+[1,0])), (pix_square_size,pix_square_size),
                ),
            )
            pg.draw.rect(
                canvas,
                (255, 130, 99),
                pg.Rect(
                    tuple(pix_square_size * (self.pred_locations[pred]+[-1,0])), (pix_square_size,pix_square_size),
                ),
            )
            pg.draw.rect(
                canvas,
                (255, 130, 99),
                pg.Rect(
                    tuple(pix_square_size * (self.pred_locations[pred]+[0,1])), (pix_square_size,pix_square_size),
                ),
            )
            pg.draw.rect(
                canvas,
                (255, 130, 99),
                pg.Rect(
                    tuple(pix_square_size * (self.pred_locations[pred]+[0,-1])), (pix_square_size,pix_square_size),
                ),
            )
        
        #draw berries
        for berry in range(self.num_berry):
            pg.draw.rect(
                canvas,
                (0, 255, 0),
                pg.Rect(
                    tuple(pix_square_size * self.berry_locations[berry]), (pix_square_size,pix_square_size),
                ),
            )

        #draw prey
        for prey in range(self.num_prey):
            if self.active_prey[prey] == True:
            #print(f"pred location * pix square size {pix_square_size*self.prey_locations[prey]}")
                if prey == 0:
                    pg.draw.rect(
                        canvas,
                        (170,170,255),
                        pg.Rect(
                            tuple(pix_square_size * self.prey_locations[prey]), (pix_square_size,pix_square_size)
                        ),
                    )
                else:
                    pg.draw.rect(
                        canvas,
                        (255,170,170),
                        pg.Rect(
                            tuple(pix_square_size * self.prey_locations[prey]), (pix_square_size,pix_square_size)
                        ),
                    )
                
            else:
                pass
        #add gridlines
        for x in range(self.grid_size[0] + 1):
            pg.draw.line(
                canvas,
                (63,63,63),
                (0, pix_square_size * x),
                (self.window_size[0], pix_square_size * x),
                width=3,
            )
            pg.draw.line(
                canvas,
                (63,63,63),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size[0]),
                width=3,
            )


        diffx = (self.window_size[0] - canvas.get_size()[0])/2
        diffy = (self.window_size[1] - canvas.get_size()[1])/2
            
        if self.render_mode == "human":
                self.window.blit(canvas, (diffx,diffy))
                pg.event.pump()
                pg.display.update()
                
                self.clock.tick(self.metadata["render_fps"])    
        else:
            return np.transpose(np.array(pg.surfarray.pixels3d(canvas)), axes=(1,0,2))
    
    def close(self):
        if self.window is not None:
            pg.display.quit()
            pg.quit()

    def key_presses(self, running):
        keys = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                    

        if keys[pg.K_ESCAPE]:
            running = False

        if keys[pg.K_UP]:
            if self.render_mode == None:
                print("we ON!")
                self.render_mode = "human"
                time.sleep(1)
                
        if keys[pg.K_DOWN]:
            if self.render_mode == "human":
                self.render_mode = None
                print("visualisation off")
                time.sleep(1)

        return running
    
    def resolve_conflicts(self, new_positions, agents_rewards):
        #print(f"proposed new positions: {new_positions}")
        unique_positions, indices, counts = np.unique(new_positions, axis=0, return_inverse=True, return_counts=True)
        
        
        conflicts = unique_positions[counts > 1]
        #print(conflicts)
        for conflict in conflicts:
            #print(f"self.preylocations: {self.prey_locations}")
            conflicting_agents = np.where((new_positions == conflict).all(axis=1))[0]
            #active_conflicting_agents = [agent for agent in conflicting_agents if self.active_prey[agent]]
            active_conflicting_agents = [agent for agent in conflicting_agents if self.active_prey[agent]]
            #identify agent with highest reward in current timestep among those in conflict
            #print(f"active conflicting agents: {active_conflicting_agents}")
            #print(f"len of active conflicting agents: {len(active_conflicting_agents)}")
            #print(f"conflcit: {self.conflict}")
            if len(active_conflicting_agents) > 1:
                self.conflict = True
                #print(f"conflcit: {self.conflict}")
                highest_reward = max(agents_rewards[agent] for agent in active_conflicting_agents)
                highest_reward_agents = [agent for agent in active_conflicting_agents if agents_rewards[agent] == highest_reward]
                chosen_agent = random.choice(highest_reward_agents)
                #print(f"chosen agent: {chosen_agent}")
                for agent in active_conflicting_agents:
                    #print(f"agent in active conflicting agents: {agent}")
                    if agent == chosen_agent:
                        if agent == 1:
                            #print(f"new positions and preylocations of 0 : {new_positions[0]}, {self.prey_locations[0]} and 1: {new_positions[1]}, {self.prey_locations[1]}")
                            if np.array_equal(new_positions[0], self.prey_locations[0]) and np.array_equal(new_positions[1], self.prey_locations[0]):
                                #agent gets the new location
                                #print("in1")
                                new_positions[1] = self.prey_locations[1]
                        elif agent == 0:
                            #print(f"new positions and preylocations of 0 : {new_positions[0]}, {self.prey_locations[0]} and 1: {new_positions[1]}, {self.prey_locations[1]}")
                            if np.array_equal(new_positions[1], self.prey_locations[1]) and np.array_equal(new_positions[0],self.prey_locations[1]):
                                #print("in2")
                                new_positions[0] = self.prey_locations[0]
                    elif agent != chosen_agent:
                        new_positions[agent] = self.prey_locations[agent]
        
                #print(f"RESOLVED POS: {new_positions}")
                #time.sleep(3)
              
        
        
        return new_positions
    




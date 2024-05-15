import numpy as np
import pygame as pg
import random
import time
import gymnasium as gym
from gymnasium import spaces

GRIDSIZE = 9

class PredPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode="human", grid_height=GRIDSIZE, grid_width=GRIDSIZE, num_prey=5, num_pred=1, num_berry=3 , num_channels=4):
        self.grid_size = (grid_width, grid_height)
        self.window_size = [730,560]
        self.num_prey = num_prey
        self.num_pred = num_pred
        self.num_berry = num_berry
        self.num_channels = num_channels
        self.num_channels = 3 + num_prey    #the three channels being all prey,pred,berry and others being self identifiers
        self.current_ep_total_reward = 0
        self.num_agents = self.num_berry+self.num_pred+self.num_prey
        self.current_turn_rewards = np.zeros(self.num_prey)
        self.action_space = spaces.Discrete(4)  # Example: 4 actions (up, down, left, right)
        self.berry_energy_amount = 10
        self.prey_energy = np.full(num_prey,self.berry_energy_amount)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_channels, self.grid_size[0], self.grid_size[1]), dtype=int)
        self.moved_last_turn = False
        self.moved_moved_last_two_turns = False
        self.turns_until_move = 5       #number of prey turns for 1 pred turn
        self.turns_until_move_counter = 0
        
        
        self.active_prey = np.ones(self.num_prey, dtype=bool)
        
        
        spawn_pool_pred = self.spawning()
        self.pred_locations = np.array([spawn_pool_pred.pop() for _ in range(self.num_pred)])
        spawn_pool_other = self.spawning_other(spawn_pool_pred)
        self.prey_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_prey)])
        self.berry_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_berry)])
        
        #print(f"berry locations: {self.berry_locations} and shape {self.berry_locations.shape}")
       
        self.active_prey_locations = self.prey_locations
        #print(self.prey_locations.shape)
                                                                              
        #print(f"prey_locations form: {self.observation_space}")
        self._action_to_direction = {
            0: np.array([0,1]),   #up
            1: np.array([1,0]),   #right
            2: np.array([0,-1]),  #down
            3: np.array([-1,0]),   #left
            -1: np.array([0,0])  #no change
        }
            
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def spawning_other(self, spawn_pool):
        excluded_positions = set()
        
        for pos in self.pred_locations:
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
        #print(f"available spawns: {random_location}")
        #print(f"berry locations before adding: {self.berry_locations}")
        #print(f"berry number: {berry_number}")
        self.berry_locations = np.delete(self.berry_locations, berry_number, axis=0)
        self.berry_locations = np.vstack([self.berry_locations, random_location])
        #print(f"berry locations after adding: {self.berry_locations}")
        #time.sleep(2)


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
    
    def _get_obs(self, prey_locations):      #adds all the shit to the grid
        grid = np.zeros((self.num_channels, self.grid_size[0], self.grid_size[1]), dtype=int)  # Assuming 1 channel; use more if needed
        
        #print(f"the grid in _get_obs(): {grid}")
        #print(f"the prey locations in _get_obs(): {prey_locations}")
        # self.prey_locations = self.prey_locations[0]          #making the list of arrays into just an array i.e. from [[x,y]] to [x,y]
        # self.pred_locations = self.pred_locations[0]
        # self.berry_locations = self.berry_locations[0]
        
        #print(len(self.prey_locations))
        #print(self.prey_location[1])
        #print(self.prey_location[0])
        for k, active in enumerate(self.active_prey):
            if active:  # This is equivalent to checking if active == True
                location = prey_locations[k]
                grid[0, location[0], location[1]] = 1  # Mark prey

        #print(f"grid 0: {grid[0]}")
        for location in self.pred_locations:
            # Get the coordinates of the predator
            pred_x, pred_y = location

            # Loop through all cells within one square of the predator
            for dx in [-1, 0, 1]:  # Offset in the x direction
                for dy in [-1, 0, 1]:  # Offset in the y direction
                    # Calculate the potentially dangerous cell's coordinates
                    zone_x = pred_x + dx
                    zone_y = pred_y + dy

                    # Check if the coordinates are within the grid bounds before marking
                    if 0 <= zone_x < grid.shape[1] and 0 <= zone_y < grid.shape[2]:
                        grid[1, zone_x, zone_y] = 1
                        # Mark predator
        for location in self.berry_locations:
            grid[2, location[0], location[1]] = 1  # Mark berry
        for i in range(self.num_prey):
            if self.active_prey[i] == True:
                grid[i+3, prey_locations[i][0], prey_locations[i][1]] = 1     # Mark self
        return grid
    
    def _get_info(self): # gets some kind of info i.e. could be distances from berry and pred etc.
        return {"prey": self.prey_locations.squeeze(), "pred": self.pred_locations.squeeze(), "berry": self.berry_locations.squeeze(), "alive_prey": self.active_prey}
    
    def get_inf(self): # gets some kind of info i.e. could be distances from berry and pred etc.
        return self.current_ep_total_reward
    
    def step(self, actions):
        #actions is a dict of the form: PreyIndexNumber: Action or False
        #need to check whether action is false or true 
        #print("-----------")
        #print(f"self.prey_locations: {self.prey_locations}")
        #print(f"self.pred_locations: {self.pred_locations}")
        assert len(actions) == self.num_prey, "each agent must have an action"
        
        #print(actions)
        self.scalar_actions = []
        episode_done = []
        
        for key in actions:
            #print(f"actions[key]: {actions[key]}")
            if actions[key] == -1:        #check for if the agent is dead
                self.scalar_actions.append(-1)           #assign -1 to action to represent dead
                self.active_prey[key] = False
            else:
                self.scalar_actions.append(actions[key].item())
        
        #print(f"active prey: {self.active_prey}")

        new_prey_locations = np.array(self.prey_locations, copy=True)
        for index, scalar_action in enumerate(self.scalar_actions):
            #print(f"index is : {index}")
            #print(f"action is : {self.scalar_actions}")
            if scalar_action != -1:        #dead check
                if scalar_action == 0 and self.prey_locations[index][1] < self.grid_size[1]-1:  # Move up
                    #print(new_prey_locations[index])
                    new_prey_locations[index][1] += 1
                    #print(new_prey_locations[index])
                elif scalar_action == 1 and self.prey_locations[index][0] < self.grid_size[0] - 1:  # Move right
                    #print(new_prey_locations[index])
                    new_prey_locations[index][0] += 1
                    #print(new_prey_locations[index])
                elif scalar_action == 2 and self.prey_locations[index][1] > 0:  # Move down
                   # print(new_prey_locations[index])
                    new_prey_locations[index][1] -= 1
                   # print(new_prey_locations[index])
                elif scalar_action == 3 and self.prey_locations[index][0] > 0:  # Move right
                    #print(new_prey_locations[index])
                    new_prey_locations[index][0] -= 1
                    #print(new_prey_locations[index])
                else:
                    new_prey_locations[index] = self.prey_locations[index] #not dead but on border
            elif self.scalar_actions[index] == -1:      #dead check
                #print("dead index shit working")
                new_prey_locations[index] = self.prey_locations[index]
        
        #test_prey_locations = np.array([np.array([3,3]),np.array([3,3]), np.array([3,3])])
        #print(f"test prey locations: {new_prey_locations}")

        #print(f"self.current turn rewards: {self.current_turn_rewards}")
        new_prey_locations_resolved = self.resolve_conflicts(new_prey_locations, self.current_turn_rewards)

        for i in range(self.num_prey):
            if self.active_prey[i] == True:
                self.active_prey_locations

        #MOVE PREDATORS
        if self.turns_until_move_counter == self.turns_until_move:
            self.move_pred()
            self.turns_until_move_counter = 0

        #print(self.pred_locations)

        #ASSIGNING REWARDS
        self.current_turn_rewards = np.zeros(self.num_prey)
        for i in range(self.num_prey):
            if self.active_prey[i] == True:
                #print(f"the prey agent is :{i}")
                for j in range(self.num_berry):
                    if np.array_equal(new_prey_locations_resolved[i], self.berry_locations[j]):
                        self.current_turn_rewards[i] = 5
                        self.prey_energy[i] = self.prey_energy[i]+self.berry_energy_amount
                        #remove berry and spawn another one
                        #print(f"self.berrylocations: {self.berry_locations}")
                        self.spawn_berry(new_prey_locations_resolved, j)
                        #print(f"new berrylocations: {self.berry_locations}")

                #DEATH
                for j in range(self.num_pred):
                    pred_x, pred_y = self.pred_locations[j]

                    # Loop through all cells within one square of the predator
                    for dx in [-1, 0, 1]:  # Offset in the x direction
                        for dy in [-1, 0, 1]:  # Offset in the y direction
                            # Calculate the potentially dangerous cell's coordinates
                            danger_zone_x = pred_x + dx
                            danger_zone_y = pred_y + dy

                            # Check if the prey is in this cell
                            if np.array_equal(new_prey_locations_resolved[i], [danger_zone_x, danger_zone_y]):
                                self.current_turn_rewards[i] = -20
                                self.active_prey[i] = False
                                break  # No need to check further cells if prey is already caught

                    # If prey has been caught, no need to check against other predators
                    if not self.active_prey[i]:
                        break
                #print(f"prey locations i: {self.prey_locations[i]}, newpreylocations: {new_prey_locations_resolved[i]}")
                if self.current_turn_rewards[i] == 0:
                    if False in self.active_prey:   #negative reward every turn if someone dies
                        self.current_turn_rewards[i] = 0
                    if np.array_equal(self.prey_locations[i], new_prey_locations_resolved[i]) and self.active_prey[i] != False:
                        self.current_turn_rewards[i] = -3
                    elif self.current_turn_rewards[i] == 0:
                        self.current_turn_rewards[i] = 0.3
                #print(f"the reward for agent {i} is {self.current_turn_rewards[i]}")
            
        #print(f"active prey: {self.active_prey}")
        #print(f"current turn rewards: {current_turn_rewards}")
        #print(f"self.current turn rewards after rewards awarded: {self.current_turn_rewards}")
        self.prey_locations = new_prey_locations_resolved #setting old positions to new positions

        episode_done = not any(self.active_prey)    #finish ep if no prey agents alive

        observation = self._get_obs(new_prey_locations_resolved)
        #print(observation)
        
        info = self._get_info()
        
        if self.render_mode == "human":
            pass
            #self.render_frame()
        
        self.turns_until_move_counter+=1
        #self.current_ep_total_reward = reward + self.current_ep_total_reward
        #print(f"self.current turn rewards: {self.current_turn_rewards}")
        return observation, self.current_turn_rewards, self.active_prey, episode_done, info    

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        # and return the initial observation
        #print("------RESET CALLED------")
        #following line to seed np.random
        super().reset(seed=seed)
        self.active_prey = np.ones(self.num_prey, dtype=bool)
        
        self.turns_until_move_counter = 0
        
        spawn_pool_pred = self.spawning()
        self.pred_locations = np.array([spawn_pool_pred.pop() for _ in range(self.num_pred)])
        spawn_pool_other = self.spawning_other(spawn_pool_pred)
        self.prey_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_prey)])
        self.berry_locations = np.array([spawn_pool_other.pop() for _ in range(self.num_berry)])
        


        #print(f"the prey locations in reset: {self.prey_locations}")
            
        #self.pred_locations = np.array([[0,0]])
        self.current_ep_total_reward = 0
        self.active_prey_locations = self.prey_locations
        observation = self._get_obs(self.prey_locations)
        #print(f"observation in reset: {observation}")
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
        # Initialize Pygame and the display window
            pg.init()
            pg.display.init()
            pg.display.set_caption('Predator, Prey and Berries')
            self.window = pg.display.set_mode(
                (self.window_size)
            )
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()
        
        #print(f"self.prey_locations_tuple in render frame: {self.prey_locations_tuple}")
        
        self.window.fill((43,43,43))
        canvas = pg.Surface((500, 500))
        canvas.fill((54, 54, 54))
        pix_square_size = (
            canvas.get_size()[0] / self.grid_size[0]
        )  # The size of a single grid square in pixels

        pix_square_size = int(pix_square_size)
        
        
        # Next we draw pred
        for pred in range(self.num_pred):
            pg.draw.rect(
                canvas,
                (255, 0, 0),
                pg.Rect(
                    tuple(pix_square_size * self.pred_locations[pred]), (pix_square_size,pix_square_size),
                ),
            )
        
        # Next we draw berries
        for berry in range(self.num_berry):
            pg.draw.rect(
                canvas,
                (0, 255, 0),
                pg.Rect(
                    tuple(pix_square_size * self.berry_locations[berry]), (pix_square_size,pix_square_size),
                ),
            )

        # draw prey
        for prey in range(self.num_prey):
            if self.active_prey[prey] == True:
            #print(f"pred location * pix square size {pix_square_size*self.prey_locations[prey]}")
                pg.draw.rect(
                    canvas,
                    (0, 0, 255),
                    pg.Rect(
                        tuple(pix_square_size * self.prey_locations[prey]), (pix_square_size,pix_square_size)
                    ),
                )
            else:
                pass
        # Finally, add some gridlines
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
        #print(keys)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                    

        if keys[pg.K_ESCAPE]:
            #print(f"yo")
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
        #print(f"new pos: {new_positions}")
        #time.sleep(1)
        unique_positions, indices, counts = np.unique(new_positions, axis=0, return_inverse=True, return_counts=True)
        conflicts = unique_positions[counts > 1]
       # print(f"conflict: {conflicts}")
        #print(f"unique_positions: {unique_positions}")
        for conflict in conflicts:
            conflicting_agents = np.where((new_positions == conflict).all(axis=1))[0]
            #print(f"conflicting agents: {conflicting_agents}")
            #print(f"active preys: {self.active_prey}")
            #active_conflicting_agents = [agent for agent in conflicting_agents if self.active_prey[agent]]
            active_conflicting_agents = [agent for agent in conflicting_agents if self.active_prey[agent]]
            
            #print(f"active conflicting agents: {active_conflicting_agents}")
            # Identify the agent with the highest reward in the current timestep among those in conflict
            if active_conflicting_agents:
                highest_reward = max(agents_rewards[agent] for agent in active_conflicting_agents)
                highest_reward_agents = [agent for agent in active_conflicting_agents if agents_rewards[agent] == highest_reward]
                chosen_agent = random.choice(highest_reward_agents)
                # Allow only the agent with the highest recent reward to move to the conflict position
                # print(f"highest reward agent: {highest_reward}")
                # print(f"chosen agent: {chosen_agent}")
                # print(f"highest reward agents: {highest_reward_agents}")
                # print(f"prey locations: {self.prey_locations}")
                for agent in active_conflicting_agents:
                    if agent != chosen_agent:
                        new_positions[agent] = self.prey_locations[agent]
                #         print(f"prey locations: {self.prey_locations}")
                #         print(f"prey locations[agent]: {self.prey_locations[agent]}")
                #         print(f"new positions: {new_positions[agent]}")
                # print(f"new positions: {new_positions}")
        #print(f"resolved pos: {new_positions}")
        return new_positions
    
    def move_pred(self):
        # Calculate intended moves first
        intended_positions = [list(pos) for pos in self.pred_locations]  # Make a copy of current positions

        # Move each predator towards the closest prey
        for i in range(self.num_pred):
            closest_prey = None
            min_dist = float('inf')

            # Find the closest prey to predator i
            for j in range(self.num_prey):
                if self.active_prey[j] == True:
                    x_diff = abs(self.pred_locations[i][0] - self.prey_locations[j][0])
                    y_diff = abs(self.pred_locations[i][1] - self.prey_locations[j][1])
                    distance = x_diff + y_diff  # Manhattan distance
                    if distance < min_dist:
                        min_dist = distance
                        closest_prey = j

            # Determine the movement based on closest prey
            if closest_prey is not None:
                x_diff = self.pred_locations[i][0] - self.prey_locations[closest_prey][0]
                y_diff = self.pred_locations[i][1] - self.prey_locations[closest_prey][1]

                # Move in the direction that minimizes the largest difference
                if abs(x_diff) > abs(y_diff):
                    intended_positions[i][0] += -1 if x_diff > 0 else 1
                else:
                    intended_positions[i][1] += -1 if y_diff > 0 else 1

        # Resolve conflicts and update actual positions
        occupied_positions = set()
        for idx, pos in enumerate(intended_positions):
            if tuple(pos) not in occupied_positions:
                self.pred_locations[idx] = pos
                occupied_positions.add(tuple(pos))
            else:
                # If position is already taken, do not move predator
                pass
                print(f"Conflict resolved for predator {idx}, stays at {self.pred_locations[idx]}")

        self.moved_last_turn = True





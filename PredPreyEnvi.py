import numpy as np
import pygame as pg
import random

import gymnasium as gym
from gymnasium import spaces


class PredPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode="human", grid_size=5, num_prey=1, num_pred=1, num_berry=1):
        self.grid_size = grid_size
        self.window_size = [480,360]
        self.num_prey = num_prey
        self.num_pred = num_pred
        self.num_berry = num_berry
        
        self.current_ep_total_reward = 0
        
        self.num_agents = self.num_berry+self.num_pred+self.num_prey
        #super(PredatorPreyEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        self.action_space = spaces.Discrete(4)  # Example: 4 actions (up, down, left, right)
        
        # self.action_space = spaces.dict.Dict(
        #     {
        #         "prey": spaces.Discrete(4)
        #     }
        # )
        
        spawn_points = self.spawning()
        random.shuffle(spawn_points)
        
        self.prey_location = [np.array(spawn_points.pop() for _ in range(self.num_prey))]     # .pop() removes last element in list and returns it
        self.pred_location = [np.array(spawn_points.pop() for _ in range(self.num_pred))]
        self.berry_location = [np.array(spawn_points.pop() for _ in range(self.num_berry))]

        #6 elements to represent preyx, preyy, berryx, berryy, predx, predy
        self.xy_low = np.array([0,0])
        self.xy_high = np.array([self.grid_size-1,self.grid_size-1])
        
        self.observation_space = spaces.Dict(
            {
                "prey": spaces.Box(low=np.array(self.xy_low), high=np.array(self.xy_high), dtype=int),
                "pred": spaces.Box(low=np.array(self.xy_low), high=np.array(self.xy_high), dtype=int),
                "berry": spaces.Box(low=np.array(self.xy_low), high=np.array(self.xy_high), dtype=int)
            }
        )
        
        self._action_to_direction = {
            0: np.array([0,1]),   #up
            1: np.array([1,0]),   #right
            2: np.array([0,-1]),  #down
            3: np.array([-1,0])   #left
        }
            
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def spawning(self):
        spawn_pool = [(x,y) for x in range(self.grid_size) for y in range(self.grid_size)]
        num_agents = self.num_agents
        return random.sample(spawn_pool, num_agents)
        
    
    def _get_obs(self):
        return {"prey": self.prey_location.squeeze(), "pred": self.pred_location.squeeze(), "berry": self.berry_location.squeeze()}
    
    def _get_info(self): # gets some kind of info i.e. could be distances from berry and pred etc.
        return {"prey": self.prey_location.squeeze(), "pred": self.pred_location.squeeze(), "berry": self.berry_location.squeeze()}
    
    def get_inf(self): # gets some kind of info i.e. could be distances from berry and pred etc.
        return self.current_ep_total_reward
    
    def step(self, action):
        # Implement the logic to take one step in the environment
        # and return observation, reward, done, info
        direction = self._action_to_direction[action]
        #print(f"direction: {direction}")
        #np.clip() to ensure agent stays on grid
        self.prey_location = np.clip(
            self.prey_location + direction , 0, self.grid_size-1
        )
        
        #episode finishes if and only if the prey reaches the berry
        episode_done_positive = np.array_equal(self.prey_location, self.berry_location)
        episode_done_negative = np.array_equal(self.prey_location, self.pred_location)
        
        rewards = {
            "positive": 5,     #agent gets berry
            "negative": -5,    #agent gets caught by pred
        }
        
        if episode_done_positive:
            condition = "positive"
        elif episode_done_negative:
            condition = "negative"
        else:
            condition = "neutral"
            
        reward = rewards.get(condition, -0.1) # default to -0.1 if condition not found
        
        episode_done = episode_done_negative or episode_done_positive
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            pass
            #self.render_frame()
        
        self.current_ep_total_reward = reward + self.current_ep_total_reward
        
        return observation, reward, episode_done, False, info    

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        # and return the initial observation
        #print("------RESET CALLED------")
        #following line to seed np.random
        super().reset(seed=seed)
        
        spawn_points = self.spawning()
        random.shuffle(spawn_points)
        
        # self.prey_location = np.array([spawn_points.pop() for _ in range(self.num_prey)])
        self.pred_location = np.array([spawn_points.pop() for _ in range(self.num_pred)])
        self.berry_location = np.array([spawn_points.pop() for _ in range(self.num_berry)])
        
        self.prey_location = np.array([4,4])
        #self.pred_location = np.array([4,1])
        #self.berry_location = np.array([4,0])
            
        self.current_ep_total_reward = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render_frame()
        
        return observation, info 

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
                   
    def render_frame(self):
        if self.window is None and self.render_mode == "human":
        # Initialize Pygame and the display window
            pg.init()
            pg.display.init()
            pg.display.set_caption('Predator, Prey and Berries')
            self.window = pg.display.set_mode(
                (self.window_size)
            )
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()
            
        self.prey_location = self.prey_location.flatten() #converting the ndarray into a tuple to make computation easier
        self.pred_location = self.pred_location.flatten()
        self.berry_location = self.berry_location.flatten()
        
        self.window.fill((43,43,43))
        canvas = pg.Surface((300, 300))
        canvas.fill((54, 54, 54))
        pix_square_size = (
            canvas.get_size()[0] / self.grid_size
        )  # The size of a single grid square in pixels

        pix_square_size = int(pix_square_size)
        
        # print(f"pix square size: {int(pix_square_size)}")
        # print(f"mult: { pix_square_size * self.prey_location}")
        # print(f"type of prey location: {type(self.prey_location)}")
        # print(f"prey location as tuple: {tuple(self.prey_location)}")
        
        
        # Next we draw pred
        pg.draw.rect(
            canvas,
            (255, 0, 0),
            pg.Rect(
                tuple(pix_square_size * self.pred_location), (pix_square_size,pix_square_size),
            ),
        )
        
        # Next we draw berries
        pg.draw.rect(
            canvas,
            (0, 255, 0),
            pg.Rect(
                tuple(pix_square_size * self.berry_location), (pix_square_size,pix_square_size),
            ),
        )

        # draw prey
        pg.draw.rect(
            canvas,
            (0, 0, 255),
            pg.Rect(
                tuple(pix_square_size * self.prey_location), (pix_square_size,pix_square_size)
            ),
        )
        
        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
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
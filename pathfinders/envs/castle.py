import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import pygame
import os

class CastleEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4  # Ensure this is set if your environment should dictate the fps
    }
    def __init__(self, render_mode=None):
        
        self.window_size = (1200, 600) 
  
        self.MAP = 1 - (np.genfromtxt(os.path.join('pathfinders','envs','castle.txt'),float)-1.0)
        self.height = self.MAP.shape[0]  # The height of the grid
        self.width = self.MAP.shape[1]  # The width of the grid
        self.graphics = np.expand_dims(self.MAP, axis = 2)
        self.graphics = np.repeat(self.graphics, 4, axis=2)
        self.graphics*=255
        self._cummlative_reward = 0 

        self._step = 0
        self.graphics[:,:,3] = 1-self.graphics[:,:,3]

        self.start_position = np.array([8, 1])
        

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Dict(
                        {"pos": spaces.Box(
                            low=np.array([0, 0]), 
                            high=np.array([self.width - 1, self.height - 1]), 
                            dtype=np.int32
                        )
                          }
                ) 
            }
        )

        # We have 3 actions, corresponding to "leftforward", "straigth", "left", "rightforward"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_to_direction = {
            0: np.array([-1, 0]), #up
            1: np.array([1, 0]), # down
            2: np.array([0, -1]), #left
            3: np.array([0, 1]),  #right
        }
       

        

        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    

    def _get_obs(self):
        return {"agent": {"pos": self._agent_location}}


    def _get_info(self):
        return {"distance": abs(self._agent_location[1] - self._target_location[1]), "steps": self._step}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = self.start_position
        self._step = 0
        self._cummlative_reward = 0

        self._target_location = np.array([1,8])
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    
    def _transition(self, action):
        r = np.random.uniform()
        if action in [0,3,6]: #backwards, 100%
            return self.action_to_direction[action]
        if  action in [1,4]: #sideways
            if r<1/16:
                return self.action_to_direction[action-1] # sideways backward
            elif r<1/8:
                return self.action_to_direction[action+1] # sideways forward
            else:
                return self.action_to_direction[action] # sideways 

        if  action in [3,5]: #forward sideways
            if r<1/16:
                return self.action_to_direction[action-1] # sideways 
            elif r<1/8:
                return self.action_to_direction[7] # forward
            else:
                return self.action_to_direction[action] # sideways forward
            
        return self.action_to_direction[action]


    
    
    def step(self, action):

        direction = self._transition(action)
        
        # We use `np.clip` to make sure we don't leave the grid
        next_pos = self._agent_location + direction

        # next_pos[0] = np.clip(self._agent_location[0] + direction[0],a_min=1, a_max=8)
        # next_pos[1] = np.clip(self._agent_location[1] + direction[1],a_min=1, a_max=8)
        

    

        self._step += 1    
        reward = -1
        if self.MAP[next_pos[0],next_pos[1]]!=0:
            self._agent_location = next_pos

        
     
        
        
   
        # An episode is done if the agent has reached the target
        reached_goal= np.array_equal(self._agent_location, self._target_location)
        terminated = reached_goal
        truncated =  self._step>=1000

        if reached_goal:
            reward =0
        self._cummlative_reward +=reward
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def render(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size[1] / self.height
        
    

        
        for row in range(self.height):
            for col in range(self.width):
                color = self.graphics[row, col]
                pygame.draw.rect(
                    canvas,
                    color[:3],  # Use the first three values (RGB) from the color
                    (col * pix_square_size, row* pix_square_size, pix_square_size, pix_square_size),
                )


        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Use the first three values (RGB) from the color
            (self._target_location[1] * pix_square_size, self._target_location[0]* pix_square_size, pix_square_size, pix_square_size),
        )
        
        
        #Now we draw the agent
        agent_center = (
            int((self._agent_location[1] + 0.5) * pix_square_size),
            int((self._agent_location[0] + 0.5) * pix_square_size)
        )
        pygame.draw.circle(canvas, (151,87,43), agent_center, pix_square_size // 3)

        pygame.font.init()
        font = pygame.font.Font(None, int(pix_square_size / 2))
        
        text_y_position = self.window_size[1] // 3 - pix_square_size

        score_text = font.render(f'Steps: {self._step}',True, (0, 0, 0))
        pos_text = font.render(f'position: ({self._agent_location[0]}, {self._agent_location[1]})',True, (0, 0, 0))
        cum_reward_text = font.render(f'reward: {self._cummlative_reward:.2f}',True, (0, 0, 0))

        canvas.blit(score_text, (self.window_size[1]+pix_square_size , text_y_position))
        canvas.blit(pos_text, (self.window_size[1]+pix_square_size, text_y_position + 20))
        canvas.blit(cum_reward_text, (self.window_size[1]+pix_square_size, text_y_position + 40))


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
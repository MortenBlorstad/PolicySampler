import gymnasium as gym
import pathfinders
from agents.agent import PolicySamplerAgent
import numpy as np
import matplotlib.pyplot as plt 

# Create the environment
env = gym.make('castle-v0', render_mode = "rgb_array")

agent = PolicySamplerAgent(env,9,9, 4,1, 1)

agent.sampler(0,1000)



env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="castle_sampler")
observation,info= env.reset()

done = False
while not done:
    state =  tuple(observation["agent"]["pos"])
    action =agent.select_action(state,agent.pi)
    observation, reward, done,_, info = env.step(action)
    
    if done:
        break
    next_state =  tuple(observation["agent"]["pos"])
    state = next_state

# Close the environment
env.close()  # This will save the video correctly
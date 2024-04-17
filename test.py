import gymnasium as gym
import pathfinders
from agents.agent import SarsaAgent,MCAgent
import numpy as np
import matplotlib.pyplot as plt 

# Create the environment
env = gym.make('castle-v0', render_mode = "rgb_array")

agent = SarsaAgent(9,9, 4,0.1, 0.2, 1)
# Reset the environment


visited = np.zeros((9,9))
avg_steps = 0
n_episodes = 10000
for episode in range(n_episodes):
    trajectory = []
    done = False
    observation,info= env.reset()
    while not done:
        state = tuple(observation["agent"]["pos"])
        visited[state]+=1
        action = agent.select_action(state)
        observation, reward, done,_, info = env.step(action)
        if done:
            avg_steps+=info["steps"]
        next_state = tuple(observation["agent"]["pos"])
        agent.learn(state,action,reward,next_state)
        trajectory.append( (state,action,reward,next_state))
        state = next_state
    
    agent.epsilon*=0.999
    print(avg_steps/(episode+1))
    #print("updating Q_values")
    #agent.learn(trajectory)
    
#plt.imshow(agent.Q.max(axis=2))
 
plt.imshow(np.log(visited+0.001))
plt.show()


env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="castle")
observation,info= env.reset()

done = False
while not done:
    state =  tuple(observation["agent"]["pos"])
    action =agent.select_action(state)
    observation, reward, done,_, info = env.step(action)
    
    if done:
        break
    next_state =  tuple(observation["agent"]["pos"])
    state = next_state

# Close the environment
env.close()  # This will save the video correctly
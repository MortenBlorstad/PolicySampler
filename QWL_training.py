import gymnasium as gym
import pathfinders
from agents.QWL import QWLAgent
import numpy as np
import matplotlib.pyplot as plt 

# Create the environment
env = gym.make('castle-v0', render_mode = "rgb_array")

agent = QWLAgent(9,9, 4, 1,0.2,1,20)
# Reset the environment


visited = np.zeros((9,9))
avg_steps = 0
n_episodes = 2000
for episode in range(n_episodes):
    trajectory = []
    done = False
    observation,info= env.reset()
    while not done:
        state = tuple(observation["agent"]["pos"])
        visited[state]+=1
        action = agent.sample_action(state)
        observation, reward, done,_, info = env.step(action)
        if done:
            avg_steps+=info["steps"]
        next_state = tuple(observation["agent"]["pos"])
        agent.learn(state,action,reward,next_state)
        trajectory.append( (state,action,reward,next_state))
        state = next_state
    
    print(avg_steps/(episode+1))
    #print("updating Q_values")
    #agent.learn(trajectory)
    
#plt.imshow(agent.Q.max(axis=2))
fig, axs = plt.subplots(1,2, figsize = (18,10))
img =axs[0].imshow(np.log(visited[1:,1:]+0.001))
axs[0].set_title(f"Number of state visit in $\ln$", fontsize = 14)
fig.colorbar(img, ax=axs[0], orientation='vertical')

axs[1].bar(range(len(agent.H)), agent.H,width=agent.binsize)
axs[1].set_title(f"Histogram of visited Q-values", fontsize = 14)
axs[1].set_xlabel(r"binned $Q(a|s)$", fontsize = 12)
axs[1].set_ylabel("frequency", fontsize = 12)
plt.tight_layout()
plt.savefig("plots/QWL_plots.png")
plt.show()

env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="castle_QWL")
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
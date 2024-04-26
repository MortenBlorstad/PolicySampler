import gymnasium as gym
import pathfinders
from agents.QWL import QWLAgent
from agents.agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt 

# Create the environment
env = gym.make('shortcutmaze-v0', render_mode = "rgb_array")



agent = QWLAgent(32,32, 4, 1,0.2,1,200)
#agent = SarsaAgent(32,32, 4, 0.1,0.2,1)
# Reset the environment


visited = np.zeros((32,32))

avg_steps = 0
n_episodes = 10000
for episode in range(n_episodes):
    trajectory = []
    done = False
    observation,info= env.reset()
    while not done:
        state = tuple(observation["agent"]["pos"])
        visited[state]+=1
        #action = agent.select_action(state)
        action = agent.sample_action(state)
        observation, reward, done,_, info = env.step(action)
        if done:
            avg_steps+=info["steps"]
        next_state = tuple(observation["agent"]["pos"])
        agent.learn(state,action,reward,next_state)
        trajectory.append( (state,action,reward,next_state))
        state = next_state
    if (episode+1) %50 ==0:
        print(f"episode {episode+1}: average trajectory length {avg_steps/(episode+1):.2f}")

    

fig, axs = plt.subplots(1,2, figsize = (18,10))
img =axs[0].imshow(np.log(visited+0.001))
axs[0].set_title(r"Number of state visit in $\ln$", fontsize = 14)
fig.colorbar(img, ax=axs[0], orientation='vertical')

axs[1].bar(range(len(agent.H)), agent.H,width=agent.binsize)
axs[1].set_title(f"Histogram of visited Q-values", fontsize = 14)
axs[1].set_xlabel(r"binned $Q(a|s)$", fontsize = 12)
axs[1].set_ylabel("frequency", fontsize = 12)
plt.tight_layout()
plt.savefig("plots/QWL_shortcut_plots.png")
#plt.savefig("plots/sarsa_shortcut_plots.png")
plt.show()

observation,info= env.reset()
env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="shortcutmaze_QWL")


done = False
while not done:
    state =  tuple(observation["agent"]["pos"])
    action =agent.select_action(state)
    observation, reward, terminated,truncated , info = env.step(action)
    done = terminated or truncated
    if done:
        break
    next_state =  tuple(observation["agent"]["pos"])
    state = next_state

# Close the environment
env.close() 
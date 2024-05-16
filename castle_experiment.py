import gymnasium as gym
import pathfinders
from agents.QWL import QWLAgent
from agents.agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize




agents = {"QWL": QWLAgent(9,9, 4, 1,0.2,1,20),
          "SARSA": SarsaAgent(9,9, 4,0.1, 0.2, 1) }


for agent_name, agent in agents.items():
    # Create the environment
    env = gym.make('castle-v0', render_mode = "rgb_array")

    # Reset the environment

    visited = np.zeros((10,10))
    avg_steps = 0
    n_episodes = 2000
    for episode in range(n_episodes):
        trajectory = []
        done = False
        observation,info= env.reset()
        while not done:
            state = tuple(observation["agent"]["pos"])
            visited[state]+=1
            action = agent.select_behaviour_action(state)
            observation, reward, done,_, info = env.step(action)
            if done:
                avg_steps+=info["steps"]
            next_state = tuple(observation["agent"]["pos"])
            agent.learn(state,action,reward,next_state)
            trajectory.append( (state,action,reward,next_state))
            state = next_state
        
        if episode % 500 ==0:
            print(f"{episode}: {avg_steps/(episode+1):.2f}")
        
    mask = visited==0
    visited[mask] = np.inf
    visited = np.log(visited)
    masked_visited = np.ma.masked_invalid(visited)

    # Get the minimum and maximum values of the valid data
    if agent_name =="QWL":
        vmin, vmax = masked_visited.min(), masked_visited.max()
    
    cmap = plt.cm.YlOrRd   # Choose any colormap you prefer
    cmap.set_bad(color='black')
    norm = Normalize(vmin=0, vmax=vmax)


    fig, axs = plt.subplots(1,2, figsize = (18,10))
    img =axs[0].imshow(masked_visited, cmap=cmap, norm=norm)
    axs[0].set_title(f"Number of state visit in $\ln$", fontsize = 14)
    fig.colorbar(img, ax=axs[0], orientation='vertical')
    
    last_nonzero = np.nonzero(agent.H)[0][-1]+1

    axs[1].bar(range(len(agent.H[:last_nonzero])), agent.H[:last_nonzero],width=agent.binsize)
    axs[1].set_title(f"Histogram of visited Q-values", fontsize = 14)
    axs[1].set_xlabel(r"binned $Q(a|s)$", fontsize = 12)
    axs[1].set_ylabel("frequency", fontsize = 12)
    plt.tight_layout()
    plt.savefig(f"plots/{agent_name}_plots.png")
    plt.show()

    env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=f"castle_{agent_name}")
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
import gymnasium as gym
import pathfinders
from agents.QWL import QWLAgent
from agents.agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import imageio



agents = {"QWL": QWLAgent(32,32, 4, 1,0.2,1,200),
          "SARSA": SarsaAgent(32,32, 4,0.1, 0.2, 1)
           }

returns = {agent_name : [] for agent_name in agents.keys()}

evaluate_policy_after = 4000
evaluate_policy_every = 5
experiment_repeats = 10

for agent_name, agent in agents.items():
    # Create the environment
    env = gym.make('shortcutmaze-v0', render_mode = "rgb_array")
    visited = np.zeros((32,32))

    avg_steps = 0
    n_episodes = 10000
    for episode in range(n_episodes):
        trajectory = []
        done = False
        observation,info= env.reset()
        steps = 0
        if (episode+1) >= n_episodes//2:
            env.open_shortcut() 
        while not done:
            state = tuple(observation["agent"]["pos"])
            visited[state]+=1
            #action = agent.select_action(state)
            action = agent.select_behaviour_action(state)
            observation, reward, done,_, info = env.step(action)
            if done:
                avg_steps+=info["steps"]
            next_state = tuple(observation["agent"]["pos"])
            agent.learn(state,action,reward,next_state)
            trajectory.append( (state,action,reward,next_state))
            state = next_state
        if (episode+1) %50 ==0:
            print(f"episode {episode+1}: trajectory length {info["steps"]:.2f} - average {avg_steps/(episode+1):.2f} ")
        
        if (episode+1) %evaluate_policy_every  ==0 and (episode+1)>=evaluate_policy_after:
            G = 0
            for i in range(experiment_repeats):
                done = False
                observation,info= env.reset()
                
                
                while not done:
                    state = tuple(observation["agent"]["pos"])
                    action = agent.select_action(state)
                    observation, reward, done,_, info = env.step(action)
                    G +=reward
                    
                    next_state = tuple(observation["agent"]["pos"])
                    state = next_state
            returns[agent_name].append(G/experiment_repeats)

        
    
    mask = visited==0
    visited[mask] = np.inf
    visited = np.log(visited)
    masked_visited = np.ma.masked_invalid(visited)

    # # Get the minimum and maximum values of the valid data
    # if agent_name =="QWL":
    #     vmin, vmax = masked_visited.min(), masked_visited.max()
    
    # cmap = plt.cm.YlOrRd   # Choose any colormap you prefer
    # cmap.set_bad(color='black')
    # norm = Normalize(vmin=0, vmax=vmax)


    # fig, axs = plt.subplots(1,2, figsize = (18,10))
    # img =axs[0].imshow(masked_visited, cmap=cmap, norm=norm)
    # axs[0].set_title(r"Number of state visit in $\ln$", fontsize = 14)
    # fig.colorbar(img, ax=axs[0], orientation='vertical')
    
    # last_nonzero = np.nonzero(agent.H)[0][-1]+1

    # axs[1].bar(range(len(agent.H[:last_nonzero])), agent.H[:last_nonzero],width=agent.binsize)
    # axs[1].set_title(f"Histogram of visited Q-values", fontsize = 14)
    # axs[1].set_xlabel(r"binned $Q(a|s)$", fontsize = 12)
    # axs[1].set_ylabel("frequency", fontsize = 12)
    # plt.tight_layout()
    # plt.savefig(f"plots/{agent_name}_shortcut_plots.png")
    # plt.show()

    observation,info= env.reset()
    env.open_shortcut()

    writer = imageio.get_writer(f'videos/{agent_name}_shortcut.mp4', fps=5)

    done = False
 
    while not done:
        state =  tuple(observation["agent"]["pos"])
        action = agent.select_action(state)
        observation, reward, terminated,truncated , info = env.step(action)
        done = terminated or truncated
        frame = env.render()
        writer.append_data(frame)
        if done:
            break
        next_state =  tuple(observation["agent"]["pos"])
        state = next_state

        


    # Close the environment
    writer.close()
    env.close() 

for agent_name in agents.keys():
    plt.plot(range(evaluate_policy_after,evaluate_policy_after + evaluate_policy_every*(len(returns[agent_name])),evaluate_policy_every), returns[agent_name],'--o', label = agent_name)
plt.axvline(x = n_episodes//2, color = 'black', linestyle = ":")
plt.ylabel("Return G")
plt.xlabel("Episode number")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
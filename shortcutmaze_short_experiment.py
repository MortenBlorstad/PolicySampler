import gymnasium as gym
import pathfinders
from agents.QWL import QWLAgent
from agents.agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import imageio



def moving_average(x, window_size, axis=-1):
    """
    Compute the moving average of a 2D array over a specified axis.

    Parameters:
    x (numpy.ndarray): Input array.
    window_size (int): The size of the moving average window.
    axis (int): The axis along which to compute the moving average.

    Returns:
    numpy.ndarray: The array of moving averages.
    """
    if x.ndim != 2:
        return np.convolve(x, np.ones(window_size), 'valid') / window_size
    
    # Create sliding windows
    shape = list(x.shape)
    shape[axis] -= window_size - 1
    if shape[axis] < 1:
        raise ValueError("window_size is too large for the specified axis.")
    
    # Generate the sliding window view
    x_strided = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_size, axis=axis)
    
    # Compute the mean across the windows
    moving_avgs = x_strided.mean(axis=-1)

    return moving_avgs


agents = {"QWL": QWLAgent(15,18, 4, 0.1 ,0.2,1,1000),
          "SARSA": SarsaAgent(15,18, 4,0.1, 0.2, 1)
           }



np.random.seed(0)
n_episodes = 5000
evaluate_policy_after = n_episodes
evaluate_policy_every = 5
experiment_repeats = 30

returns = {agent_name : [] for agent_name in agents.keys()}

eps_checks = [3332, 3500,4250,5000]

Q_avgs =  {agent_name : {eps_number : np.zeros((15,18)) for eps_number in eps_checks} for agent_name in agents.keys()}


shortcut_found = {agent_name : np.zeros((n_episodes,experiment_repeats)) for agent_name in agents.keys()}


optimal_lenght_shorcut_closed = 30

open_shorcut_factor = 1.5


# cmap = plt.cm.viridis 
# cmap.set_bad(color='black')

# fig, axs = plt.subplots(nrows=2,ncols=len(eps_checks),figsize=(16, 8))
# for row,agent_name in enumerate(Q_avgs.keys()):
#     print(agent_name)
#     fig.text(0.025,(2 * (1-row) +1) / 4, agent_name, ha='center', va='center', fontsize=16)
#     for col,(eps_num, Q_avg) in enumerate(Q_avgs[agent_name].items()):
#         if row ==0:
#             fig.text((2 * col +1) / (2*4), 0.95, f"Episode {eps_num}", ha='center', va='center', fontsize=14)
#         Q_avg/=experiment_repeats
#         mask = Q_avg == 0
#         Q_avg[mask] = np.inf
#         masked_Q_avg = np.ma.masked_invalid(Q_avg)
#         img =axs[row,col].imshow(masked_Q_avg,cmap = cmap)
#         #axs[row,col].set_title(f"Episode {eps_num}", fontsize = 12)
#         axs[row,col].axis("off")
#         fig.colorbar(img, ax=axs[row,col], orientation='vertical')
# plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
# plt.show()

for i in range(experiment_repeats):
    print(f"\nIteration {i}")
    for agent_name, agent in agents.items():
        agent.reset()
        print(f"Agent {agent_name}:")
        # Create the environment
        env = gym.make('shortcutshortmaze-v0', render_mode = "rgb_array")

        visited = np.zeros((15,20))
        visite_open = np.zeros((15,20))

        avg_steps = 0
      
       
        for episode in range(n_episodes):
       
            trajectory = []
            done = False
            observation,info= env.reset()
      
            if (episode+1) >= n_episodes//open_shorcut_factor:
                env.open_shortcut() 
            while not done:
                state = tuple(observation["agent"]["pos"])
                if (episode+1) >= n_episodes//open_shorcut_factor:
                    visite_open[state]+=1
                else: 
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

            shortcut_found[agent_name][episode,i] = info["steps"]<optimal_lenght_shorcut_closed
            
            if (episode+1) %100 ==0:
                print(f"episode {episode+1}: trajectory length {info["steps"]:.2f} - average {avg_steps/(episode+1):.2f} ")
                # fig,axs = plt.subplots(1,5)
                # Q = np.copy(agent.Q)
                # Q_avg = Q.mean(axis=-1)
                # mask = Q == 0
                # Q[mask] = np.inf
                # masked_Q = np.ma.masked_invalid(Q)
                # mask = Q_avg == 0
                # Q_avg[mask] = np.inf
                # masked_Q_avg = np.ma.masked_invalid(Q_avg)
                # cmap = plt.cm.viridis 
                # cmap.set_bad(color='black')
                # axs[0].imshow(Q_avg,cmap = cmap)
                # axs[1].imshow(masked_Q[:,:,0],cmap = cmap)
                # axs[2].imshow(masked_Q[:,:,1],cmap = cmap)
                # axs[3].imshow(masked_Q[:,:,2],cmap = cmap)
                # axs[4].imshow(masked_Q[:,:,3],cmap = cmap)
                # plt.show()
            if (episode+1) in eps_checks:
                Q = np.copy(agent.Q)
                Q_avg = Q.mean(axis=-1)
                print(Q_avg.shape,Q_avgs[agent_name][episode+1].shape )
                Q_avgs[agent_name][episode+1]+= Q_avg
                

            
                # fig,axs = plt.subplots(1,5)
                # last_nonzero = np.nonzero(agent.H)[0][-1]+1
                # Q = np.copy(agent.Q)
                # mask = Q ==0
                # Q[mask] = np.inf
                # masked_Q = np.ma.masked_invalid(Q)
                # cmap = plt.cm.viridis 
                # cmap.set_bad(color='black')
                # axs[0].bar(range(len(agent.H[:last_nonzero])), agent.H[:last_nonzero],width=1)
                # axs[1].imshow(masked_Q[:,:,0],cmap = cmap)
                # axs[2].imshow(masked_Q[:,:,1],cmap = cmap)
                # axs[3].imshow(masked_Q[:,:,2],cmap = cmap)
                # axs[4].imshow(masked_Q[:,:,3],cmap = cmap)
                # plt.show()
                
         
            
            if (episode+1) %evaluate_policy_every  ==0 and (episode+1)>=evaluate_policy_after:
                G = 0
                
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

        
    
        # mask = visited==0
        # visited[mask] = np.inf
        # visited = np.log(visited)
        # masked_visited = np.ma.masked_invalid(visited)

        # mask = visite_open==0
        # visite_open[mask] = np.inf
        # visite_open = np.log(visite_open)
        # masked_visite_open = np.ma.masked_invalid(visite_open)

        # # Get the minimum and maximum values of the valid data
        # if agent_name =="QWL":
        #     vmin, vmax = masked_visited.min(), masked_visited.max()
        
        # cmap = plt.cm.YlOrRd   # Choose any colormap you prefer
        # cmap.set_bad(color='black')
        # norm = Normalize(vmin=0, vmax=vmax)


        # fig, axs = plt.subplots(1,3, figsize = (18,10))
        # img =axs[0].imshow(masked_visited, cmap=cmap, norm=norm)
        # axs[0].set_title(r"Number of state visit in $\ln$ before open shortcut", fontsize = 14)
        # fig.colorbar(img, ax=axs[0], orientation='vertical')

        # img =axs[1].imshow(masked_visite_open, cmap=cmap, norm=norm)
        # axs[1].set_title(r"Number of state visit in $\ln$ after open shortcut", fontsize = 14)
        # fig.colorbar(img, ax=axs[1], orientation='vertical')
        
        # last_nonzero = np.nonzero(agent.H)[0][-1]+1

        # axs[2].bar(range(len(agent.H[:last_nonzero])), agent.H[:last_nonzero],width=agent.binsize)
        # axs[2].set_title(f"Histogram of visited Q-values", fontsize = 14)
        # axs[2].set_xlabel(r"binned $Q(a|s)$", fontsize = 12)
        # axs[2].set_ylabel("frequency", fontsize = 12)
        # plt.tight_layout()
        # plt.savefig(f"plots/{agent_name}_shortcut_short_plots.png")
        # plt.show()

    # observation,info= env.reset()
    # env.open_shortcut()

    # writer = imageio.get_writer(f'videos/{agent_name}_shortcut_short.mp4', fps=5)

    # done = False
 
    # while not done:
    #     state =  tuple(observation["agent"]["pos"])
    #     action = agent.select_action(state)
    #     observation, reward, terminated,truncated , info = env.step(action)
    #     done = terminated or truncated
    #     frame = env.render()
    #     writer.append_data(frame)
    #     if done:
    #         break
    #     next_state =  tuple(observation["agent"]["pos"])
    #     state = next_state

    # # Close the environment
    # writer.close()
    # env.close() 

colors = {"QWL": "#1f77b4", "SARSA": "#ff7f0e"}
average_window_size = 10
x_pos = int(-1 + n_episodes // open_shorcut_factor)


cmap = plt.cm.viridis 
cmap.set_bad(color='black')

fig, axs = plt.subplots(nrows=2,ncols=len(eps_checks),figsize=(16, 8))
for row,(agent_name, value_funcs) in enumerate(Q_avgs.items()):
    fig.text(0.025,(2 * (1-row) +1) / 4, agent_name, ha='center', va='center', fontsize=16)
    for col,(eps_num, Q_avg) in enumerate(value_funcs.items()):
        if row ==0:
            fig.text((2 * col +1) / (2*4), 0.95, f"Episode {eps_num}", ha='center', va='center', fontsize=14)
        Q_avg/=experiment_repeats
        mask = Q_avg == 0
        Q_avg[mask] = np.inf
        masked_Q_avg = np.ma.masked_invalid(Q_avg)
        img =axs[row,col].imshow(masked_Q_avg,cmap = cmap)
        #axs[row,col].set_title(f"Episode {eps_num}", fontsize = 12)
        axs[row,col].axis("off")
        fig.colorbar(img, ax=axs[row,col], orientation='vertical')
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
plt.savefig(f"plots/shortcut_short_average_Q_values_over_actions.png")
plt.close()
#plt.show()



for agent_name in agents.keys():




    # cumsum = np.cumsum(shortcut_found[agent_name], axis=0)
    # mean_ma = np.mean(cumsum, axis=1)
    # std_ma = np.std(cumsum, axis=1)
    
    moving_averages = moving_average(shortcut_found[agent_name], average_window_size, axis=0)
    mean_ma = np.mean(moving_averages, axis=1)
    q25_ma = np.percentile(moving_averages,q = 25, axis=1)
    q75_ma = np.percentile(moving_averages,q = 75, axis=1)
    print(q25_ma)
    print(mean_ma)
    print(q75_ma)
    

    # avg_shortcut_found = np.mean(shortcut_found[agent_name], axis=1)
    # #std_shortcut_found = np.std(shortcut_found[agent_name], axis=1)
    # q25_shortcut_found = np.percentile(shortcut_found[agent_name],q = 25, axis=1)
    # q75_shortcut_found = np.percentile(shortcut_found[agent_name],q = 75, axis=1)
    # mean_ma = moving_average(avg_shortcut_found, average_window_size)
    # #std_ma = moving_average(std_shortcut_found, average_window_size)
    # q25_ma = moving_average(q25_shortcut_found, average_window_size)
    # q75_ma = moving_average(q75_shortcut_found, average_window_size)
    
    x = np.arange(len(mean_ma))

    plt.plot(x , mean_ma,'--',color = colors[agent_name], label = f"{agent_name} w/ uncertainty (25-75th Percentile)" )
    plt.fill_between(x, q25_ma , q75_ma, color=colors[agent_name], alpha=0.3)


plt.axvline(x = -1 + n_episodes//open_shorcut_factor, color = 'black', linestyle = ":")
# Annotate the vertical line
plt.annotate('Shortcut opens', xy=(x_pos, 0.5), xytext=(x_pos-200, 0.6),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            ha='center', va='center', fontsize=12)

plt.xlim(left = x_pos-500 )
plt.ylabel("Shortcut found", fontsize=12)
plt.xlabel("Episode number", fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(f"plots/shortcut_found_short_plots.png")
plt.show()

# for agent_name in agents.keys():
#     plt.plot(range(evaluate_policy_after,evaluate_policy_after + evaluate_policy_every*(len(returns[agent_name])),evaluate_policy_every), returns[agent_name],'--o', label = agent_name)
# plt.axvline(x = n_episodes//2, color = 'black', linestyle = ":")
# plt.ylabel("Return G")
# plt.xlabel("Episode number")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()


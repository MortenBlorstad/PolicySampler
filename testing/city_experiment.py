import gymnasium as gym
import pathfinders
from agents.QWL import QWLAgent
from agents.agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import imageio
from mpl_toolkits.mplot3d import Axes3D

from scipy.special import softmax



np.random.seed(0)
agent = SarsaAgent(15,15, 4,0.1, 0.2, 0.99)
agent = QWLAgent(15,15, 4, 0.1 ,0.2,0.99,1000)






cmap = plt.cm.viridis 
cmap.set_bad(color='black')




env = gym.make('city-v0', render_mode = "rgb_array")
observation,info= env.reset()

agent.Q = env.get_valid_Q().copy()
agents = {"QWL": QWLAgent(15,18, 4, 1 ,0.2,1,1000),
          "SARSA": SarsaAgent(15,18, 4,0.1, 0.2, 1)
           }


writer = imageio.get_writer(f'videos/city.mp4', fps=5)
visited = np.zeros((15,15,21))
avg_steps = 0
done = False
n_episodes = 5000
experiment_repeats = 10
shortcut_found = {agent_name : np.zeros((experiment_repeats)) for agent_name in agents.keys()}

for i in range(experiment_repeats):
    print(f"\nIteration {i}")
    for agent_name, agent in agents.items():
        agent.reset()
        agent.Q = env.get_valid_Q().copy()
        print(f"Agent {agent_name}:")

        for episode in range(n_episodes):
            done = False
            observation, info= env.reset()
            G = 0
            
           
            while not done:
                state = tuple(list(observation["agent"]["pos"])+ [observation["agent"]["charge"]])
                visited[state]+=1

                action = agent.select_behaviour_action(state)
                
                
                observation, reward, done,_, info = env.step(action)
                G+=reward
                    
                next_state = tuple(list(observation["agent"]["pos"])+ [observation["agent"]["charge"]])
                
                agent.learn(state,action,reward,next_state)
                state = next_state
            
            print(G)
            if G> -100:
                shortcut_found[agent_name][i] = episode+1
                print(f"episode {episode+1}: trajectory length {info["steps"]:.2f} - Return {G:.2f} ")
                break
                
                #plt.imshow(np.log(visited.mean(axis=-1)))
                # Q = np.copy(agent.Q)
                # mask = Q == 0
                # Q[mask] = np.inf
                # masked_Q = np.ma.masked_invalid(Q)

                # masked_Q = masked_Q.mean(axis=-1)

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # elevation = 30  # degrees
                # azimuth = -80    # degrees
                # ax.view_init(elev=elevation, azim=azimuth)
                # x, y, z = np.nonzero(masked_Q) 
                # colors = masked_Q[x, y, z]

                # x_image, y_image = np.meshgrid(np.arange(0,16,1), np.arange(0,16,1))
                # z_image = np.zeros(x_image.shape) 
                # # graphics = np.random.randint(0, 256, (15, 15, 3))
                # facecolors =  env.graphics[:16, :16, :3].copy()/255.# graphics[:x_image.shape[0], :y_image.shape[1], :]/255.0 #
                
                
                # ax.plot_surface(x_image, y_image, z_image-5, rstride = 1,cstride = 1,facecolors=np.flip(facecolors,axis=0))
                # ax.set_zlim(-5,21)
                # ax.set_xlim(-1,16)
                # ax.set_ylim(-1,16)
                # ax.set_xlabel("x")
                # ax.set_ylabel("y")
                # ax.set_zlabel("charge")

                # #x, y, z = masked_Q.mean(axis=-1)
                # #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16, 8))
            
                # img = ax.scatter(x, y, z,c=colors,alpha=0.5,cmap= cmap)
                # fig.colorbar(img, ax=ax, orientation='vertical')

                # # print(env.graphics[:,:,:3]/255)
                

                
                # plt.show()
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
colors = {"QWL": "#1f77b4", "SARSA": "#ff7f0e"}

print(shortcut_found)

for agent_name in agents.keys():
    print(shortcut_found[agent_name].mean())
    moving_averages = moving_average(shortcut_found[agent_name], 10, axis=0)
    mean_ma = np.mean(moving_averages, axis=1)
    q25_ma = np.percentile(moving_averages,q = 25, axis=1)
    q75_ma = np.percentile(moving_averages,q = 75, axis=1)
    x = np.arange(len(mean_ma))

    plt.plot(x , mean_ma,'--',color = colors[agent_name], label = f"{agent_name} w/ uncertainty (25-75th Percentile)" )
    plt.fill_between(x, q25_ma , q75_ma, color=colors[agent_name], alpha=0.3)


plt.ylabel("Better than Argh", fontsize=12)
plt.xlabel("Episode number", fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()


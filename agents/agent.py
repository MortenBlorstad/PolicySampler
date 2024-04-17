import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from scipy.special import softmax
import imageio
import os

class AbstractAgent(ABC):
    """
    This is an abstract class for creating agents to interact with environments in Gymnasium.
    """

    def __init__(self):
        """
        Initialize the agent with the given action and observation space.
        
        Parameters:
        action_space: The action space of the environment.
        """
    
  
    @abstractmethod
    def select_action(self, observation):
        """
        Abstract method to define how the agent selects an action based on the current observation.
        
        Parameters:
        observation: The current observation from the environment.
        
        Returns:
        The action to be taken.
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """
        Abstract method to define the learning process of the agent.
        """
        pass



class SarsaAgent(AbstractAgent):
    
    def __init__(self, nrows:int,ncols:int,  action_space:int,epsilon = 0.3, alpha:float = 0.1, gamma:float = 1.0) -> None:
        self.Q = np.zeros((nrows,ncols,action_space))
        self.offset = np.ones((nrows,ncols,action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        

    def learn(self, state, action, reward, next_state):
        """
        TD0 update of action value

        """
        next_action = self.select_action(next_state)
        td_error =  reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error


    def select_action(self, state):
        

        action_values = self.Q[state[0], state[1]]*self.offset[state[0], state[1]]
        if np.random.uniform()<self.epsilon:
            return np.random.choice(len(action_values))  
    
        max_value = np.max(action_values)
        max_actions = np.where(action_values == max_value)[0]
        
        
        action = np.random.choice(max_actions)
        return action

import matplotlib.pyplot as plt

class MCAgent(SarsaAgent):
    
    def __init__(self, nrows:int,ncols:int,  action_space:int,epsilon = 0.3, alpha:float = 0.1, gamma:float = 1.0) -> None:
        self.Q = np.zeros((nrows,ncols,action_space))
        self.offset = np.ones((nrows,ncols,action_space))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon



    def learn(self, tau: np.ndarray):
        """
        MC update of action value from trajactory tau 
            tau: trajactory

        """
        G = 0
        returns = {}
        first_visit = set()
        tau_ = []
        for idx, step in enumerate(tau):
            state, action, reward, next_state = step
            if (state, action) not in first_visit:
                first_visit.add((state, action))
                tau_.append(step)
        
        for state, action, reward, next_state  in reversed(tau_):
            if (state, action) not in returns:
                returns[(state, action)] = []
        
        
            returns[(state, action)].append(G)
            G  = self.gamma*G + reward
            self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*np.mean(returns[(state, action)])


class PolicySamplerAgent(SarsaAgent):
    def __init__(self, env, nrows:int,ncols:int,  action_space:int,epsilon = 0.3, gamma:float = 1.0) -> None:
        self.pi = softmax(np.ones((nrows,ncols,action_space)),axis=2)
        self.gamma = gamma
        self.env = env
        self.epsilon = epsilon


    def sampler(self,E_min,E_max, flatness_criteria=0.8, final_ln_f=1e-2, binsize= 10):
        num_bins = (E_max - E_min) // binsize + 1
        E_range = np.arange(E_min, E_max + binsize,binsize) # set seectrum Gamma
        S = np.zeros(num_bins)  # entropy
        H = np.zeros(num_bins)  # histogram of visited E
    
        E = self.evaluate(self.pi) # current E given policy
        current_E_bin = round((E - E_min) // binsize)
        print(E,current_E_bin)
        best_E = E
        best_pi = self.pi

        ln_f = 1.0  # Initial modification factor (ln(f))
        iteration = 0  # Counter for iterations
        images = []

        while ln_f > final_ln_f:
            iteration += 1
            proposed_pi = softmax(self.pi + self.epsilon*np.random.normal(size=self.pi.shape),axis=2)
            proposed_E = self.evaluate(proposed_pi)
            proposed_E_bin = round((proposed_E - E_min) // binsize)
            if proposed_E < E_min or proposed_E > E_max:
                proposed_E_bin = current_E_bin  # Reject moves that go out of the allowed energy range
                proposed_E = E
            
            # Metropolis-Hastings acceptance criterion (in log form)
            if np.log(np.random.rand()) < S[current_E_bin] - S[proposed_E_bin]:
                current_E_bin = proposed_E_bin  # Accept move
                self.pi = proposed_pi
    

            if proposed_E<best_E:
                best_E = proposed_E
                best_pi = self.pi
                print(best_E)
            

        
            
            # Update the density of states and the histogram
            
            S[current_E_bin] += ln_f
            H[current_E_bin] += 1
            # Check for histogram "flatness"
            if iteration % 200 == 0:  # Check flatness every n iterations
                H_ = H[H>0]
                half_ind = len(H_)//2
                meanH = H_[half_ind:].mean()  # mean Histogram
                minH = H_[:half_ind].mean()  # minimum of the Histogram
                fig, ax = plt.subplots()
                plt.bar(range(len(H)), H,width=binsize)
                ax.set_title(f'Iteration {iteration}')
                plt.savefig('temp.png') # Save the plot as a PNG
                images.append(imageio.imread('temp.png'))  # Load the saved image
                plt.close(fig)  # Close the plot figure
                if  flatness_criteria < minH/meanH < 1/flatness_criteria:  # Flatness condition
                    print(f"Reducing ln_f: {ln_f} -> {ln_f / 2} at iteration {iteration}")
                    
                    ln_f /= 2.0  # Reduce the modification factor
                    H[:] = 0  # Reset histogram

        # Create GIF
        imageio.mimsave('sampling_animation.gif', images, fps=2, loop = 0)
        os.remove('temp.png')

        self.pi = best_pi

    def select_action(self, state, pi):
        
        action_prob = pi[state[0], state[1]]

        return np.random.choice(len(action_prob),p=action_prob)  


    def evaluate(self,pi, n_episodes = 10):
        

        avgG = 0
        
        for i in range(n_episodes):
            done = False
            observation, info= self.env.reset()
            G = 0
            while not done:
                state = tuple(observation["agent"]["pos"])

                action = self.select_action(state,pi)
                observation, reward, terminated,truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = tuple(observation["agent"]["pos"])
                state = next_state
                G+= self.gamma*reward
            avgG+=G

        avgG/=n_episodes
        return -avgG
    

        


    def learn(self, state, action, reward, next_state):
        """
        TD0 update of action value

        """


        next_action = self.select_action(next_state)
        td_error =  reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error
    






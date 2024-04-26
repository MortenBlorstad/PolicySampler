
from typing import Tuple, List
from scipy.special import softmax
import imageio
import os
import numpy as np


class QWLAgent():
    def __init__(self, nrows:int,ncols:int,  action_space:int, binsize:int=2,alpha = 0.2, gamma:float = 1.0, E_max = 20) -> None:
        self.Q = np.zeros((nrows,ncols,action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.E_max = E_max
        self.binsize = binsize
        self.num_bins = (E_max - 0) // binsize + 1
        self.S = np.zeros(self.num_bins)  # entropy
        self.H = np.zeros(self.num_bins)  # histogram of visited E

        




    def sample_action(self, state):
        """Sample action using wang landau sampling"""
        action_values = self.Q[state[0], state[1]]
        E = abs(np.max(action_values))
        current_E_bin = round((E - 0) // self.binsize)
        while True:
            action = np.random.choice(len(action_values))
            proposed_E = abs(action_values[action])
            proposed_E_bin = round((proposed_E - 0) // self.binsize)
            if proposed_E < 0 or proposed_E > self.E_max :
                continue
            # Metropolis-Hastings acceptance criterion (in log form)
            if np.log(np.random.rand()) < self.S[current_E_bin] - self.S[proposed_E_bin]:
                current_E_bin = proposed_E_bin  # Accept move        
                break
        self.S[current_E_bin] += 1
        self.H[current_E_bin] += 1
        return action

        

    def select_action(self, state):
        """Select greedy action from Q-function"""
        action_values = self.Q[state[0], state[1]]
        max_value = np.max(action_values)
        max_actions = np.where(action_values == max_value)[0]
        action = np.random.choice(max_actions)
        return action


    def learn(self, state, action, reward, next_state):
        """
        TD0 update of action value
        """
        
        next_action = self.select_action(next_state)
        td_error =  reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error
   
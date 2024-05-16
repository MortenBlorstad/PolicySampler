
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
        self.num_bins = int((E_max - 0) // binsize + 1)
        self.S = np.zeros(self.num_bins)  # entropy
        self.H = np.zeros(self.num_bins)  # histogram of visited E
        self.f = 1
        self.greedy_prob = 0.5
        

        
    def reset(self):
        self.Q = np.zeros(self.Q.shape) 
        self.S = np.zeros(self.num_bins)  # entropy
        self.H = np.zeros(self.num_bins)  # histogram of visited E

    def is_histogram_flat(self, tolerance=0.1):
        """Check if the histogram is flat within the specified tolerance."""
        nonzeroes = np.nonzero(self.H)[0]
        
        if len(nonzeroes) < 2:
            return False  # Not enough data to determine flatness
        
        first_non_zero = nonzeroes[0]
        last_non_zero = nonzeroes[-1]+1

        nonzero_bins = self.H[first_non_zero:last_non_zero]

        nonzero_bins = self.moving_average(nonzero_bins,2)
        average_count = np.mean(nonzero_bins)
        if average_count == 0:
            return False  # Avoid division by zero
        min_count = (1 - tolerance) * average_count
        max_count = (1 + tolerance) * average_count
        return np.all((nonzero_bins >= min_count) & (nonzero_bins <= max_count))

    def moving_average(self,x,w):
        return np.convolve(x, np.ones(w), 'valid') / w




    def select_behaviour_action(self, state):
        """Sample action using wang landau sampling"""
        action_values = self.Q[state].copy()
        action = np.argmax(action_values)
        E = abs(np.max(action_values))
        current_E_bin = round((E - 0) // self.binsize)

        accepted = False
    
        while not accepted:
            action = np.random.choice(len(action_values))
            
            proposed_E = abs(action_values[action])
            proposed_E_bin = round((proposed_E - 0) // self.binsize)
            if proposed_E < 0 or proposed_E > self.E_max:
                continue
            # Metropolis-Hastings acceptance criterion (in log form)
            if np.log(np.random.rand()) < self.S[current_E_bin] - self.S[proposed_E_bin]:
                current_E_bin = proposed_E_bin  # Accept move      
                accepted = True
        self.S[current_E_bin] += self.f
        self.H[current_E_bin] += 1 
        if self.H.ravel().sum() > 1000:
            self.H = np.zeros(self.num_bins)  # histogram of visited E
            self.S = np.zeros(self.num_bins)  # histogram of visited E
            
            
        return action
          
    
    def select_action(self, state):
        """Select greedy action from Q-function"""
        action_values = self.Q[state]
        max_value = np.max(action_values)
        

        max_actions = np.where(action_values == max_value)[0]
        action = np.random.choice(max_actions)
        return action



    def learn(self, state, action, reward, next_state):
        """
        TD0 update of action value
        """
        
        next_action = self.select_action(next_state)
        #next_action = self.Q[next_state].argmax()
        td_error =  reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action]

        self.Q[state][action] += self.alpha * td_error
   
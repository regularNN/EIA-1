import numpy as np
import pickle
import sys


class EnvChannel:
    def __init__(self, resolution_list, d1=.01, d2=.02):

        self.num_delay_bins = 3
        self.num_resolutions = len(resolution_list)

        self.resolution_dict = {resolution: i for i, resolution in enumerate(resolution_list)}
        

        self.num_states = self.num_resolutions*self.num_delay_bins
        self.curr_state = (0,0)
        self.prev_state = (0,0)
        self.reward = 0
        self.action = 0
        self.num_actions = 3
        self.valid_actions = [0, 1, 2]  # Reduce, No change, Increase
        self.d1 = d1
        self.d2 = d2
        self.avg_delay = 0
        self.curr_resolution = 0
        self.error_score=  0

    def sample_action(self):
        return np.random.choice(self.valid_actions)
    

    def get_reward(self):
        return 0.5*self.resolution_state/(self.delay_state+1) +0.5*self.error_score
    
    
    def step(self, action):
        self.action = action
        self.reward = self.get_reward()
        return self.reward, self.estimate_state()

    
    def estimate_state(self):
        self.resolution_state = self.resolution_dict[self.curr_resolution]
        if self.avg_delay <= self.d1:
            self.delay_state = 0  # Good state
        elif self.avg_delay <= self.d2:
            self.delay_state = 1  # Medium state
        else:
            self.delay_state = 2  # Bad state
        self.curr_state = (self.delay_state, self.resolution_state)
        return self.curr_state

    
    def reset(self):
        self.__init__(num_states=self.num_states)
        return self.curr_state


class ControlAgent:
    def __init__(self,
                 resolution_list,
                 d1,
                 d2,
                 alpha=0.1,
                 gamma=.1,
                 epsilon=.3,
                 random_actions=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = EnvChannel(resolution_list, d1, d2)
        self.q_table = np.zeros([self.env.num_delay_bins, self.env.num_resolutions, self.env.num_actions])
        self.all_epochs = []
        self.action_record = []
        self.state_record = []
        self.penalties = []
        self.iteration_i = 0
        self.prev_state = None
        self.prev_avg_delay = 0
        self.random_actions = random_actions
        
  
    def map_action(self, state):
        if state == 0:
            return -1
        if state == 1:
            return 0
        if state == 2:
            return 1
        
 
    def get_signal(self, delay_list, curr_resolution, error_score):
        self.env.avg_delay = np.average(delay_list)
        self.env.curr_resolution = curr_resolution
        self.env.error_score = np.abs(error_score)/ 100.0
        
        self.iteration_i += 1
        if np.mod(self.iteration_i+2, 10) == 0:
            self.alpha = self.alpha*.95
            self.epsilon = self.epsilon*.95
            

        if self.iteration_i == 1:
            state = self.env.estimate_state()
            action = self.env.sample_action()
            reward = 0 

        else:
  
            reward, state = self.env.step(self.prev_action)
            self.state_record.append(self.env.curr_state)
            

            if not self.random_actions:
                old_qvalue = self.q_table[self.prev_state[0],self.prev_state[1], 
                                          self.prev_action]
                next_max = np.max(self.q_table[state[0],state[1], :])

                new_qvalue = (1 - self.alpha) * old_qvalue + \
                    self.alpha * (reward + self.gamma * next_max)
                self.q_table[self.prev_state[0],self.prev_state[1], 
                             self.prev_action] = new_qvalue
                self.penalties.append(reward)
                self.action_record.append(self.prev_action)
                

                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.sample_action()  # Explore action space

                else:
                    # Exploit learned values
                    action = np.argmax(self.q_table[state[0],state[1], :])
            else:
                action = self.env.sample_action()  # Explore action space

        self.prev_state = state
        self.prev_action = action
        return self.map_action(action), action, state, reward 



import numpy as np
import pickle
import sys, os, math
import datetime


# Define an environment channel class for reinforcement learning.
class EnvChannel:
    def __init__(self, resolution_list, d1=.01, d2=.02):
        # Initialize the environment with given parameters.
        self.num_delay_bins = 3  # Number of delay bins (states)
        self.num_resolutions = len(resolution_list)  # Number of resolutions
        resolution_list = np.sort(resolution_list)
        self.resolution_dict = {resolution: i for i, resolution in enumerate(resolution_list)}  # Map resolutions to indices
        
        # Calculate the total number of states, initialize current and previous states, and other attributes.
        self.num_states = self.num_delay_bins * self.num_resolutions
        self.curr_state = (0, 0) # delay, resolution
        self.prev_state = (0, 0)
        self.reward = 0
        self.action = 0
        self.num_actions = 3  # Number of possible actions
        self.valid_actions = [0, 1, 2]  # Valid actions: Reduce, No change, Increase
        self.d1 = d1  # Threshold for "good" delay state
        self.d2 = d2  # Threshold for "medium" delay state
        self.avg_delay = 0
        self.curr_resolution = 0
        self.deviation_error = 0
        self.deviation_state = 0

    def sample_action(self):
        # Randomly sample an action from the valid actions.
        return np.random.choice(self.valid_actions)

    def get_reward(self):
        # Calculate the reward based on the current state and error score.
        return 0.5 / (self.delay_state + 1) + 0.5 * self.deviation_state

    def step(self, action):
        # Perform a step in the environment based on the chosen action.
        self.action = action
        self.reward = -self.get_reward()
        return self.reward, self.estimate_state()

    def estimate_state(self):
        # Estimate the current state based on resolution and average delay.
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
        # Reset the environment to its initial state.
        self.__init__(num_states=self.num_states)
        return self.curr_state

# Define a control agent class for reinforcement learning.
class ControlAgent:
    def __init__(self, resolution_list, d1, d2, alpha=0.1, gamma=0.1, epsilon=0.9, random_actions=False):
        # Initialize the control agent with various parameters.
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.env = EnvChannel(resolution_list, d1, d2)  # Create an environment instance
        self.q_table = np.zeros([self.env.num_delay_bins, self.env.num_resolutions, self.env.num_actions])  # Q-table for storing action values

        self.all_epochs = []  # List to record training epochs
        self.action_record = []  # List to record taken actions
        self.state_record = []  # List to record observed states
        self.penalties = []  # List to record penalties (rewards)
        self.iteration_i = 0  # Iteration counter
        self.prev_state = None  # Previous state
        self.prev_avg_delay = 0  # Previous average delay
        self.random_actions = random_actions  # Flag for random actions
        self.path = "Channel_Data"
        self.f_name = "/Data_"+datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        self.q_table_hist = []
        self.create_directory()
        self.q_table_count_unchanged = 0
        self.q_table_converged_iter = 0
        self.q_table_prev = self.q_table.copy()
        
        
    def create_directory(self):
        # check if channel_data directory exists
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def map_action(self, state):
        # Map an action to a change in delay state.
        if state == 0:
            return -1  # Reduce delay
        if state == 1:
            return 0  # No change in delay
        if state == 2:
            return 1  # Increase delay
            
    def get_deviation_state(self, deviation):
    	if deviation==0:
    		return 0
    	elif deviation <= 25:
    		return 1
    	elif deviation <= 50:
    		return 2
    	else:
    		return 3
    	

    def get_signal(self, delay_list, curr_resolution, deviation_error):
        
        self.q_table_prev = self.q_table.copy()
        
        # Receive input signals and update the agent's knowledge.
        self.env.avg_delay = np.average(delay_list)  # Update average delay
        self.env.curr_resolution = curr_resolution  # Update current resolution
        self.env.deviation_state = self.get_deviation_state(deviation_error)
        # self.env.deviation_error = -np.abs(deviation_error) / 100.0  # Update error score (normalized)
        
        self.iteration_i += 1  # Increment iteration counter
        
        # Decay alpha and epsilon values over time.
        if np.mod(self.iteration_i + 2, 10) == 0:
            self.alpha = self.alpha * 0.95
            self.epsilon = self.epsilon * 0.95

        if self.iteration_i == 1:
            state = self.env.estimate_state()
            action = self.env.sample_action()
            reward = 0
        else:
            reward, state = self.env.step(self.prev_action)
#             self.state_record.append(self.env.curr_state)

            if not self.random_actions:
                old_qvalue = self.q_table[self.prev_state[0], self.prev_state[1], self.prev_action]
                next_max = np.max(self.q_table[state[0], state[1], :])

                # Update Q-value using Q-learning equation.
                new_qvalue = (1 - self.alpha) * old_qvalue + self.alpha * (reward + self.gamma * next_max)
                self.q_table[self.prev_state[0], self.prev_state[1], self.prev_action] = new_qvalue
                self.penalties.append(reward)
                self.action_record.append(self.prev_action)

                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.sample_action()  # Explore action space
                else:
                    # Exploit learned values to select an action.
                    action = np.argmax(self.q_table[state[0], state[1], :])
            else:
                action = self.env.sample_action()  # Explore action space

        self.prev_state = state  # Update previous state
        self.prev_action = action  # Update previous action
        self.q_table_hist.append(self.q_table.copy())
        if (self.q_table_prev == self.q_table).all():
            self.q_table_count_unchanged += 1
            if self.q_table_count_unchanged==10:
                self.q_table_converged_iter = self.iteration_i
        else:
            self.q_table_count_unchanged = 0
        
        if self.iteration_i%20 == 0: 
            with open(self.path+self.f_name,'wb') as fp:
                pickle.dump([self.q_table_hist, self.alpha, self.epsilon], fp)
        # combine total iterations and number of iterations it took for convergence into iter_data
        iter_data = self.q_table_converged_iter + (1.0*self.iteration_i)/10**(int(math.log10(self.iteration_i))+1)
        return self.map_action(action), state, reward, iter_data  # Return mapped action, current state, and reward

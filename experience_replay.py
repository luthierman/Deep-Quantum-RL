from collections import namedtuple,deque
import random
import numpy as np
import pandas as pd
import torch
class ER_Memory(object):
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, 4), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity,1), dtype=torch.int64,device=self.device)
        self.rewards = torch.zeros((capacity,1), dtype=torch.int64,device=self.device)
        self.next_states = torch.zeros((capacity, 4), dtype=torch.float32,device=self.device)
        self.dones = torch.zeros(capacity, dtype=torch.int64,device=self.device)
        self.index = 0
        self._len = 0

    def remember(self, *args):
        exp = [*args]
        self.states[self.index] = torch.from_numpy(exp[0])
        self.actions[self.index] = int(exp[1])
        self.rewards[self.index] = exp[2]
        self.next_states[self.index] = torch.from_numpy(exp[3])
        self.dones[self.index] = exp[4]
        self.index = (self.index+1)%self.capacity
        self._len = min(self._len+1, self.capacity)

    def sample(self, batch_size):
        batch_idxs= np.random.choice(np.arange(self._len), size=min(self._len, batch_size), replace=True)
        return self.states[batch_idxs], self.actions[batch_idxs], self.rewards[batch_idxs], self.next_states[batch_idxs], self.dones[batch_idxs]

    def __len__(self):
      return self._len

    def save(self, path):
      d = {"states": self.states,
           "actions":self.actions,
           "next_states": self.next_states,
           "rewards": self.rewards,
           "dones": self.dones,
           "index": self.index}
      torch.save(d,path)

    def load(self,path):
      d = torch.load(path,map_location=self.device)
      self.states = d["states"]
      self.actions = d["actions"]
      self.next_states = d["next_states"]
      self.rewards = d["rewards"]
      self.dones = d["dones"]
      self.index = d["index"]

    def show(self):
      df =  pd.DataFrame()
      df["states"]= list(self.states.numpy())
      df["actions"]= self.actions.numpy()
      df["rewards"]= self.rewards.numpy()
      df["next_states"] = list(self.next_states.numpy())
      df["dones"] = self.dones.numpy()
      return df

class PER_Memory(object):
    def __init__(self, capacity, device="cpu"):
        self.e = 0.1
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_per = 0.01
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, 4), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity,1), dtype=torch.int64,device=self.device)
        self.rewards = torch.zeros((capacity,1), dtype=torch.int64,device=self.device)
        self.next_states = torch.zeros((capacity, 4), dtype=torch.float32,device=self.device)
        self.dones = torch.zeros(capacity, dtype=torch.int64,device=self.device)
        self.priorities = torch.zeros((capacity,1), dtype=torch.float32,device=self.device)
        self.index = 0
        self._len = 0
        
    def remember(self, *args):
        exp = [*args]
        self.states[self.index] = torch.from_numpy(exp[0])
        self.actions[self.index] = int(exp[1])
        self.rewards[self.index] = exp[2]
        self.next_states[self.index] = torch.from_numpy(exp[3])
        self.dones[self.index] = exp[4]
        self.priorities[self.index] = max(self.priorities.max(), 1)
        self.index = (self.index+1)%self.capacity
        self._len = min(self._len+1, self.capacity)

    def get_probabilities(self):
        scaled_priorities = self.priorities.pow(self.alpha)
        sample_probabilities = scaled_priorities/ torch.sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = (1/self._len) * (1/probabilities)
        importance_normalized = importance / np.max(importance)
        return importance_normalized

    def sample(self, batch_size):
        sample_probs = self.get_probabilities()[:self._len].detach().numpy().flatten()
        batch_idxs = np.random.choice(np.arange(self._len),
                                      size=min(self._len, batch_size),
                                      p=sample_probs, replace=True)
        minibatch = self.states[batch_idxs], self.actions[batch_idxs], self.rewards[batch_idxs], self.next_states[batch_idxs], self.dones[batch_idxs]
        w = self.get_importance(sample_probs[batch_idxs])
        return batch_idxs, minibatch,  w

    def __len__(self):
        return self._len

    def update_priorities(self, indices, priorities):
        clipped_errors = np.minimum(priorities.detach().numpy(), 1.0)
        ps = np.power(clipped_errors, self.alpha)
        for i,p in zip( indices, ps):
          self.priorities[i]= p+self.e
  
    def show(self):
        df =  pd.DataFrame()
        df["states"]= list(self.states.numpy())
        df["actions"]= self.actions.numpy()
        df["rewards"]= self.rewards.numpy()
        df["next_states"] = list(self.next_states.numpy())
        df["dones"] = self.dones.numpy()
        df["priorities"] = self.priorities.numpy()
        return df

    def save(self, path):
        d = {"states": self.states,
            "actions":self.actions,
            "next_states": self.next_states,
            "rewards": self.rewards,
            "dones": self.dones,
            "index": self.index,
            "priorities": self.priorities}
        torch.save(d,path)

    def load(self,path):
        d = torch.load(path,map_location=self.device)
        self.states = d["states"]
        self.actions = d["actions"]
        self.next_states = d["next_states"]
        self.rewards = d["rewards"]
        self.dones = d["dones"]
        self.index = d["index"]
        if "priorities" in d.keys():
          self.priorities = d["priorities"]
        else: 
          for i in range(self.index):
            self.priorities[i] = 1

class SumTree(object):
    data_pointer = 0
    
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        self.device= "cpu"
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = torch.zeros((2 * capacity - 1,1), dtype=torch.float32, device = self.device)
        # Contains the experiences (so the size of data is capacity)
        self.states = torch.zeros((capacity, 4), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity,1), dtype=torch.int64,device=self.device)
        self.next_states = torch.zeros((capacity, 4), dtype=torch.float32,device=self.device)
        self.rewards = torch.zeros((capacity,1), dtype=torch.int64,device=self.device)

        self.dones = torch.zeros(capacity, dtype=torch.int64,device=self.device)
    
    
    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.states[self.data_pointer] = data[0]
        self.actions[self.data_pointer] = data[1]
        self.rewards[self.data_pointer] = data[2]
        self.next_states[self.data_pointer] = data[3]

        self.dones[self.data_pointer] = data[4]

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
            
    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = float(priority)

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
        
    # Here build a function to get a leaf from our tree. So we'll build a function to get the leaf_index, priority value of that leaf and experience associated with that leaf index:
    def get_leaf(self, v):
        parent_index = 0
        v = float(v[0])
        # the while loop is faster than the method in the reference code
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], (self.states[data_index], 
                                                   self.actions[data_index], 
                                                   self.rewards[data_index],
                                                   self.next_states[data_index],
                                                   self.dones[data_index])
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

# Now we finished constructing our SumTree object, next we'll build a memory object.
class PER_MemoryTree(object):  # stored as ( state, action, reward, next_state ) in SumTree
    
    def __init__(self, capacity):
        # Making the tree 
        self.capacity = capacity
        self.tree= SumTree(capacity)
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001
        self.absolute_error_upper = 1.  # clipped abs error
        self._len = 0

    # Next, we define a function to store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def remember(self, state,action,reward, next_state,done):
        experience = (torch.from_numpy(state),action,reward, torch.from_numpy(next_state), done)
        # Find the max priority
        max_priority = torch.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max priority for new priority
        self._len = min(self._len+1, self.capacity)
    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []
        states, actions, rewards,next_states, dones = [],[],[],[],[]
        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i]= index
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])
        states = torch.stack(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones).unsqueeze(1)
        minibatch = states, actions, rewards, next_states,dones  
        return b_idx, minibatch
    
    # Update the priorities on the tree
    def update_priorities(self, tree_idx, abs_errors):
        abs_errors = abs_errors.detach().numpy()
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    def __len__(self):
        return self._len
    def show(self):
        df =  pd.DataFrame()
        df["states"]= list(self.tree.states.numpy())
        df["actions"]= self.tree.actions.numpy()
        df["rewards"]= self.tree.rewards.numpy()
        df["next_states"] = list(self.tree.next_states.numpy())
        df["dones"] = self.tree.dones.numpy()
        df["priorities"] = self.tree.tree.numpy()
        return df

    def save(self, path):
        d = {"states": self.tree.states,
            "actions":self.tree.actions,
            "next_states": self.tree.next_states,
            "rewards": self.tree.rewards,
            "dones": self.tree.dones,
            "priorities": self.tree.tree}
        torch.save(d,path)

    def load(self,path):
        d = torch.load(path,map_location=self.device)
        self.states = d["states"]
        self.actions = d["actions"]
        self.next_states = d["next_states"]
        self.rewards = d["rewards"]
        self.dones = d["dones"]
        self.index = d["index"]
        if "priorities" in d.keys():
          self.priorities = d["priorities"]
        else: 
          for i in range(self.index):
            self.priorities[i] = 1




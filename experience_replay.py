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

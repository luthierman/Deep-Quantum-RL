from hyperparameters import *
from experience_replay import *
from dense_models import *
from quantum_models import *
from copy import copy
import gym
import matplotlib.pyplot as plt
from torch.optim import *
from collections import deque
import random
import numpy as np
import os
import time
import datetime
from pathlib import Path
import wandb

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

class DQN(object):
    def __init__(self, model, config, name, use_wandb=False, env="CartPole-v0") -> None:
        # GYM environment
        self.name =name
        self.use_wandb =use_wandb
        if self.use_wandb:
          self.run = wandb.init(project="test1", 
                 entity="luthier-man",
                 name=self.name,
                 config=config,
                 monitor_gym=True,
                 save_code=True, reinit=True)
          self.run.log_code(".")
          
          self.run.define_metric("Train Episode", hidden=True)
          self.run.define_metric("Test Episode", hidden=True)

          self.run.define_metric("Average Loss Per Episode","Train Episode")
          self.run.define_metric("Running Average Rewards (50)","Train Episode")
          self.run.define_metric("Q-values","Train Episode")
          self.run.define_metric("Average Q-values","Train Episode")

          self.run.define_metric("Episode Time", "Train Episode")
          self.run.define_metric("Total Train Rewards", "Train Episode")
          self.run.define_metric("Total Test Rewards", "Test Episode")

        self.env = gym.wrappers.RecordVideo(gym.make(env), f"videos",  episode_trigger = lambda x: x % 10 == 0)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        print("State Space: {}".format(self.state_space))
        print("Action Space: {}".format(self.action_space))
        
        # HYPERPARAMETERS
        self.config = config
        print(config)
        self.lr = self.config["learning_rate"]
        self.gamma = self.config["gamma"]
        self.epsilon = self.config["epsilon"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.epsilon_min = self.config["epsilon_min"]
        self.batch = self.config["batch_size"]
        self.episodes = self.config["n_episodes"]
        self.buff = self.config["buffer_size"]
        self.update_target = self.config["update_target"]
        self.update_model = self.config["update_model"]
        self.train_start = self.config["train_start"]
        self.ddqn= self.config["is_DDQN"]
        self.use_per = self.config["use_PER"]
        self.reupload= self.config["is_Reupload"]
        self.n_layers = self.config["n_layers"]

        use_cuda = False
        
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        # Q-network
        self.q_network = model(self.n_layers)
        self.q_network_test = model(self.n_layers)

        # Quantum Reupload prep
        if self.reupload:
          self.lr_in = self.config["learning_rate_in"]
          self.lr_out = self.config["learning_rate_out"]
          self.param_in = [self.q_network.qvc.w_in]
          self.param_weights = [self.q_network.qvc.weights]
          self.param_out = [self.q_network.w_out]
          self.opt_in =Adam(self.param_in, lr=self.lr_in,amsgrad=True)
          self.opt_var =Adam(self.param_weights,lr=self.lr,amsgrad=True)
          self.opt_out =Adam(self.param_out, lr=self.lr_out,amsgrad=True)
        else:
          self.opt = Adam(self.q_network.parameters(),lr=self.lr,amsgrad=True)
        # Target network
        self.target = model(self.n_layers)
        self.sync_weights()
        # GPU setup

        if use_cuda:
            print("GPU being used:", torch.cuda.get_device_name(0))
            self.q_network.cuda(self.device)
            self.target.cuda(self.device)
        self.target.eval()
        
        # DQN setup
        self.loss_fn = torch.nn.MSELoss()
        if self.use_wandb:
          self.run.watch(self.q_network, log="all",log_freq=1,
            criterion=self.loss_fn, log_graph=True)
        if self.use_per:
          self.memory = PER_Memory(self.buff)
        else:
          self.memory = ER_Memory(self.buff)
        self.counter = 0
        self.step = 0
        self.current_episode = 0
        # stat tracking
        self.rewards = []
        self.avg_rewards = []
        self.losses = []
        self.q_values = []

        self.test_rewards = []
        self.test_avg_rewards = []
        self.test_q_values = []

        self.name = name
        self.path = "{}_logs".format(self.name)
        self.save_path = "{}/{}_ep_{}.pt".format(self.path,self.name,self.current_episode+1)
        os.makedirs(self.path, exist_ok=True)
        
    def preprocess_state(self,x):
        x = np.stack(x)
        state = torch.FloatTensor(x).to(self.device)
        return state
    
    def sync_weights(self):
        self.target.load_state_dict(self.q_network.state_dict())

    def sync_weights_test(self):
        self.q_network_test.load_state_dict(self.q_network.state_dict())
    
    def opt_zero_grad(self):
        if self.reupload:
            self.opt_in.zero_grad(set_to_none=True)
            self.opt_var.zero_grad(set_to_none=True)
            self.opt_out.zero_grad(set_to_none=True)
        else:
            self.opt.zero_grad(set_to_none=True)
    def opt_step(self):
        if self.reupload:
            self.opt_in.step()
            self.opt_var.step()
            self.opt_out.step()
        else:
            self.opt.step()
    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)
        self.counter += 1

    def get_action(self, obs, is_test=False):
        if is_test:
            self.q_network_test.eval()
            obs = self.preprocess_state([obs])
            return int(self.q_network_test(obs).argmax().detach())
        if len(self.memory) > self.batch:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            self.q_network.eval()
            obs = self.preprocess_state([obs])
            return int(self.q_network(obs).argmax().detach())

    def learn(self):
        if len(self.memory) < self.train_start:
            return 0
        if self.use_per:
          idx, minibatch, ws = self.memory.sample(self.batch)
        else: 
          minibatch = self.memory.sample(min(len(self.memory), self.batch))
        states, actions, rewards, next_states, dones = minibatch
        self.q_network.train()
        self.target.eval()
        # DDQN
        if self.ddqn:
          Q = self.q_network.forward(states).gather(1, actions).squeeze(-1)# Q(s, a, wq)
          A_best = self.q_network.forward(next_states).argmax(1).reshape(-1,1) # 
          Q_next = self.target.forward(next_states).gather(1, A_best).squeeze(-1) # max _a Q(ns, argmax_a(Q(ns, a, wq)) , wt)
        # DQN
        else:
          Q = self.q_network.forward(states).gather(1, actions).squeeze(-1)# Q(s, a, wq)
          Q_next = self.target.forward(next_states).max(1)[0].detach() # max _a Q(ns, a, wt)
        y = torch.flatten(rewards) + self.gamma *(1-torch.flatten(dones)) * Q_next # bellman 
        if self.use_per:
          # PER
          error = torch.absolute(y-Q)
          self.memory.update_priorities(idx, error)
          self.opt_zero_grad()
          loss = torch.multiply((error).pow(2),self.preprocess_state(ws).pow(1-self.epsilon))
          loss=loss.mean()
          loss.backward()
          self.opt_step()
        else:
          # ER
          self.opt_zero_grad()
          loss = self.loss_fn(Q,y)
          loss.backward()
          self.opt_step()
         
        return float(loss), Q.detach().numpy().flatten()
    
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.sync_weights()

    def train(self, episodes=None):
        alert=True
        if episodes == None:
          episodes = self.episodes
        start_episode = self.current_episode
        episodes +=start_episode
        for t in range(start_episode, episodes):
          start_time = time.time()
          s1 = self.env.reset()
          steps = 1
          train_steps = 0
          done = False
          total_reward = 0
          total_loss = [] 
          total_q = []
          while not done:
              # self.env.render()
              action = self.get_action(s1)
              s2, reward, done, _ = self.env.step(action)
              total_reward+=reward
              self.remember(s1,action,reward,s2,done)
              if done:
                  episode_time = time.time() - start_time
                  # Store Trackable Stats
                  self.rewards.append(total_reward),
                  avg = np.mean(self.rewards[-50:])
                  losses = np.asarray(total_loss)
                  qs = np.asarray(total_q).flatten()
                  if train_steps!=0:
                    self.losses.append(losses.mean())
                    self.q_values.append(qs)
                  else:
                    self.losses.append(0)
                    self.q_values.append(0)
                  self.avg_rewards.append(avg)
                  print("\rTRAIN: Episode {c_ep}/{n_ep} [{t} sec.]|| 50 Running Avg {a}, Episode Reward {tr}, Loss {l}, Q {q}, eps {eps}".format(
                        c_ep=self.current_episode+1,
                        n_ep=episodes,
                        t= np.round(episode_time, 3),
                        a=np.round(avg, 3),
                        tr=np.round(total_reward, 3),
                        l=np.round(self.losses[self.current_episode], 3),
                        q=np.round(qs.mean(),3),
                        eps=np.round(self.epsilon, 3)
                    ),flush=True, end="")
                  self.save_path = "{}/{}_ep_{}.pt".format(self.path,self.name,self.current_episode+1)
                  self.memory.save("{}/{}_memory.pt".format(self.path,self.name))
                  torch.save(self.q_network.state_dict(), self.save_path)
                  if self.use_wandb:
                    if self.rewards[t]==200 and alert:
                      wandb.alert(title="Reached 200", 
                      text="{} Agent has reached 200 points!!!".format("DQN Classical ER"))
                    alert= False
                    self.run.log({"Total Train Rewards":self.rewards[t],
                      "Average Loss Per Episode": self.losses[t],
                      "Running Average Rewards (50)": self.avg_rewards[t],
                      "Q-values": wandb.Histogram(self.q_values[t]),
                      "Average Q-values": qs.mean(),
                      "Episode Time":episode_time,
                      "Train Episode": t})
                    self.run.save("{}/*pt".format(self.path))
                  self.current_episode+=1
                  break
              s1 = s2
              if self.counter%self.update_model==0:
                loss, q = self.learn()
                if loss != None:
                    total_loss.append(loss)
                    for j in q:
                      total_q.append(j)
                    train_steps+=1
              if self.counter %self.update_target==0:
                  self.sync_weights()
              steps+=1
        print("Total Average Training Reward: ", np.mean(np.asarray(self.rewards)))
        self.run.log({"Total Average Training Reward": np.mean(np.asarray(self.rewards))})
        
    def test(self,episodes=None):
        self.test_rewards = []
        self.test_avg_rewards = []
        self.test_q_values = []
        self.sync_weights_test()
        if episodes == None:
          episodes = self.episodes
        for t in range(episodes):
          start_time = time.time()
          s1 = self.env.reset()
          steps = 1
          done = False
          total_reward = 0
          while not done:
              # self.env.render()
              action = self.get_action(s1, is_test=True)
              s2, reward, done, _ = self.env.step(action)
              total_reward+=reward
              if done:
                  episode_time = time.time() - start_time
                  # Store Trackable Stats
                  self.test_rewards.append(total_reward),
                  avg = np.mean(self.test_rewards[-50:])
                  self.test_avg_rewards.append(avg)
                  print("\rTEST: Episode {c_ep}/{n_ep} [{t} sec.]|| 50 Running Avg {a}, Episode Reward {tr}, eps {eps}".format(
                        c_ep=t+1,
                        n_ep=episodes,
                        t= np.round(episode_time, 3),
                        a=np.round(avg, 3),
                        tr=np.round(total_reward, 3),
                        eps=np.round(self.epsilon, 3)
                    ),flush=True, end="")
                  if self.use_wandb:
                    self.run.log({"Total Test Rewards": self.test_rewards[t], "Test Episode": t})
                  break
              s1 = s2
              steps+=1


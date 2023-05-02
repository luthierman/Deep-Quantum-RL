from DQN import *
from hyperparameters import *

torch.cuda.empty_cache()
config = {
  "learning_rate": LR,
  "learning_rate_in": .01,
  "learning_rate_out": .1,
  "n_episodes": 200,
  "batch_size": BATCH_SIZE,
  "gamma": GAMMA,
  "epsilon": EPSILON,
  "epsilon_decay": EPSILON_DECAY,
  "epsilon_min": EPSILON_MIN,
  "buffer_size": 10000,
  "update_target": 10,
  "update_model": 1,
  "train_start": TRAIN_START,
  "is_DDQN": False,
  "use_PER": True,
  "is_Reupload":True,
  "n_layers":3
}

def main():
  agent_dqn = DQN(Reupload_Net, config, "Classical-3-layer-agent", True)
  agent_dqn.train()
# agent_dqn_er.test()
  agent_dqn.run.finish()

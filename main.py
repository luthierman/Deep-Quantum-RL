from DQN import *


def main():
  agent_dqn = DQN(Reupload_Net, config, "Classical-3-layer-agent", True)
  agent_dqn.train()
# agent_dqn_er.test()
  agent_dqn.run.finish()

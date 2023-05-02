import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

class Dense_Net(nn.Module):
    def __init__(self,  n_layers=5):
        super(Dense_Net, self).__init__()
        self.linear_in = nn.Linear(4, 64)
        self.layers = nn.ModuleList()
        for i in range(n_layers-2):
          self.layers.append(nn.Linear(64, 64))
       
        self.linear_out = nn.Linear(64,2)
        nn.init.xavier_normal_(self.linear_in.weight)
        for l in self.layers:
          nn.init.xavier_normal_(l.weight)
        nn.init.xavier_normal_(self.linear_out.weight)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        for l in self.layers:
          x = F.relu(l(x))
        x = self.linear_out(x)
        return x

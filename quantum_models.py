import pennylane as qml
from pennylane import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt

class Reupload_Net(nn.Module):
    def __init__(self, n_layers=3, n_qubits=4):
      super(Reupload_Net, self).__init__()
      self.n_layers = n_layers
      self.n_qubits = n_qubits
      self.weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3), 
                            "w_in": (self.n_layers,self.n_qubits) 
                            }
      dev = qml.device("default.qubit.autograd", wires=self.n_qubits)
      def layer( W):
        for i in range(self.n_qubits):
          qml.RY(W[i,1], wires=i)
          qml.RZ(W[i,2], wires=i)
      @qml.qnode(dev, interface='torch', diff_method="adjoint")
      def circuit(weights, inputs, w_in):
          # W: Layer Variable Parameters, s: State Variable
          for i in range(self.n_layers):
            # Weighted Input Encoding
            qml.AngleEmbedding(torch.tanh(torch.multiply(inputs,w_in[i])), wires=range(self.n_qubits), rotation="X")
            # Parameterized Layer
            layer(weights[i])
            # Entangling
            for j in range(n_qubits):
                qml.CNOT(wires=[j%self.n_qubits,(j+1)%self.n_qubits])
          return [qml.expval(qml.PauliZ(0)@qml.PauliZ(1) ), 
                  qml.expval(qml.PauliZ(2)@qml.PauliZ(3) )]
      self.circuit = circuit
      self.qvc = qml.qnn.TorchLayer(circuit, self.weight_shapes)
      nn.init.uniform(self.qvc.weights, a=0, b=torch.pi)
      d = self.qvc.state_dict()
      d["w_in"] = torch.autograd.Variable(torch.ones(n_layers,n_qubits,requires_grad=True))
      self.w_out = nn.Parameter(torch.ones(1,2),requires_grad=True)
      self.qvc.load_state_dict(d)
      self.thetas = self.qvc.state_dict()["weights"]
      
    def forward(self, x):
      # rescaled out 
      out =  (self.qvc(x)+1)/2
      return torch.mul(out, self.w_out)
  
      
    def save_visual(self,path):
        self.fig.savefig(path)
    def print_circuit(self):
        self.drawer = qml.draw(self.circuit, show_all_wires=True)(
                                        self.thetas, torch.tensor([0,0,0,0]), self.qvc.state_dict()['w_in'])
        print(self.drawer)
    def print_circuit_mpl(self):
        qml.drawer.use_style("black_white_dark")
        self.drawer = qml.draw_mpl(self.circuit,fontsize="xx-large", expansion_strategy="device")(self.thetas, torch.tensor([0,0,0,0]),self.qvc.state_dict()['w_in'])


        
Reupload_Net(5).print_circuit()

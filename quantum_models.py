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
      dev = qml.device("qulacs.simulator", wires=self.n_qubits)
      def layer( W):
        for i in range(self.n_qubits):
          qml.RY(W[i,1], wires=i)
          qml.RZ(W[i,2], wires=i)
      @qml.qnode(dev, interface='torch')
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
      self.qvc = qml.qnn.TorchLayer(circuit, self.weight_shapes,
                                    init_method={"weights":torch.nn.init.uniform,
                                                 "w_in": torch.ones(n_layers,n_qubits)})
      # print(self.qvc.state_dict())
      # d = self.qvc.state_dict()
      # d["w_in"] = torch.autograd.Variable()
      # self.qvc.load_state_dict(d)

      self.w_out = nn.Parameter(torch.ones(1,2),requires_grad=True)
      
    def forward(self, x):
      # rescaled out 
      out =  (self.qvc(x)+1)/2
      return torch.mul(out, self.w_out)
      
    def save_visual(self,path):
        self.fig.savefig(path)
    def print_circuit(self):
        self.drawer = qml.draw(self.circuit, show_all_wires=True)(
                                        self.qvc.state_dict()["weights"], torch.tensor([0,0,0,0]), self.qvc.state_dict()['w_in'])
        print(self.drawer)
    def print_circuit_mpl(self):
        qml.drawer.use_style("black_white_dark")
        self.drawer = qml.draw_mpl(self.circuit,fontsize="xx-large", expansion_strategy="device")(self.qvc.state_dict()["weights"], torch.tensor([0,0,0,0]),self.qvc.state_dict()['w_in'])


        
# Reupload_Net(5).print_circuit()

class IBS_Net(nn.Module):
  def __init__(self, n_layers=3, n_qubits=4, use_cuda =False):
      super(IBS_Net, self).__init__()
      self.L = 2
      self.M = 2
      self.n = 4
      self.use_cuda = use_cuda
      dev = qml.device("qulacs.simulator", wires=self.n)
      self.thetas = self.initialize()
      self.weight_shapes = {"weights": self.thetas.shape}
      @qml.qnode(dev, interface='torch')
      def circuit(inputs, weights):
        qml.AngleEmbedding(inputs*(np.pi/4), 
                          wires = range(self.n),
                          rotation="Y")
        for i in range(0,2*self.M*self.L,2*self.L):
          self.block(weights[i:i+2*self.L],self.n, self.L, self.M)
        return [qml.expval(qml.PauliY(ind)) for ind in range(4)]
        
      self.circuit = circuit
      self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)#.to('cuda' if use_cuda else 'cpu')
      self.linear1 = nn.Linear(self.n, 2)#.to('cuda' if use_cuda else 'cpu')
      nn.init.xavier_normal_(self.linear1.weight)
      self.linear1.bias.data.zero_()
      new_state_dict = self.qlayer.state_dict()
      new_state_dict['weights'] = torch.tensor(self.thetas)
      self.qlayer.load_state_dict(new_state_dict)
      self.print_circuit()

  def initialize(self):
      parameters = np.zeros((self.M,2*self.L,self.n))
      for m in range(self.M):
        stack = []
        for l in range(self.L):
          for i in range(self.n):
            theta = np.random.uniform(0,2*np.pi)
            stack.append(theta)
            parameters[m,l,i] = theta
        for l in range(self.L,2*self.L):
          for i in range(self.n):
            parameters[m,l,self.n-i-1] = stack.pop()
      return qml.math.concatenate(parameters)
  def block(self,parameters,n, L, M):
      U_m = []
      ops = [qml.RX, qml.RY, qml.RZ]
      for l in range(L):
        for i in range(n):
          U = random.choice(ops)
          U_m.append(U)
          U(parameters[l,i], wires=i)
        for i in range(n-1):
          qml.CZ(wires=[(i)%n,(i+1)%n])
      for l in range(L, 2*L):
        for i in range(n-1):
          qml.CZ(wires=[n-i-2,n-i-1])
        for i in range(n):
          U = U_m.pop()
          qml.adjoint(U(parameters[l,n-i-1], wires=n-i-1))
  def forward(self, x):
        x = torch.tensor(x)
        x = x.float()#.to('cuda' if self.use_cuda else 'cpu')
        x.requires_grad = True
        x = self.qlayer(x)
        x = self.linear1(x)
        return x.float()

  def print_circuit(self):
        self.thetas = self.qlayer.state_dict()["weights"]
        drawer = qml.draw(self.circuit)(torch.tensor([0,0,0,0]),
                                        self.thetas)
        print(drawer)

  def print_circuit_mpl(self):
        self.thetas = self.qlayer.state_dict()["weights"]
        qml.drawer.use_style("black_white_dark")

        drawer = qml.draw_mpl(self.circuit,fontsize="xx-large", expansion_strategy="device")(torch.tensor([0,0,0,0]),
                                        self.thetas)
        return drawer

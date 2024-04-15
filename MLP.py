import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from activations import *
from dibujar import MLP_binary_draw

class DenseNet:
    def __init__(self, layers_dim, hidden_activation=tanh, output_activation=logistic):
        # Atributes
        self.L = len(layers_dim)
        self.w = [None] * (self.L) 
        self.b = [None] * (self.L)
        self.f = [None] * (self.L)

        for l in range(1, self.L):
            self.w[l] = -1 * 2 * np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = -1 * 2 * np.random.rand(layers_dim[l], 1)

            if l == self.L-1:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation
            
    def predict(self, X):
        A = np.array(X)
        for l in range(1, self.L):
            Z = self.w[l] @ A + self.b[l]
            A = self.f[l](Z)
        return A

    def fit(self, X, Y, epoch=500, learning_rate=0.1):
        p = X.shape[1]
        for _ in range(epoch):
            # Containers
            A = [None] * self.L
            dA = [None] * self.L
            local_gradient = [None] * self.L
            # Propagation
            A[0] = np.array(X)
            for l in range(1, self.L):
                Z = self.w[l] @ A[l-1] + self.b[l]
                A[l], dA[l] = self.f[l](Z, derivative=True)
            # Backpropagation
            for l in range(self.L-1, 0, -1):
                if l == self.L - 1:
                    local_gradient[l] = -(Y - A[l]) * dA[l]
                else:
                    local_gradient[l] = (self.w[l+1].T @ local_gradient[l+1]) * dA[l]
            # Gradient Descent
            for l in range(1, self.L):
                self.w[l] -= (learning_rate/p) * local_gradient[l] @ A[l-1].T
                self.b[l] -= (learning_rate/p) * np.sum(local_gradient[l])
    
def create(filename,graficName,neurons,layers):
    df = pd.read_csv(filename)
    data = df.to_numpy().T
    X = data[:-1]
    Y = data[-1:]
    net = DenseNet((2, neurons, layers, 1))
    net.fit(X,Y, epoch=10000)
    MLP_binary_draw(X, Y, net,graficName)
    
create('moons.csv','DataSet: MOONS',8,4)
create('blobs.csv','DataSet: BLOBS',8,4)    
create('concentlite.csv','DataSet: concentlite.csv',4,6)


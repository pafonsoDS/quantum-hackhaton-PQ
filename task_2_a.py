import numpy as np
import matplotlib.pyplot as plt


from qadence import zero_state, one_state, product_state, ghz_state, random_state
from qadence import RX, RY, RZ, PI, CNOT
from qadence import X, Y, Z
from qadence import chain

import torch


Data = np.loadtxt("Los_Quantums\\Datasets\\dataset_2_a.txt")
tensor_Data = torch.tensor(Data) 

x_train, y_train = tensor_Data[:, 0], tensor_Data[:, 1]

# plt.plot(x_train.numpy(), y_train.numpy(), label="Dados") 
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

from qadence import RX, FeatureParameter, VariationalParameter, kron
from sympy import cos, sin, exp
from qadence.draw import display
from qadence import feature_map
from qadence import hea

# --------------------------------------------------------------#
## Feature map
n_qubits = 2

x = FeatureParameter("x")

A1 = VariationalParameter("A1")
A2 = VariationalParameter("A2")

theta1 = VariationalParameter("theta1")
theta2 = VariationalParameter("theta2")

# --------------------------------------------------------------#
## ansatz

fm = kron(RX(0, x)@ ( RX(1, 2*x)) )
ansatz = RX(0, theta1) @ RX(1, theta2)

block = fm * ansatz
#display(block)
# display(ansatz)


from qadence import QuantumCircuit


circuit = QuantumCircuit(n_qubits, block)

 # ---------------------

from qadence import Z, QuantumModel, add

obs = A1*Y(0)+A2*Y(1) # add(Y(i) for i in range(n_qubits))
#print(obs)
model = QuantumModel(circuit, observable = obs)

 # ---------------------
values = {"x": x_train}

y_pred_initial = model.expectation(values).squeeze().detach()


# Computes the Mean Squared Error between every entry of two tensors
criterion = torch.nn.MSELoss()

def loss_fn(x_train, y_train):
    output = model.expectation({"x": x_train}).squeeze()
    loss = criterion(output, y_train)
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

n_epochs = 100

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_fn(x_train, y_train)
    loss.backward()
    optimizer.step()

y_pred_final = model.expectation({"x": x_train}).squeeze().detach()

plt.plot(x_train, y_pred_initial, label = "Initial prediction")
plt.plot(x_train, y_pred_final, label = "Final prediction")
plt.scatter(x_train, y_train, label = "Training points")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

print (model.vparams)
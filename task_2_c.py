import numpy as np
import matplotlib.pyplot as plt


from qadence import zero_state, one_state, product_state, ghz_state, random_state
from qadence import RX, RY, RZ, PI, CNOT
from qadence import X, Y, Z
from qadence import chain

import torch


Data = np.loadtxt("Los_Quantums\\Datasets\\dataset_2_c.txt")

tensor_Data = torch.tensor(Data) #, dtype=torch.float32)

x_train, y_train = tensor_Data[:, 0], tensor_Data[:, 1]

from qadence import RX, FeatureParameter, VariationalParameter, kron
from sympy import cos, sin, exp
from qadence.draw import display
from qadence import feature_map
from qadence import hea

# --------------------------------------------------------------#
## Feature map
n_qubits = 3

x = FeatureParameter("x")

A1 = VariationalParameter("A1")
A2 = VariationalParameter("A2")
A3 = VariationalParameter("A3")
theta1 = VariationalParameter("theta1")
theta2 = VariationalParameter("theta2")
theta3 = VariationalParameter("theta3")
omega1 = VariationalParameter("omega1")
omega2 = VariationalParameter("omega2")
omega3 = VariationalParameter("omega3")


fm = kron( RX(0, omega1* x) @ ( RX(1, omega2*x)) @  RX(2,omega3*x)) 
ansatz = RX(0, theta1) @ RX(1, theta2) @ RX(2, theta3) 

block = fm * ansatz
#display(block)

from qadence import QuantumCircuit


circuit = QuantumCircuit(n_qubits, block)

 # ---------------------

from qadence import Z, QuantumModel, add

obs = A1*Y(0)+A2*Y(1) + A3*Y(2) 
# print(obs)
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

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

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

print(model.vparams)

Data = np.loadtxt("Los_Quantums\\Datasets\\dataset_2_c_test.txt")

x_train1 = torch.tensor(Data)

X_1= x_train1[:,0]
X_2= x_train1[:,1]
print(X_1)
print(X_2)

y_pred_1 = model.expectation({"x": X_1}).squeeze().detach().cpu().numpy()
y_pred_2 = model.expectation({"x": X_2}).squeeze().detach().cpu().numpy()

plt.plot(x_train, y_pred_final, label = "Final prediction")
plt.plot(X_1, y_pred_1,".", label = "Final prediction")
plt.plot(X_2, y_pred_2,".", label = " dataset_2_test")
plt.show()


import pandas as pd

df = pd.DataFrame({
    "x": np.concatenate([X_1.numpy(), X_2.numpy()]),
    "y": np.concatenate([y_pred_1, y_pred_2])
})

df.to_csv("solution_2_c.csv", index=False)


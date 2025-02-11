import numpy as np
import matplotlib.pyplot as plt


from qadence import zero_state, one_state, product_state, ghz_state, random_state
from qadence import RX, RY, RZ, PI, CNOT
from qadence import X, Y, Z
from qadence import chain
import graphviz
import torch

# Load DATA
Data = np.loadtxt("Los_Quantums\\Datasets\\dataset_1_a.txt")

tensor_Data = torch.tensor(Data) #, dtype=torch.float32

x_train, y_train = tensor_Data[:, 0], tensor_Data[:, 1]


from qadence import RX, FeatureParameter, VariationalParameter, kron
from sympy import cos, sin, exp
from qadence.draw import display
from qadence import hea, QuantumCircuit
from qadence.draw import display
from qadence import Z, QuantumModel, add

from qadence import feature_map

n_qubits = 1
x = FeatureParameter("x")

fm = RX(0, x) # Feature map


theta = VariationalParameter("theta")
ansatz = RX(0, theta) # ansatz

block = fm * ansatz

circuit = QuantumCircuit(n_qubits, block)

obs = Z(0) 

model = QuantumModel(circuit, observable = obs)

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

# plt.plot(x_train, y_pred_initial, label = "Initial prediction")
# plt.plot(x_train, y_pred_final, label = "Final prediction")
# plt.scatter(x_train, y_train, label = "Training points")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.legend()
# plt.show()

print("Theta: ", model.vparams['theta'] )
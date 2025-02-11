import torch
from qadence import *
from qadence import feature_map, hea, chain
from qadence import QNN, QuantumCircuit, Z
from qadence.types import BasisSet, ReuploadScaling
from qadence import add

k = 2.3

class ScaledModel(torch.nn.Module):
    def __init__(self, qnn, x_min=0.0, x_max=10.0):
        super().__init__()
        self.qnn = qnn
        self.x_min = x_min
        self.x_max = x_max
        self.input_scale = 2.0 / (x_max - x_min)
        self.output_scale = torch.nn.Parameter(torch.tensor(1.0))  #train output scaling

    def forward(self, x):
        #scale input to [-1, 1]!!
        x_scaled = (x - self.x_min) * self.input_scale - 1.0
        qnn_output = self.qnn(x_scaled)
        return self.output_scale * qnn_output  #output scaling
    
################################################################################
def calc_deriv(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(inputs),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad

def calc_second_deriv(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    first_deriv = calc_deriv(outputs, inputs)
    second_deriv = calc_deriv(first_deriv, inputs)
    return second_deriv

criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    model_output = model(inputs)
    deriv_model = calc_second_deriv(model_output, inputs)
    deriv_exact = -k * model_output
    ode_loss = criterion(deriv_model, deriv_exact)

    #set f(0) = 1 constraint
    boundary_input = torch.tensor([[0.0]], dtype=inputs.dtype, requires_grad=True)
    boundary_output = model(boundary_input)
    boundary_f_loss = criterion(boundary_output, torch.tensor([[1.0]], dtype=inputs.dtype))

    #f'(0) = 1.2
    boundary_deriv = calc_deriv(boundary_output, boundary_input)
    boundary_deriv_loss = criterion(boundary_deriv, torch.tensor([[1.2]], dtype=inputs.dtype))

    return ode_loss + boundary_f_loss + boundary_deriv_loss

n_qubits = 5
depth = 3

#fourier fm
fm = feature_map(
    n_qubits=n_qubits,
    param="x",
    fm_type=BasisSet.FOURIER,
    reupload_scaling=ReuploadScaling.TOWER, #keep tomwer
)

ansatz = hea(n_qubits=n_qubits, depth=depth) #hea ansatz
observable = add(Z(i) for i in range(n_qubits))

circuit = QuantumCircuit(n_qubits, fm * ansatz)

qnn = QNN(circuit=circuit, observable=observable, inputs=["x"])

model = ScaledModel(qnn, x_min=0.0, x_max=10.0)

n_epochs = 500
n_points = 225

#fixed Chebyshev points in [0, 10]
cheb_points = torch.cos(torch.pi * (2*torch.arange(n_points)+1) / (2*n_points))  #chebyshev nodes in [-1,1]
x_train_fixed = 5.0 * (cheb_points + 1.0)  #scale to [0,10]
x_train_fixed = x_train_fixed.unsqueeze(1).requires_grad_(True) #to take derivative

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  #lower learning rate

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    #use fixed Chebyshev collocation points
    loss = loss_fn(model, x_train_fixed)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


import matplotlib.pyplot as plt
import numpy as np

def f_exact(x: torch.Tensor) -> torch.Tensor:
    return np.cos(np.sqrt(2.3)*x)+(1.2/np.sqrt(2.3))*np.sin(np.sqrt(2.3)*x)

xmin = 0; xmax = 10
x_test = torch.arange(xmin, xmax, step = 0.01).unsqueeze(1)

result_exact = f_exact(x_test).flatten()

result_model = model(x_test).flatten().detach()

plt.plot(x_test, result_exact, label = "Exact solution")
plt.plot(x_test, result_model, label = " Trained model")
plt.show()


Data = np.loadtxt("Datasets\\dataset_3_test.txt")

x_train = torch.tensor(Data) #, dtype=torch.float32)
print(len(x_train))
result_model = model(x_train).flatten().detach()

plt.plot(x_test, result_exact,"-", label = "Exact solution")
plt.plot(x_train, result_model,".", label = " dataset_3_test")
plt.show()


import pandas as pd

df = pd.DataFrame({"t": x_train, "x": result_model})
df.to_csv("solution_3_a.csv", index=False)

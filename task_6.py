import pandas as pd
import torch
from qadence import *
from qadence.draw import display
import random as rd
import matplotlib.pyplot as plt
import torch

###############################################################################################################

data6 = pd.read_csv("Datasets/dataset_6.txt", header = None, delim_whitespace = True) #fixing problems with header
data6.columns = ['t', 'x', 'y']

t_data = torch.tensor(data6['t'].values, dtype=torch.float64).unsqueeze(1)
#so that we have column vector w shape (N, 1)

population_data = torch.tensor(data6[['x', 'y']].values, dtype=torch.float64)

##############################################################################

alpha = torch.nn.Parameter(torch.tensor(rd.random(), dtype=torch.float32))
beta  = torch.nn.Parameter(torch.tensor(rd.random(), dtype=torch.float32))
gamma = torch.nn.Parameter(torch.tensor(rd.random(), dtype=torch.float32))
delta = torch.nn.Parameter(torch.tensor(rd.random(), dtype=torch.float32))

def calc_deriv_vector(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    calculate time deriv for both components, ie
    for a model output f(t) = [x(t), y(t)] with shape (n_points,2),
    returns a tensor of the same shape with d/dt ([x,y])
    """
    derivs = []
    #loops over each output dimension and compute derivative wrt inputs
    for i in range(outputs.shape[1]):
        grad_i = torch.autograd.grad(
            outputs=outputs[:, i],
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        derivs.append(grad_i)
    #cat deriv (second dim)
    return torch.cat(derivs, dim=1)

#mse
criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, inputs: torch.Tensor, alpha, beta, gamma, delta) -> torch.Tensor:
    """
    we combine two losses:
      (a) ODE loss: Forces the QNN output to satisfy the LV equations (physics informed)
      (b) N Boundary conditions for points, maybe not consider all points
    """
    #qnn output: shape (n_points,2); first column is x(t), second is y(t)
    outputs = model(inputs)
    #get both time derivs
    deriv = calc_deriv_vector(outputs, inputs)  #shape (n_points,2)
    dxdt_pred = deriv[:, 0:1]
    dydt_pred = deriv[:, 1:2]
    
    #extract predictions for x and y.
    x_pred = outputs[:, 0:1]
    y_pred = outputs[:, 1:2]
    
    #compute exact lotka volterra pd
    dxdt_exact = alpha * x_pred - beta * x_pred * y_pred
    dydt_exact = delta * x_pred * y_pred - gamma * y_pred
    
    #PIL: compare QNN time derivs with the LV model.
    ode_loss = criterion(dxdt_pred, dxdt_exact) + criterion(dydt_pred, dydt_exact) #physics informed loss basically
    
    ###############################################################################################################
    #naively we have N = 365 boundary conditions
    
    data_loss = criterion(model(t_data), population_data)
    
    ###############################################################################################################
    #total loss is the sum of ODE loss and N boundary condition losses.
    total_loss = ode_loss + data_loss #we could and SHOULD weight them!!
    return total_loss

# init qnn for two outputs, parametrized by t
n_qubits = 4
depth = 3

#fm w/ t fourier
fm = feature_map(
    n_qubits = n_qubits,
    param = "t",  # note: we use "t" as the input variable name
    fm_type = BasisSet.FOURIER,
    reupload_scaling = ReuploadScaling.TOWER,
)

#define hea ansatz
ansatz = hea(n_qubits = n_qubits, depth = depth)

observable_x = add(Z(i) for i in range(n_qubits))
observable_y = add(X(i) for i in range(n_qubits))
observables = [observable_x, observable_y]

#init the quantum circuit.
circuit = QuantumCircuit(n_qubits, fm*ansatz)

model = QNN(circuit=circuit, observable=observables, inputs=["t"])

#train
n_epochs = 50
n_points = 365

tmin = 0.0
tmax = 365.0

#include both the QNN parameters and the LV parameters in the optimizer.
optimizer = torch.optim.Adam(list(model.parameters()) + [alpha, beta, gamma, delta], lr=0.005)

#now just the training loop remaining
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    #sample training points uniformly in [tmin, tmax]
    t_train = tmin + (tmax - tmin) * torch.rand(n_points, requires_grad=True)
    t_train = t_train.unsqueeze(1)  #shape (n_points, 1)
    
    loss = loss_fn(model, t_train, alpha, beta, gamma, delta)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.6f}")
        print(f"LV parameters: α = {alpha.item():.4f}, β = {beta.item():.4f}, γ = {gamma.item():.4f}, δ = {delta.item():.4f}")

print("Training complete. Printing predicted x, y, and plotting vs. dataset.")
#after training the learned LV parameters we can print them


############################################################################# lets plot and compare with real distrib
#generate time points
t_values = torch.linspace(data6['t'].min(), data6['t'].max(), steps=365).unsqueeze(1)

#get model predictions
with torch.no_grad():
    predictions = model(t_values)
    
pred_x, pred_y = predictions[:, 0].detach().numpy(), predictions[:, 1].detach().numpy()

print(f"predictions, first col x, second y: \n{predictions.detach()}")

# Plot actual data
plt.figure(figsize=(10, 5))
plt.plot(data6['t'].to_numpy(), data6['x'].to_numpy(), 'bo', label="Actual Rabbits (x)")
plt.plot(data6['t'].to_numpy(), data6['y'].to_numpy(), 'ro', label="Actual Foxes (y)")

# Plot model predictions
plt.plot(t_values.numpy(), pred_x, 'b-', label="Predicted Rabbits")
plt.plot(t_values.numpy(), pred_y, 'r-', label="Predicted Foxes")

plt.xlabel("Time (t)")
plt.ylabel("Population")
plt.legend()
plt.title("Predator-Prey Population Dynamics")
plt.show()


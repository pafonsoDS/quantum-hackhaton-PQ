import torch
from qadence import *
from qadence.draw import display
import sympy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_qubits = 4
depth = 2 
k = 1.0
w1 = torch.pi
w2 = 2 * torch.pi
q_coeff = (k**2 - w1**2 - w2**2)

# Define parameters
x_param = Parameter("x", trainable=False)
y_param = Parameter("y", trainable=False)


#TRYING MORE EXPRESSIVE FM
def create_2d_feature_map():  
    fm = chain()
    #encode x and y with multiplicative interactions
    for q in range(n_qubits):
        angle = (x_param * y_param) * torch.pi #product of x and y terms to capture wave interactions
        fm *= RX(q, angle)
        #individual coordinate encoding
        fm *= RZ(q, x_param * torch.pi)
        fm *= RZ(q, y_param * torch.pi)
    return fm

#DIFFERENT ANSATZ
def create_ansatz():
    ansatz = chain()
    for d in range(depth):
        layer = chain()
        for q in range(n_qubits):
            layer *= RY(q, Parameter(f"theta_y{q}_{d}"))
            layer *= RZ(q, Parameter(f"theta_z{q}_{d}"))
        
        #parametrized entanglement (instead of fixed CNOTs)
        for q in range(n_qubits - 1):
            layer *= CRX(q, q + 1, Parameter(f"theta_crx{q}_{d}"))
            layer *= CRZ(q, q + 1, Parameter(f"theta_crz{q}_{d}"))
        
        #resid connections
        ansatz = ansatz + layer if d > 0 else layer
    return ansatz


fm = create_2d_feature_map()
ansatz = create_ansatz()
observable = add(Z(i) for i in range(n_qubits))
full_block = fm*ansatz

circuit = QuantumCircuit(n_qubits, full_block)
model = QNN(circuit, observable, inputs=[x_param, y_param]) 

############################################################################################

#pde computation
def calc_partial_deriv(output, input_var):
    grad = torch.autograd.grad(
        outputs=output,
        inputs=input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad

#loss function pde + boundary
criterion = torch.nn.MSELoss()

############################################################################################
#adaptive weight
lambda_bc = torch.tensor(1.0, requires_grad=False)
lambda_pde = torch.tensor(1.0, requires_grad=False)
    
def loss_fn(model, interior_points, boundary_points):
    global lambda_pde, lambda_bc  #declare them global
    
    x_int = interior_points[:, 0].requires_grad_(True)
    y_int = interior_points[:, 1].requires_grad_(True)
    u = model({"x": x_int, "y": y_int})
    
    #second derivatives
    du_dx = calc_partial_deriv(u, x_int)
    d2u_dx2 = calc_partial_deriv(du_dx, x_int)
    du_dy = calc_partial_deriv(u, y_int)
    d2u_dy2 = calc_partial_deriv(du_dy, y_int)
    
    #pde residual
    q = q_coeff * torch.sin(w1 * x_int) * torch.sin(w2 * y_int)
    pde_residual = d2u_dx2 + d2u_dy2 + (k**2) * u - q
    pde_loss = criterion(pde_residual, torch.zeros_like(pde_residual))
    
    #boundary points loss
    x_bdry = boundary_points[:, 0]
    y_bdry = boundary_points[:, 1]
    u_bdry = model({"x": x_bdry, "y": y_bdry})
    bc_loss = criterion(u_bdry, torch.zeros_like(u_bdry))
    
    #compute loss
    loss = lambda_pde * pde_loss + lambda_bc * bc_loss

    #adjust weights dynamically
    with torch.no_grad():  #(no gradients are computed for loss balancing)
        lambda_pde = torch.abs(pde_loss) / (torch.abs(pde_loss) + torch.abs(bc_loss) + 1e-8)
        lambda_bc = 1.0 - lambda_pde  #guarantee normaliz

    return loss

#####################################################

#train
n_epochs = 250  
n_interior = 450
n_boundary = 150
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

boundary_points = torch.cat([
    torch.cat([torch.ones(n_boundary//2, 1), torch.rand(n_boundary//2, 1)*2-1], dim=1),
    torch.cat([-torch.ones(n_boundary//2, 1), torch.rand(n_boundary//2, 1)*2-1], dim=1),
    torch.cat([torch.rand(n_boundary//2, 1)*2-1, torch.ones(n_boundary//2, 1)], dim=1),
    torch.cat([torch.rand(n_boundary//2, 1)*2-1, -torch.ones(n_boundary//2, 1)], dim=1),
], dim=0) # define boundary conditions

#training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    #generate training data
    interior_points = torch.rand(n_interior, 2)*2 - 1
    
    loss = loss_fn(model, interior_points, boundary_points)
    loss.backward()

    #Clip gradients before updating parameters
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
    ###################################################
    optimizer.step()
    
    if epoch % 20 == 0:
        with torch.no_grad():
            grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
            print(f"Epoch: {epoch}, Gradient norm: {grad_norm:.4e}") #grad norm above

########################### end of training

#################################
#load test data

input_file = "Datasets/dataset_4_test.txt" 
data = np.loadtxt(input_file, delimiter=" ")  #assuming CSV format without header

x_test = data[:, 0]
y_test = data[:, 1]

# Convert to tensor
xy_tensor = torch.tensor(np.column_stack((x_test, y_test)), dtype=torch.float32)
x_tensor = xy_tensor[:, 0].unsqueeze(1)
y_tensor = xy_tensor[:, 1].unsqueeze(1)

# Evaluate the model
with torch.no_grad():
    u_pred = model({"x": x_tensor, "y": y_tensor})

# Convert predictions to numpy
u_pred_np = u_pred.detach().cpu().numpy()

# Save to CSV
output_file = "solution_4.csv"
pd.DataFrame({"x": x_test, "y": y_test, "u": u_pred_np.flatten()}).to_csv(output_file, index=False)

# Define grid parameters for plotting
n_points = 150
x_vals = np.linspace(-1, 1, n_points)
y_vals = np.linspace(-1, 1, n_points)
X, Y = np.meshgrid(x_vals, y_vals)

# Flatten the grid for model evaluation
XY = np.stack([X.flatten(), Y.flatten()], axis=1)
xy_grid_tensor = torch.tensor(XY, dtype=torch.float32)
x_grid = xy_grid_tensor[:, 0].unsqueeze(1)
y_grid = xy_grid_tensor[:, 1].unsqueeze(1)

# Evaluate model on the grid
with torch.no_grad():
    u_grid_pred = model({"x": x_grid, "y": y_grid})

# Convert prediction to numpy and reshape
u_grid_pred_np = u_grid_pred.detach().cpu().numpy().reshape(n_points, n_points)

# Plot heatmap
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u_grid_pred_np, levels=100, cmap='viridis')
plt.colorbar(contour, label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Heatmap of QNN Predicted Solution')
plt.show()

print(f"Results saved to {output_file}")

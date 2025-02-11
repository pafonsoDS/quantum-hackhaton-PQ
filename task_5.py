import torch
from qadence import *
from qadence import feature_map, hea, chain
from qadence import QNN, QuantumCircuit, Z
from qadence.types import BasisSet, ReuploadScaling
from qadence import add

# Constante da equação diferencial
k = -2.5

class ScaledModel(torch.nn.Module):
    def __init__(self, qnn, x_min=0.0, x_max=10.0):
        super().__init__()
        self.qnn = qnn
        self.x_min = x_min
        self.x_max = x_max
        self.input_scale = 2.0 / (x_max - x_min)
        self.output_scale = torch.nn.Parameter(torch.tensor(1.0))  # Train output scaling

    def forward(self, x):
        # Escala a entrada para [-1, 1]
        x_scaled = (x - self.x_min) * self.input_scale - 1.0
        return self.output_scale * self.qnn(x_scaled)  # Aplicar escala de saída

################################################################################

def parameter_shift_derivative(model, x, shift=torch.pi / 2):
    """
    Calcula df/dt usando a Parameter Shift Rule (PSR).
    """
    derivatives = []
    
    for i in range(n_qubits):
        # Avaliações do circuito em Rx(t ± π/2) para o qubit i
        x_plus = x.clone()
        x_plus += shift  # t -> t + π/2

        x_minus = x.clone()
        x_minus -= shift  # t -> t - π/2

        f_plus = model(x_plus).detach()
        f_minus = model(x_minus).detach()

        # Aplicar PSR: (f+ - f-) / 2
        derivative_i = 0.5 * (f_plus - f_minus)
        derivatives.append(derivative_i)

    # Somamos as contribuições dos n_qubits
    return torch.stack(derivatives).sum(dim=0)

criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    model_output = model(inputs)

    # Derivada usando a regra de Parameter Shift
    deriv_model = parameter_shift_derivative(model, inputs)

    # Equação diferencial dx/dt = kx
    deriv_exact = k * model_output
    ode_loss = criterion(deriv_model, deriv_exact)

    # Condição inicial: f(0) = 1
    boundary_input = torch.tensor([[0.0]], dtype=inputs.dtype)
    boundary_output = model(boundary_input)
    boundary_f_loss = criterion(boundary_output, torch.tensor([[1.0]], dtype=inputs.dtype))

    return ode_loss + boundary_f_loss

################################################################################

n_qubits = 2
depth = 2

# Feature map usando apenas Rx(t) em cada qubit
fm = feature_map(
    n_qubits=n_qubits,
    param="x",
    fm_type=BasisSet.FOURIER,
    reupload_scaling=ReuploadScaling.TOWER,
)

ansatz = hea(n_qubits=n_qubits, depth=depth)  # Ansatz do circuito
observable = add(Z(i) for i in range(n_qubits))  # Medida

circuit = QuantumCircuit(n_qubits, fm * ansatz)

qnn = QNN(circuit=circuit, observable=observable, inputs=["x"])

model = ScaledModel(qnn, x_min=0.0, x_max=2.0)



################################################################################

n_epochs = 500
n_points = 225

# Pontos de treinamento usando distribuição de Chebyshev
cheb_points = torch.cos(torch.pi * (2 * torch.arange(n_points) + 1) / (2 * n_points))  
x_train_fixed = 5.0 * (cheb_points + 1.0)  # Escala para [0,10]
x_train_fixed = x_train_fixed.unsqueeze(1)  # Preparar para batch training

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    loss = loss_fn(model, x_train_fixed)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

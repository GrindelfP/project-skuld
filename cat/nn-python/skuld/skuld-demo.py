#################################################

        #################################
    ####### SKULD NNI LIBRARY (DEMO ONLY) #######
           ###### version May 25 #######
        #################################

      ###### a neural network based ######
      ### numerical integration library ###
#################################################

from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mpmath import polylog
from sklearn.model_selection import train_test_split
import time
import itertools

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.input_hidden_layer = nn.Linear(input_size, hidden_size)
        self.sigmoid_activation = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size, 1)
        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        x = self.input_hidden_layer(x)
        x = self.sigmoid_activation(x)
        x = self.output_layer(x)
        return x

    def compile(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def compile_default(self, learning_rate: float) -> None:
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def train(self, x_train, y_train, epochs, verbose=True):
        loss_history = []
        start_time = time.time()
        for epoch in range(epochs):
            predictions = self(x_train)
            loss = self.criterion(predictions, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            if verbose:
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')
        total_time = time.time() - start_time
        print(f'Training done! Time elapsed: {total_time:.2f} seconds')
        return loss_history

    def test(self, x_test, y_test):
        with torch.no_grad():
            predictions = self(x_test)
            loss = self.criterion(predictions, y_test)
        return loss.item()

    def predict_with(self, x_test):
        with torch.no_grad():
            prediction = self(x_test)
        return prediction

    def extract_params(self):
        b1 = self.input_hidden_layer.bias.detach().numpy()
        w1 = self.input_hidden_layer.weight.detach().numpy()
        b2 = self.output_layer.bias.detach().numpy()
        w2 = self.output_layer.weight.detach().numpy().flatten()
        return b1, w1, b2, w2

def init_model(input_size: int, hidden_size: int) -> MLP:
    return MLP(input_size=input_size, hidden_size=hidden_size)

def split_data(X: Tensor, y: Tensor, test_size: float, shuffle: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)

class NeuralNumericalIntegration:
    @staticmethod
    def calculate2(alphas, betas, network_params):
        alpha1, alpha2, beta1, beta2 = alphas[0], alphas[1], betas[0], betas[1]
        B1, W1, B2, W2 = network_params

        def Phi_j(alp1, bt1, alp2, bt2, b1, w1_1, w1_2):
            term_1 = polylog(2, -np.exp(-b1 - w1_1 * alp1 - w1_2 * alp2))
            term_2 = polylog(2, -np.exp(-b1 - w1_1 * alp1 - w1_2 * bt2))
            term_3 = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * alp2))
            term_4 = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * bt2))
            return term_1 - term_2 - term_3 + term_4

        integral_sum = 0
        for w2_j, w1_1j, w1_2j, b1_j in zip(W2, W1[:, 0], W1[:, 1], B1):
            phi_j = Phi_j(alpha1, beta1, alpha2, beta2, b1_j, w1_1j, w1_2j)
            summ = w2_j * ((beta1 - alpha1) * (beta2 - alpha2) + phi_j / (w1_1j * w1_2j))
            integral_sum += summ

        return B2 * (beta1 - alpha1) * (beta2 - alpha2) + integral_sum

    @staticmethod
    def calculate1(alphas, betas, network_params):
        alpha, beta = alphas[0], betas[0]
        B1, W1, B2, W2 = network_params
        W1 = W1.flatten()

        def Phi_j(alp, bt, b1, w1):
            term_alpha = polylog(1, -np.exp(-b1 - w1 * alp))
            term_beta = polylog(1, -np.exp(-b1 - w1 * bt))
            return term_alpha - term_beta

        integral_sum = 0
        for w2_j, w1_j, b1_j in zip(W2, W1, B1):
            phi_j = Phi_j(alpha, beta, b1_j, w1_j)
            integral_sum += w2_j * ((beta - alpha) + phi_j / w1_j)
        return B2 * (beta - alpha) + integral_sum

    @staticmethod
    def integrate(model, alphas, betas, n_dims):
        network_params = model.extract_params()
        if n_dims == 1:
            return NeuralNumericalIntegration.calculate1(alphas, betas, network_params)
        elif n_dims == 2:
            return NeuralNumericalIntegration.calculate2(alphas, betas, network_params)
        else:
            print("Integration of functions with dimensions higher than 2 is not implemented!")
            return None

def generate_data(func, lower, upper, n_samples=100, n_dim=1, *func_args):
    if n_dim == 1:
        X = torch.linspace(lower[0], upper[0], n_samples).view(n_samples, 1)
        y = func(X, *func_args).view(n_samples, 1)
    else:
        ranges = [torch.linspace(lower[n], upper[n], n_samples).tolist() for n in range(n_dim)]
        combinations = list(itertools.product(*ranges))
        X = torch.tensor(combinations, dtype=torch.float32)
        y = func(X, *func_args).view(-1, 1)
    return X, y

def generate_data_uniform(func, lower, upper, n_samples=100, n_dim=1, *func_args):
    if n_dim == 1:
        X = torch.rand(n_samples, 1) * (upper[0] - lower[0]) + lower[0]
        y = func(X, *func_args).view(n_samples, 1)
    else:
        X = torch.rand(n_samples, n_dim)
        for i in range(n_dim):
            X[:, i] = X[:, i] * (upper[i] - lower[i]) + lower[i]
        y = func(X, *func_args).view(n_samples, 1)
    return X, y

def scale_data(
        X_init: torch.Tensor, y_init: torch.Tensor, frange: tuple[int, int] = (0, 1)
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(X_init, torch.Tensor) or not isinstance(y_init, torch.Tensor):
        raise TypeError("Input X_init and y_init must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor (m, n_dim).")
    if y_init.ndim != 2 or y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor (m, 1).")
    if X_init.shape[0] != y_init.shape[0]:
        raise ValueError("X_init and y_init must have the same number of rows (m).")
    min_val_x = torch.min(X_init, dim=0).values
    max_val_x = torch.max(X_init, dim=0).values
    min_val_y = torch.min(y_init)
    max_val_y = torch.max(y_init)
    scaled_X = (frange[1] - frange[0]) * (X_init - min_val_x) / (max_val_x - min_val_x) + frange[0]
    scaled_y = (frange[1] - frange[0]) * (y_init - min_val_y) / (max_val_y - min_val_y) + frange[0]
    return scaled_X, scaled_y


def descale_result(
        nni_scaled: float, X_init: torch.Tensor, y_init: torch.Tensor, frange: tuple[int, int] = (0, 1), n_dim: int = 1
) -> float:
    if not isinstance(X_init, torch.Tensor) or not isinstance(y_init, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor.")
    if y_init.ndim != 1 and y_init.ndim != 2:
        raise ValueError("y_init must be a 2D tensor with shape (num_of_points,1).")
    if y_init.ndim == 2 and y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor with shape (num_of_points,1).")
    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values
    f_min = torch.min(y_init)
    f_max = torch.max(y_init)
    frange_size = frange[1] - frange[0]
    VS = torch.prod(x_max - x_min)
    VSS = frange_size ** n_dim
    return (nni_scaled * (VS * (f_max - f_min) / (VSS * frange_size)) + (
            f_min - (f_max - f_min) / frange_size * frange[0]) * VS).item()

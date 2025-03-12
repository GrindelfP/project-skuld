import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
from mpmath import polylog
from sklearn.model_selection import train_test_split
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split
import time
import itertools
from sklearn.preprocessing import MinMaxScaler


class MLP(nn.Module):
    """
        Class defining neural network approximator for the integrand.
    """

    def __init__(self, input_size, hidden_size):
        """
            Main constructor.

            :param input_size: input layer size (number of integrand dimensions)
            :param hidden_size: hidden layer size (arbitrary value, which would be chosen when training the network)
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.input_hidden_layer = nn.Linear(input_size, hidden_size)  # initialisation of input->hidden layers structure
        self.sigmoid_activation = nn.Sigmoid()  # hidden layer activation function
        self.output_layer = nn.Linear(hidden_size, 1)  # output layer initialisation (always size 1)

    def forward(self, x):
        """
            Forward propagation of the data through the network.

            :param x:    the data to be propagated

            :returns: data after the forward propagation
        """
        x = self.input_hidden_layer(x)  # input->hidden propagation
        x = self.sigmoid_activation(x)  # sigmoid activation function applied
        x = self.output_layer(x)  # hidden->output propagation

        return x


def train(model, criterion, optimizer, x_train, y_train, epochs, verbose=True):
    """
        Trains the model.

        :param model:      The model to be trained
        :param criterion:  Loss function
        :param optimizer:  Optimization algorithm
        :param x_train:    Training inputs
        :param y_train:    True labels
        :param epochs:     Number of training epochs
        :param verbose:    Boolean depicting whether should the network print each 100 epochs done message,
                           or only the completion message.

        :returns: loss functions history
    """
    loss_history = []
    start_time = time.time()
    for epoch in range(epochs):
        predictions = model(x_train)  # forward propagation of all the data
        loss = criterion(predictions, y_train)  # loss function calculation

        optimizer.zero_grad()  # gradients are being reset
        loss.backward()  # backwards data propagation
        optimizer.step()  # optimization step (network params are being updated)

        loss_history.append(loss.item())  # current loss function added to history

        # hereon the logs are being printed
        if verbose:
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')
    total_time = time.time() - start_time
    print(f'Training done! Time elapsed: {total_time:.2f} seconds')

    return loss_history  # loss history is returned


def test(model, criterion, x_test, y_test):
    """
        Tests the model.

        :param model:     The trained model
        :param criterion: Loss function
        :param x_test:    Test inputs
        :param y_test:    True labels

        :returns: test loss function value
    """
    with torch.no_grad():  # no gradients will be calculated
        predictions = model(x_test)  # forward data propagation
        loss = criterion(predictions, y_test)  # loss function for the test data

    return loss.item()  # loss function value (item() required to convert tensor to scalar)


def predict_with(model, x_test):
    """
        Uses the model to predict values based on x_test arguments.

        :param model:  the trained model
        :param x_test: test inputs

        :returns: predicted function value
    """
    with torch.no_grad():
        prediction = model(x_test)  # forward data propagation

    return prediction


def extract_params(model):
    """
        Extracts weights and biases from the network.

        :param model: the trained model

        :returns: tuple of 4 numpy.ndarray-s with biases1, weights1, biases2 and weights2
                  (number represents 1 - input->hidden layers params, 2 - hidden->output layers params)
    """

    b1 = model.input_hidden_layer.bias.detach().numpy()
    w1 = model.input_hidden_layer.weight.detach().numpy()
    b2 = model.output_layer.bias.detach().numpy()
    w2 = model.output_layer.weight.detach().numpy().flatten()

    return b1, w1, b2, w2


class NeuralNumericalIntegration:

    @staticmethod
    def calculate(alphas, betas, network_params, n_dims=1):
        """
            Calculates integrand value using neural network model params
            across given boundaries.

            :param alphas:         lower boundaries sequence (should be placed in integration order)
            :param betas:          upper boundaries sequence (should be placed in integration order)
            :param network_params: params for the trained neural network
            :param n_dims:          number of integrand dimensions (default value is 1)

            :returns: integrand value
        """

        b1, w1, b2, w2 = network_params

        def Phi_j(b1_j_, w1_j_):
            def xi(r_):
                prod_ = 1
                for d in range(1, n_dims + 1):
                    prod_ *= (-1) ** (r_ / (2 ** (n_dims - d)))

                return prod_

            def l(i_, r_):
                if (r_ / (2 ** (n_dims - (i_ + 1)))) % 2 == 0:
                    return alphas[i_]
                else:
                    return betas[i_]

            Phi_sum = 0

            for r in range(1, (2 ** n_dims) + 1):
                sum_ = 0
                for i in range(n_dims):
                    sum_ += w1_j_[i] * l(i, r)
                Phi_sum += xi(r) * polylog(n_dims, -np.exp(-b1_j_ - sum_))

            return Phi_sum

        integral_sum = 0

        prod_bound = 1
        for i in range(n_dims):
            prod_bound *= betas[i] - alphas[i]

        for w2_j, w1_j, b1_j in zip(w2, w1, b1):

            Phi_j_ = Phi_j(b1_j, w1_j)
            prod_w = 1
            for i in range(n_dims):
                prod_w *= w1_j[i]

            integral_sum += w2_j * (prod_bound + Phi_j_ / prod_w)

        return float((b2 * prod_bound + integral_sum)[0])

    @staticmethod
    def integrate(model, alphas, betas, n_dims=1):
        """
            Integrates function-approximator (model) of n-dim dimensions over given boundaries.

            :param model: the trained model-approximator
            :param alphas: lower boundaries sequence (should be placed in integration order)
            :param betas:  upper boundaries sequence (should be placed in integration order)
            :param n_dims: number of integrand dimensions (default value is 1)

            :returns: neural numeric integration result
        """
        network_params = extract_params(model)

        return NeuralNumericalIntegration.calculate(alphas, betas, network_params, n_dims)


def generate_data(func, lower, upper, n_samples=100, n_dim=1):
    """
        Generates data in the form of a 2D tensor of variables for the function and neural network input
        as well as the function values for the generated tensor of variables.

        :param func:     function to provide values for the variables
        :param lower:    lower bounds of variable values
        :param upper:    upper bounds of variable values
        :param n_samples: number of points of data to generate per dimension (default value is 100)
        :param n_dim:     number of dimensions of the function func (default value is 1)

        :returns: dataset of variables X and function values y
    """
    X, y = None, None
    if n_dim == 1:
        X = torch.linspace(lower[0], upper[0], n_samples).view(n_samples, 1)
        y = func(X).view(n_samples, 1)
    else:
        ranges = [torch.linspace(lower[n], upper[n], n_samples).tolist() for n in range(n_dim)]
        combinations = list(itertools.product(*ranges))
        X = torch.tensor(combinations, dtype=torch.float32)
        y = func(X).view(-1, 1)

    return X, y

##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
from typing import TypeAlias, Annotated

import mpmath
import numpy as np
from mpmath import polylog
from torch import Tensor

from .model import MLP

Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]


class NeuralNumericalIntegration:

    @staticmethod
    def _effective_biases(
            B1: np.ndarray,
            W1: np.ndarray,
            n_int_dims: int,
            theta: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits W1 into integration weights W1_x and parameter weights W1_t,
        then absorbs the parameter contribution into effective biases.

        W1 has shape (k, n_int_dims + n_params).
        Returns:
            W1_x   — shape (k, n_int_dims), used in the polylog formula
            b1_eff — shape (k,),  b1_eff[j] = B1[j] + W1_t[j,:] @ theta
        """
        W1_x = W1[:, :n_int_dims]                  # integration columns
        W1_t = W1[:, n_int_dims:]                  # parameter columns

        if len(theta) > 0:
            theta_arr = np.asarray(theta, dtype=float)
            b1_eff = B1 + W1_t @ theta_arr         # shape (k,)
        else:
            b1_eff = B1.copy()

        return W1_x, b1_eff

    # ------------------------------------------------------------------
    @staticmethod
    def calculaten(
            alphas: list[float],
            betas: list[float],
            network_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            theta: list[float] = [],
    ) -> float:
        """
        n-dimensional NNI with optional parameter vector theta.

        alphas / betas are the integration bounds only (length = n_int_dims).
        theta holds fixed parameter values (length = n_params).
        """
        B1, W1, B2, W2 = network_params
        alphas = np.asarray(alphas)
        betas  = np.asarray(betas)
        n = len(alphas)                             # number of integration dims

        W1_x, b1_eff = NeuralNumericalIntegration._effective_biases(
            B1, W1, n, theta)

        k = len(b1_eff)
        prod_beta_alpha = np.prod(betas - alphas)
        I_hat = B2 * prod_beta_alpha

        for j in range(k):
            b1_j  = b1_eff[j]
            w1_j  = W1_x[j]
            Phi_j = 0.0
            for r in range(1, 2 ** n + 1):
                xi_r = np.prod([(-1) ** (int(np.floor(r / (2 ** (n - d)))))
                                for d in range(1, n + 1)])
                l_i_r = np.array([
                    alphas[i] if int(np.floor(r / (2 ** (n - (i + 1))))) % 2 == 0
                    else betas[i]
                    for i in range(n)
                ])
                argument = -mpmath.exp(-b1_j - float(np.sum(w1_j * l_i_r)))
                Phi_j += float(xi_r * polylog(n, argument))

            I_hat += W2[j] * (prod_beta_alpha + Phi_j / np.prod(w1_j))

        return float(I_hat)

    # ------------------------------------------------------------------
    @staticmethod
    def calculate2(
            alphas: list[float],
            betas: list[float],
            network_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            theta: list[float] = [],
    ) -> float:
        alpha1, alpha2, beta1, beta2 = alphas[0], alphas[1], betas[0], betas[1]
        B1, W1, B2, W2 = network_params

        W1_x, b1_eff = NeuralNumericalIntegration._effective_biases(
            B1, W1, 2, theta)

        def Phi_j(a1, bt1, a2, bt2, b1, w1_1, w1_2):
            t1 = polylog(2, -np.exp(-b1 - w1_1 * a1  - w1_2 * a2))
            t2 = polylog(2, -np.exp(-b1 - w1_1 * a1  - w1_2 * bt2))
            t3 = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * a2))
            t4 = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * bt2))
            return t1 - t2 - t3 + t4

        integral_sum = 0.0
        for w2_j, w1_1j, w1_2j, b1_j in zip(W2, W1_x[:, 0], W1_x[:, 1], b1_eff):
            phi = Phi_j(alpha1, beta1, alpha2, beta2, b1_j, w1_1j, w1_2j)
            integral_sum += w2_j * (
                (beta1 - alpha1) * (beta2 - alpha2) + phi / (w1_1j * w1_2j)
            )

        return float(B2 * (beta1 - alpha1) * (beta2 - alpha2) + integral_sum)

    # ------------------------------------------------------------------
    @staticmethod
    def calculate1(
            alphas: list[float],
            betas: list[float],
            network_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            theta: list[float] = [],
    ) -> float:
        alpha, beta = alphas[0], betas[0]
        B1, W1, B2, W2 = network_params

        W1_x, b1_eff = NeuralNumericalIntegration._effective_biases(
            B1, W1, 1, theta)
        W1_x = W1_x.flatten()

        def Phi_j(alp, bt, b1, w1):
            return (polylog(1, -np.exp(-b1 - w1 * alp)) -
                    polylog(1, -np.exp(-b1 - w1 * bt)))

        integral_sum = 0.0
        for w2_j, w1_j, b1_j in zip(W2, W1_x, b1_eff):
            phi = Phi_j(alpha, beta, b1_j, w1_j)
            integral_sum += w2_j * ((beta - alpha) + phi / w1_j)

        return float(B2 * (beta - alpha) + integral_sum)

    # ------------------------------------------------------------------
    @staticmethod
    def integrate(
            model: MLP,
            alphas: list[float] = [],
            betas: list[float] = [],
            n_dims: int = 1,
            n_int_dims: int | None = None,   # NEW
            theta: list[float] = [],          # NEW: fixed parameter values
            unit_cube: bool = False,
    ) -> float | None:
        """
        :param n_dims:     total input size of the model (integration vars + params)
        :param n_int_dims: how many of those are integration variables.
                           Defaults to n_dims (old behaviour, no parameters).
        :param theta:      fixed values of the parameter inputs, in the same
                           order and scaling as they appear in columns n_int_dims..n_dims-1
                           of the training data.
        :param unit_cube:  if True, integration bounds are set to [0,1]^n_int_dims.
        """
        if n_int_dims is None:
            n_int_dims = n_dims

        network_params = model.extract_params()

        if unit_cube:
            alphas = [0.0] * n_int_dims
            betas  = [1.0] * n_int_dims

        if n_int_dims == 1:
            return NeuralNumericalIntegration.calculate1(
                alphas, betas, network_params, theta)
        elif n_int_dims == 2:
            return NeuralNumericalIntegration.calculate2(
                alphas, betas, network_params, theta)
        else:
            return NeuralNumericalIntegration.calculaten(
                alphas, betas, network_params, theta)

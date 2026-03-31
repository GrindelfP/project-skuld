##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
from typing import TypeAlias, Annotated

import mpmath
import numpy as np
from mpmath import polylog
from torch import Tensor

from .model import MLP

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]


##########################################################################
###                  NEURAL NETWORK INTEGRATION CLASS                  ###
##########################################################################
class NeuralNumericalIntegration:
    """ Class-toolbox for neural numerical integration routines. """
    
    
    @staticmethod
    def calculaten(
            alphas: list[float],
            betas: list[float],
            network_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> float:
        """
        Calculates nD integrand value using neural network params
        across given boundaries.
        Args:
        ----------
        alphas : list[float]
            Lower integration boundaries (α_i)
        betas : list[float]
            Upper integration boundaries (β_i)
        network_params : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple of network params (B1, W1, B2, W2)

        Returns
        -------
        float
            I-hat approximation of the integral.
        """
        B1, W1, B2, W2 = network_params                             # network parameters
        alphas, betas = np.asarray(alphas), np.asarray(betas)       # boundaries
        n, k = len(alphas), len(B1)                                 # size parameters
        prod_beta_alpha = np.prod(betas - alphas)                   # PROD(β_i - α_i)
        I_hat = B2 * prod_beta_alpha                                # I_hat first term
        for j in range(k):                                          # SUM from j=1 to k
            b1_j, w1_j = B1[j], W1[j]                            # first layer bias, first layer weights
            Phi_j = 0.0                                             # Φ_j calculation
            for r in range(1, 2 ** n + 1):                          # SUM from r=1 to 2^n
                xi_r = np.prod([(-1) ** (np.floor(
                    r / (2 ** (n - d)))) for d in range(1, n + 1)]) # ξ_r term
                l_i_r = np.array([alphas[i] 
                    if int(np.floor(r / (2 ** (n-(i+1)))))%2 == 0 
                    else betas[i] for i in range(n)])               # l_ir term
                argument = -mpmath.exp(-b1_j - np.sum(w1_j * l_i_r))       # Li_b argument
                Phi_j += float(xi_r * polylog(n, argument))                # Li_b evaluation
            I_hat += W2[j]*(prod_beta_alpha+Phi_j/np.prod(w1_j))    # I_hat second term and evaluation

        print(I_hat)
        return float(I_hat)   

    
    @staticmethod
    def calculate2(
            alphas: list[float],
            betas: list[float],
            network_params: tuple[np.ndarray, np.ndarray,
            np.ndarray, np.ndarray]
    ) -> float:
        """
            Calculates 2D integrand value using neural network params
            across given boundaries.

            :param alphas:         lower boundaries sequence (should be placed in integration order)
            :param betas:          upper boundaries sequence (should be placed in integration order)
            :param network_params: params for the trained neural network (default value is 2)

            :returns: integrand value
        """
        alpha1, alpha2, beta1, beta2 = alphas[0], alphas[1], betas[0], betas[1]
        B1, W1, B2, W2 = network_params

        def Phi_j(alp1: float, bt1: float, alp2: float, bt2: float,
                  b1: float, w1_1: float, w1_2: float) -> float:
            term_1: float = polylog(2, -np.exp(-b1 - w1_1 * alp1 - w1_2 * alp2))
            term_2: float = polylog(2, -np.exp(-b1 - w1_1 * alp1 - w1_2 * bt2))
            term_3: float = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * alp2))
            term_4: float = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * bt2))
            return term_1 - term_2 - term_3 + term_4

        integral_sum: float = 0.0
        for w2_j, w1_1j, w1_2j, b1_j in zip(W2, W1[:, 0], W1[:, 1], B1):
            phi_j: float = Phi_j(alpha1, beta1, alpha2, beta2, b1_j, w1_1j, w1_2j)
            integral_sum += w2_j * ((beta1 - alpha1) * (beta2 - alpha2) + phi_j / (w1_1j * w1_2j))

        return float(B2 * (beta1 - alpha1) * (beta2 - alpha2) + integral_sum)

    
    @staticmethod
    def calculate1(
            alphas: list[float],
            betas: list[float],
            network_params: tuple[np.ndarray, np.ndarray,
            np.ndarray, np.ndarray]
    ) -> float:
        """
            Calculates 1D integrand value using neural network model params
            across given boundaries.

            :param alphas:         lower boundaries sequence (should be placed in integration order)
            :param betas:          upper boundaries sequence (should be placed in integration order)
            :param network_params: params for the trained neural network (default value is 1)

            :returns: integrand value
        """
        alpha, beta = alphas[0], betas[0]
        B1, W1, B2, W2 = network_params
        W1 = W1.flatten()

        def Phi_j(alp: float, bt: float, b1: float, w1: float) -> float:
            term_alpha: float = polylog(1, -np.exp(-b1 - w1 * alp))
            term_beta: float = polylog(1, -np.exp(-b1 - w1 * bt))
            return term_alpha - term_beta

        integral_sum: float = 0.0
        for w2_j, w1_j, b1_j in zip(W2, W1, B1):
            phi_j: float = Phi_j(alpha, beta, b1_j, w1_j)
            integral_sum += w2_j * ((beta - alpha) + phi_j / w1_j)

        return float(B2 * (beta - alpha) + integral_sum)

    @staticmethod
    def integrate(
            model: MLP,
            alphas: list[float],
            betas: list[float],
            n_dims: int = 1,
            unit_cube: bool = False
    ) -> float | None:
        """
            Integrates function-approximator (model) of n-dim dimensions over
            given boundaries.

            :param model: the trained model-approximator
            :param alphas: lower boundaries sequence (should be placed in integration order)
            :param betas:  upper boundaries sequence (should be placed in integration order)
            :param n_dims: number of integrand dimensions
            :param unit_cube: whether integration is performed across the unit cube

            :returns: neural numeric integration result
        """
        network_params = model.extract_params()

        if unit_cube:
            alphas: list[float] = []
            betas: list[float] = []
            for _ in range(n_dims):
                alphas.append(0.0)
                betas.append(1.0)
        
        if n_dims == 1:
            return NeuralNumericalIntegration.calculate1(alphas, betas, network_params)
        elif n_dims == 2:
            return NeuralNumericalIntegration.calculate2(alphas, betas, network_params)
        else:
            return NeuralNumericalIntegration.calculaten(alphas, betas, network_params)

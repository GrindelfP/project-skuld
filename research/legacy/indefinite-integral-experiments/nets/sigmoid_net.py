import torch
import torch.nn as nn

##############################################################################
# 4.  NETWORK
##############################################################################

class PrimitiveNet(nn.Module):
    """
    Полносвязная нейросеть N(a, b, m, n, u₁, u₂, u₃) ≈ F(s; u).

    Аппроксимирует первообразную, т.е. ∂³N/∂u₁∂u₂∂u₃ ≈ f̃.

    Активация: Sigmoid.
    """

    def __init__(self,
                 n_params: int = 4,
                 n_int_vars: int = 3,
                 hidden_sizes: list = None,
                 output_scale: float = 1e4):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.n_params = n_params
        self.n_int_vars = n_int_vars

        layers = []
        in_dim = n_params + n_int_vars
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Sigmoid()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_xavier(output_scale)

    def _init_xavier(self, output_scale: float = 1.0):
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
            last_linear.weight.data *= output_scale
            last_linear.bias.data *= output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

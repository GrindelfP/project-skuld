##########################################################################
########################### SKULD NNI LIBRARY ############################
##########################################################################
                  ###         version 0.2.1         ###
                  ###    a neural network based     ###
                  ### numerical integration library ###
##########################################################################

__version__ = "0.2.1"

from .model import (
        MLP,
        init_model,
        split_data,
        set_global_device
)

from .scalers import (
        scale_data,
        descale_result
)

from .nni import (
        NeuralNumericalIntegration
)

from .generators import (
        generate_data,
)

__all__ = [
        "MLP",
        "NeuralNumericalIntegration",
        "init_model",
        "split_data",
        "scale_data",
        "descale_result",
        "generate_data",
        "set_global_device"
]

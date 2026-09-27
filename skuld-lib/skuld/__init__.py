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

from .siren import (
        SirenLayer,
        SirenPrimitiveNet,
        SirenIntegrator,
        mixed_partial_3,
)

from .wire import (
        WireLayer,
        WireResidualBlock,
        WirePrimitiveNet,
        WireIntegrator,
)

from .kan import (
        BSplineBasis,
        KANLayer,
        KANPrimitiveNet,
        KANIntegrator,
)

from .broknet import (
        SindriNet,
        BrokNet,
        BrokNetIntegrator,
        mixed_partial_3_expert,
)

__all__ = [
        "MLP",
        "NeuralNumericalIntegration",
        "init_model",
        "split_data",
        "scale_data",
        "descale_result",
        "generate_data",
        "set_global_device",
        "SirenLayer",
        "SirenPrimitiveNet",
        "SirenIntegrator",
        "mixed_partial_3",
        "WireLayer",
        "WireResidualBlock",
        "WirePrimitiveNet",
        "WireIntegrator",
        "BSplineBasis",
        "KANLayer",
        "KANPrimitiveNet",
        "KANIntegrator",
        "SindriNet",
        "BrokNet",
        "BrokNetIntegrator",
        "mixed_partial_3_expert",
]

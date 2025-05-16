from typing import Annotated
from torch import Tensor
from typing_extensions import TypeAlias

########################################
   ####### GLOBAL DEFINITIONS #######
########################################

Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]

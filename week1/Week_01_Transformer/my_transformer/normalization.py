import torch.nn as nn
from torch import Tensor

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        #todo one line!
        self.layerNorm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        #todo one line!
        return self.layerNorm(x)
from torch import nn
from multi_layer_convstar_net import MultiLayerConvStar
import torch

class MultistagesConvStarNet(nn.Module):
    def __init__(self, nclasses_level1, nclasses_level2, nclasses_level3, input_dim:int = 4, hidden_dim=64, kernel_size = 3):
        super.__init__()

        self.convStarNet = MultiLayerConvStar(input_size=input_dim, hidden_sizes=hidden_dim, kernel_sizes=kernel_size)

        self.conv_l1 = torch.nn.Conv2d(hidden_dim, nclasses_level1, (3, 3), padding=1)
        self.conv_l2 = torch.nn.Conv2d(hidden_dim, nclasses_level2, (3, 3), padding=1)
        self.conv_l3 = torch.nn.Conv2d(hidden_dim, nclasses_level3, (3, 3), padding=1)

    def forward(self, x, hidden_states = None):
        raise NotImplemented
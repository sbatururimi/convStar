from torch import nn
from .multi_layer_convstar_net import MultiLayerConvStar
import torch
import torch.nn.functional as F


class MultistagesConvStarNet(nn.Module):
    def __init__(
        self,
        nclasses_level1,
        nclasses_level2,
        nclasses_level3,
        test: bool = False,
        n_layers: int = 6,
        input_dim: int = 4,
        hidden_dim=64,
        kernel_size=3,
        wo_softmax: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.test = test
        self.wo_softmax = wo_softmax

        self.convStarNet = MultiLayerConvStar(
            input_size=input_dim, hidden_sizes=hidden_dim, kernel_sizes=kernel_size
        )

        self.conv_l1 = torch.nn.Conv2d(hidden_dim, nclasses_level1, (3, 3), padding=1)
        self.conv_l2 = torch.nn.Conv2d(hidden_dim, nclasses_level2, (3, 3), padding=1)
        self.conv_l3 = torch.nn.Conv2d(hidden_dim, nclasses_level3, (3, 3), padding=1)

    def forward(self, x, hidden_states=None):
        # (b x t x c x h x w) -> (b x c x t x h x w)
        # RNN expectation: Many RNN architectures, including ConvRNNs, expect the temporal dimension
        # to be placed after the channel dimension.
        # This is because RNNs are designed to process sequences,
        # and they typically operate on sequences of feature maps where each feature map is a 2D spatial grid (height x width).
        #
        # Convolution and RNN Compatibility: Convolutional layers operate on spatial dimensions (h and w),
        # and when combined with RNN layers, the network often needs to maintain the spatial dimensions
        # while iterating over temporal sequences. Permuting the dimensions
        # to place the temporal dimension (t) after the channel dimension (c)
        # aligns the data with the expected input format of RNN layers that will process these sequences.
        #
        #
        # Check dataset.py, L.95 [get_item] when we transpose X's dimensions
        x = x.permute(0, 2, 1, 3, 4)
        b, c, T, h, w = x.shape

        # convStar step---------------------------------
        # hidden_states is a list (number of layer) of hidden states of size [b x c x h x w]
        if hidden_states is None:
            # hidden_states = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            hidden_states = [torch.zeros((b, self.hidden_dim, h, w)).to(x.device) for _ in range(self.n_layers)]

        for t in range(T):
            hidden_states = self.convStarNet.forward(x[:, :, t, :, :], hidden_states)

        # we have 6 hidden states, each stage has 2 layer
        local_1 = hidden_states[1]
        local_2 = hidden_states[3]

        # XXX: From author's implementation, the similar effect is achived using nstage=3
        # if self.n_layers == 3:
            # local_1 = hidden_states[0]
            # local_2 = hidden_states[1]
        # elif self.nstage == 3:
        #     raise NotImplementedError # I don't undersatdn what are nstage
        #     # local_1 = hidden_states[1]
        #     # local_2 = hidden_states[3]
        # elif self.nstage == 2:
        #     raise NotImplementedError # I don't undersatdn what are nstage
        #     # local_1 = hidden_states[1]
        #     # local_2 = hidden_states[2]
        # elif self.nstage == 1:
        #     raise NotImplementedError # I don't undersatdn what are nstage
        #     # local_1 = hidden_states[-1]
        #     # local_2 = hidden_states[-1]

        local_1 = self.conv_l1(local_1)
        local_2 = self.conv_l2(local_2)

        last = hidden_states[-1] # output of last stage
        last = self.conv_l3(last)

        if self.test:
            return (
                F.softmax(last, dim=1),
                F.softmax(local_1, dim=1),
                F.softmax(local_2, dim=1),
            )
        elif self.wo_softmax:
            return last, local_1, local_2
        else:
            return (
                F.log_softmax(last, dim=1),
                F.log_softmax(local_1, dim=1),
                F.log_softmax(local_2, dim=1),
            )

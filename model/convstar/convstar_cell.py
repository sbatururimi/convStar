from torch import nn
import torch

class ConvStarCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()

        padding = kernel_size // 2 # "same" padding: ensure that the spatial dimensions (height and width) of the input tensor are preserved after the convolution operation
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv_Wx_K = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.conv_Wh_K = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.conv_Wx_Z = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        # Bias terms for K and Z
        self.B_K = nn.Parameter(torch.zeros(hidden_size))
        self.B_Z = nn.Parameter(torch.zeros(hidden_size))

        # Initializing weights and biases
        nn.init.orthogonal_(self.conv_Wx_K.weight)
        nn.init.orthogonal_(self.conv_Wh_K.weight)
        nn.init.orthogonal_(self.conv_Wx_Z.weight)
        nn.init.constant_(self.conv_Wx_K.bias, 0.)
        nn.init.constant_(self.conv_Wh_K.bias, 0.)
        nn.init.constant_(self.conv_Wx_Z.bias, 0.)
        nn.init.constant_(self.B_K, 1.)
        nn.init.constant_(self.B_Z, 0.)

    def forward(self, input_, prev_state):
        # Get batch and spatial sizes
        batch_size = input_.size(0)
        spatial_size = input_.size()[2:]

        # Generate empty prev_state if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size)

        # Formula 1: Compute K_t^l
        K_t = torch.sigmoid(self.conv_Wx_K(input_) + self.conv_Wh_K(prev_state) + self.B_K.view(1, -1, 1, 1))

        # Formula 2: Compute Z_t^l
        Z_t = torch.tanh(self.conv_Wx_Z(input_) + self.B_Z.view(1, -1, 1, 1))

        # Formula 3: Compute H_t^l (new state)
        new_state = torch.tanh(prev_state + K_t * (Z_t - prev_state))

        return new_state
import torch
from torch import nn


class LabelRefinementNet(nn.Module):
    # Ref: "Crop mapping from image time series: Deep learning with multi-scale
    # label hierarchies":
    # Fig. 3. Label refinement CNN architecture
    def __init__(
        self,
        num_classes_l1: int,
        num_classes_l2: int,
        num_classes_l3: int,
        out_channels: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes_l1 = num_classes_l1
        self.num_classes_l2 = num_classes_l2
        self.num_classes_l3 = num_classes_l3

        input_dim = self.num_classes_l1 + self.num_classes_l2 + self.num_classes_l3

        # conv 1
        kernel_size = 1
        self.conv1 = nn.Conv2d(
            input_dim,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="replicate",
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # conv 2
        kernel_size = 3
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="replicate",
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        # conv 3
        kernel_size = 3
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="replicate",
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.drop3 = nn.Dropout(dropout)

        # conv 4
        kernel_size = 1
        self.conv4 = nn.Conv2d(
            out_channels,
            self.num_classes_l3,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="replicate",
        )

    def forward(self, x):
        x1_, x2_, x3 = x
        x_concat = torch.cat((x1_, x2_, x3), dim=1)

        # fig 3 --------
        out1 = self.conv1(x_concat)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out = self.conv2(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += out1

        out = self.relu(out)
        out = self.drop3(out)

        out = self.conv4(out)
        # ------ end of fig 3 implementation

        out += x3  # This is visible in formula (5) where: out=f(Y_hat1, Y_hat2,..., Y_hatN), x3 = Y_hatN
        return out

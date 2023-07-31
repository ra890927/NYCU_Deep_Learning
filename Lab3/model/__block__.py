from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        down_sample: nn.Module = None,
        activation: nn.modules.activation = nn.ReLU(inplace=True)
    ) -> None:
        super(BasicBlock, self).__init__()

        self.expansion = expansion
        self.activation = activation
        self.down_sample = down_sample

        # architecture

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * self.expansion,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

    def forward(self, inputs: TensorDataset) -> Tensor:
        outputs = self.block(inputs)

        residual = inputs
        if self.down_sample:
            residual = self.down_sample(residual)

        outputs = self.activation(outputs + residual)

        return outputs


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        down_sample: nn.Module = None,
        activation: nn.modules.activation = nn.ReLU(inplace=True)
    ) -> None:
        super(BottleneckBlock, self).__init__()

        self.expansion = expansion
        self.activation = activation
        self.down_sample = down_sample

        # architecture

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * expansion,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels * expansion)
        )

    def forward(self, inputs: TensorDataset) -> Tensor:
        outputs = self.block(inputs)

        residual = inputs
        if self.down_sample:
            residual = self.down_sample(residual)

        outputs = self.activation(outputs + residual)

        return outputs

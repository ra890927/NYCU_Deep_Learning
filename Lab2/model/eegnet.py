from torch import nn, Tensor
from torch.utils.data import TensorDataset


class EEGNet(nn.Module):
    def __init__(
        self,
        activation: nn.modules.activation,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        # model params

        self.activation = activation

        # model architecture

        # Layer 1
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=16,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        )

        # Layer 2
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            self.activation(),
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=0
            ),
            nn.Dropout(p=dropout)
        )

        # Layer 3
        self.seperable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            self.activation(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(p=dropout)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, inputs: TensorDataset) -> Tensor:
        # Block 1
        first_conv_result = self.first_conv(inputs)
        depthwise_conv_result = self.depthwise_conv(first_conv_result)

        # Block 2
        seperable_conv_result = self.seperable_conv(depthwise_conv_result)

        return self.classify(seperable_conv_result)

from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset
from typing import Type, Union, Optional, List

from .__block__ import BasicBlock, BottleneckBlock


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, BottleneckBlock]],
        layer_blocks: List[int],
        expansion: int,
        num_classes: int
    ) -> None:
        super(ResNet, self).__init__()

        self.current_channels = 64
        self.expansion = expansion

        # architecture

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.conv_2 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[0],
            in_channels=64
        )

        self.conv_3 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[1],
            in_channels=128,
            stride=2
        )

        self.conv_4 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[2],
            in_channels=256,
            stride=2
        )

        self.conv_5 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[3],
            in_channels=512,
            stride=2
        )

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * self.expansion, num_classes)
        )

    def forward(self, inputs: TensorDataset) -> Tensor:
        partial_result = inputs
        for i in range(1, 6):
            partial_result = getattr(self, f'conv_{i}')(partial_result)
        return self.classify(partial_result)

    def __make_layer(
        self,
        block: Type[Union[BasicBlock, BottleneckBlock]],
        num_of_blocks: int,
        in_channels: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Make a layer with given block module
        :param block: block module to be used
        :param num_of_blocks: number of block layers
        :param in_channels: number od block input channels
        :param stride: stride
        """

        down_sample = None
        out_channels = in_channels * self.expansion
        if stride != 1 or self.current_channels != out_channels:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.current_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        layers = [
            block(
                in_channels=self.current_channels,
                out_channels=in_channels,
                stride=stride,
                down_sample=down_sample
            )
        ]

        # update and it will be used for next block layer
        self.current_channels = out_channels

        # in_channels:  in order to connect with prev block,
        #               input channel equals out channels
        # out_channels: block module has expansion,
        #               output channel equals input channels

        layers += [
            block(
                in_channels=out_channels,
                out_channels=in_channels,
            )
            for _ in range(1, num_of_blocks)
        ]

        return nn.Sequential(*layers)


def ResNet18(
    num_classes: int,
    layer_blocks: List[int] = [2, 2, 2, 2]
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        layer_blocks=layer_blocks,
        expansion=1,
        num_classes=num_classes
    )


def ResNet50(
    num_classes: int,
    layer_blocks: List[int] = [3, 4, 6, 3]
) -> ResNet:
    return ResNet(
        BottleneckBlock,
        layer_blocks=layer_blocks,
        expansion=4,
        num_classes=num_classes
    )


def ResNet152(
    num_classes: int,
    layer_blocks: List[int] = [3, 8, 36, 3]
) -> ResNet:
    return ResNet(
        BottleneckBlock,
        layer_blocks=layer_blocks,
        expansion=4,
        num_classes=num_classes
    )

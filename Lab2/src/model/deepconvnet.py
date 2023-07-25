from torch import nn, Tensor
from torch.utils.data import TensorDataset
from typing import Tuple
from functools import reduce
from collections import OrderedDict


class DeepConvNet(nn.Module):
    def __init__(
        self,
        activation: nn.modules.activation,
        dropout: float,
        num_of_linear: int,
        filters: Tuple[int] = (25, 50, 100, 200)
    ) -> None:
        super().__init__()

        # model params

        self.filters = filters
        self.activation = activation

        # model architecture

        self.conv_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=filters[0],
                kernel_size=(1, 5),
                bias=False
            ),
            nn.Conv2d(
                in_channels=filters[0],
                out_channels=filters[0],
                kernel_size=(2, 1),
                bias=False
            ),
            nn.BatchNorm2d(filters[0]),
            self.activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        for idx, num_of_filters in enumerate(filters[:-1], start=1):
            setattr(self, f'conv_{idx}', nn.Sequential(
                nn.Conv2d(
                    in_channels=num_of_filters,
                    out_channels=filters[idx],
                    kernel_size=(1, 5),
                    bias=False
                ),
                nn.BatchNorm2d(filters[idx]),
                self.activation(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=dropout)
            ))

        self.flatten_size = filters[-1] * \
            reduce(lambda x, _: round((x - 4) / 2), filters[:-1], 373)
        interval = round((100 - 2) / num_of_linear)
        features = [self.flatten_size]
        next_layer = 100

        while next_layer > 2:
            features.append(next_layer)
            next_layer -= interval
        features.append(2)

        layers = [('flatten', nn.Flatten())]
        for idx, in_features in enumerate(features[:-1]):
            layers.append((f'linear_{idx}', nn.Linear(
                in_features=in_features,
                out_features=features[idx + 1],
                bias=True
            )))

            if idx != len(features) - 2:
                layers.append((f'activation_{idx}', self.activation()))
                layers.append((f'dropout_{idx}', nn.Dropout2d()))

        self.classify = nn.Sequential(OrderedDict(layers))

    def forward(self, inputs: TensorDataset) -> Tensor:
        partial_result = inputs
        for idx in range(len(self.filters)):
            partial_result = getattr(self, f'conv_{idx}')(partial_result)
        return self.classify(partial_result)

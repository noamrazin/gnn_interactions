from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F


class SimpleConvnet(nn.Module):

    def __init__(self, input_size: Tuple[int, int, int], output_size: int, filters=(32, 32, 64, 64, 128, 128),
                 dropout_init_val: float = 0.2, use_dropout: bool = True, use_batchnorm: bool = True):
        """
         :param input_size: size of inputs in format (C,H,W).
         :param output_size: size of the network output.
         :param filters: A list of of length N containing the number of filters in each conv layer.
         :param dropout_init_val: Initial dropout value. Dropout value is incremented by 0.1 each downsampling.
         :param use_dropout: A flag that determines whether to use dropout or not.
         :param use_batchnorm: A flag that determines whether to use batchnorm or not.
         """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.filters = filters
        self.dropout_init_val = dropout_init_val
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.feature_extractor = self.__make_feature_extractor()
        self.classifier = nn.Linear(self.filters[-1], self.output_size)

    def __make_feature_extractor(self):
        in_channels, in_h, in_w, = self.input_size

        layers = []
        prev_channels = in_channels
        p_dropout = self.dropout_init_val
        for i, num_channels in enumerate(self.filters):
            layers.append(nn.Conv2d(prev_channels, num_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(num_channels))

            if i == len(self.filters) - 1 or num_channels != self.filters[i + 1]:
                layers.append(nn.MaxPool2d(2))
                if self.use_dropout:
                    layers.append(nn.Dropout2d(p=p_dropout))
                    p_dropout += 0.1

            prev_channels = num_channels

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

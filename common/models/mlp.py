from typing import Sequence, Callable, Tuple

import torch.nn as nn
from torch import Tensor


class MultiLayerPerceptron(nn.Module):
    """
    Simple MultiLayer Perceptron model.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layer_sizes: Sequence[int] = None, bias: bool = True, dropout: float = 0,
                 hidden_layer_activation: Callable[[Tensor], Tensor] = nn.ReLU(inplace=True), output_activation: Callable[[Tensor], Tensor] = None,
                 flatten_inputs: bool = True):
        """
        :param input_size: input size.
        :param output_size: output size.
        :param hidden_layer_sizes: sequence of hidden dimension sizes.
        :param bias: if set to False, the linear layers will not use biases.
        :param dropout: dropout rate for the hidden layers.
        :param hidden_layer_activation: activation for the hidden layers.
        :param output_activation: optional activation for the output layer.
        :param flatten_inputs: if True (default), will flatten all inputs to 2 dimensional tensors (i.e. flattens all non-batch dimensions) in the
        forward function.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes is not None else []
        self.depth = len(self.hidden_layer_sizes) + 1
        self.output_activation = output_activation if output_activation is not None else lambda x: x
        self.flatten_inputs = flatten_inputs

        self.hidden_layers_sequential = self.__create_hidden_layers_sequential_model(hidden_layer_activation)
        self.output_layer = nn.Linear(self.hidden_layer_sizes[-1] if self.hidden_layer_sizes else input_size, output_size, bias=self.bias)

    def __create_hidden_layers_sequential_model(self, activation):
        layers = []

        prev_size = self.input_size
        for hidden_layer_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_layer_size, bias=self.bias))
            layers.append(activation)

            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))

            prev_size = hidden_layer_size

        return nn.Sequential(*layers)

    def get_linear_layers(self) -> Sequence[nn.Linear]:
        linear_layers = []
        for child_module in self.hidden_layers_sequential.children():
            if isinstance(child_module, nn.Linear):
                linear_layers.append(child_module)

        linear_layers.append(self.output_layer)
        return linear_layers

    def get_layer_dims(self, layer_index: int) -> Tuple[int, int]:
        num_rows = self.hidden_layer_sizes[layer_index - 1] if layer_index != 0 else self.input_size
        num_cols = self.hidden_layer_sizes[layer_index] if layer_index != self.depth - 1 else self.output_size
        return num_rows, num_cols

    def forward(self, x):
        if self.flatten_inputs:
            x = x.view(x.size(0), -1)

        x = self.hidden_layers_sequential(x)
        return self.output_activation(self.output_layer(x))

import math
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t, _size_4_t
from torch.nn.modules.utils import _pair, _quadruple


def conv2d_same_padding_layer(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], dilation: Union[int, tuple] = 1,
                              bias: bool = True) -> nn.Conv2d:
    """
    Creates a same padding Conv2d layer.
    :param in_channels: number of input channels.
    :param out_channels: number of output channels.
    :param kernel_size: (height, width) or an int for a square filter
    :param dilation: (height_dilation, width_dilation) or an int for same dilation
    :param bias: bool flag whether to use bias or not.
    :return: Conv2d layer with the specified parameters and same padding.
    """
    padding = conv2d_same_padding(kernel_size, dilation)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias)


def conv2d_output_size(input_size: tuple, num_filters: int, kernel_size: Union[int, tuple], padding: Union[int, tuple] = 0,
                       stride: Union[int, tuple] = 1, dilation: Union[int, tuple] = 1) -> tuple:
    """
    Calculates the output size of a conv2d layer.
    :param input_size: (in_channels, in_height, in_width)
    :param num_filters: number of filters in the conv2d layer.
    :param kernel_size: (height, width) or an int for a square filter
    :param padding: (height_pad, width_pad) or an int for same padding
    :param stride: (height_stride, width_stride) or an int for same stride
    :param dilation: (height_dilation, width_dilation) or an int for same dilation
    :return: (out_channels, out_height, out_width)
    """
    kernel_size = kernel_size if not isinstance(kernel_size, int) else (kernel_size, kernel_size)
    padding = padding if not isinstance(padding, int) else (padding, padding)
    dilation = dilation if not isinstance(dilation, int) else (dilation, dilation)
    stride = stride if not isinstance(stride, int) else (stride, stride)

    out_height = math.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    out_width = math.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
    return num_filters, out_height, out_width


def conv2d_same_padding_with_stride(input_size: tuple, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1,
                                    dilation: Union[int, tuple] = 1) -> tuple:
    """
    Calculates the necessary padding for conv2d output to be of same width and height of input.
    :param input_size: (in_channels, in_height, in_width)
    :param kernel_size: (height, width) or an int for a square filter
    :param stride: (height_stride, width_stride) or an int for same stride
    :param dilation: (height_dilation, width_dilation) or an int for same dilation
    :return: (height_pad, width_pad)
    """
    kernel_size = kernel_size if not isinstance(kernel_size, int) else (kernel_size, kernel_size)
    stride = stride if not isinstance(stride, int) else (stride, stride)
    dilation = dilation if not isinstance(dilation, int) else (dilation, dilation)

    height_pad = ((input_size[1] - 1) * (stride[0] - 1) + dilation[0] * (kernel_size[0] - 1)) // 2
    width_pad = ((input_size[2] - 1) * (stride[1] - 1) + dilation[1] * (kernel_size[1] - 1)) // 2
    return height_pad, width_pad


def conv2d_same_padding(kernel_size: Union[int, tuple], dilation: Union[int, tuple] = 1) -> tuple:
    """
    Calculates the necessary padding for conv2d output to be of same width and height of input for layer with stride=1.
    :param kernel_size: (height, width) or an int for a square filter
    :param dilation: (height_dilation, width_dilation) or an int for same dilation
    :return: (height_pad, width_pad)
    """
    kernel_size = kernel_size if not isinstance(kernel_size, int) else (kernel_size, kernel_size)
    dilation = dilation if not isinstance(dilation, int) else (dilation, dilation)

    height_pad = (dilation[0] * (kernel_size[0] - 1)) // 2
    width_pad = (dilation[1] * (kernel_size[1] - 1)) // 2
    return height_pad, width_pad


class ProductPool2d(nn.Module):
    """
    Applies a 2D product pooling over an input signal of input size (N, C, H, W).
    """

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_4_t = 0):
        """
        :param kernel_size: size of pooling kernel, int or 2-tuple.
        :param stride: pooling stride, int or 2-tuple. Default value is kernel_size.
        :param padding: pooling padding by ones, int or 4-tuple (l, r, t, b) as in pytorch F.pad.
        """
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        stride = stride if stride is not None else kernel_size
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return prod_pool2d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


class ProductPool1d(nn.Module):
    """
    Applies a 1D product pooling over an input signal of input size (N, C, D).
    """

    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: _size_2_t = 0):
        """
        :param kernel_size: size of pooling kernel, int.
        :param stride: pooling stride, int. Default value is kernel_size.
        :param padding: pooling padding by ones, int or 2-tuple (l, r) as in pytorch F.pad.
        """
        super().__init__()
        self.kernel_size = kernel_size
        stride = stride if stride is not None else kernel_size
        self.stride = stride
        self.padding = _pair(padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return prod_pool1d(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)


def prod_pool2d(input: torch.Tensor, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_4_t = 0):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride) if stride is not None else kernel_size
    padding = _quadruple(padding)

    input = F.pad(input, padding, mode='constant', value=1)
    input = input.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    input = input.contiguous().view(input.size()[:4] + (-1,)).prod(dim=-1)
    return input


def prod_pool1d(input: torch.Tensor, kernel_size: int, stride: Optional[int] = None, padding: _size_2_t = 0):
    stride = stride if stride is not None else kernel_size
    padding = _pair(padding)

    input = F.pad(input, padding, mode='constant', value=1)
    input = input.unfold(2, kernel_size, stride)
    input = input.prod(dim=-1)
    return input


def get_prod_pool2d_output_size(in_height: int, in_width: int, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_4_t = 0):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride) if stride is not None else kernel_size
    padding = _quadruple(padding)

    out_height = get_prod_pool1d_output_size(in_dim=in_height, kernel_size=kernel_size[0], stride=stride[0], padding=(padding[2], padding[3]))
    out_width = get_prod_pool1d_output_size(in_dim=in_width, kernel_size=kernel_size[1], stride=stride[1], padding=(padding[0], padding[1]))
    return out_height, out_width


def get_prod_pool1d_output_size(in_dim: int, kernel_size: int, stride: Optional[int] = None, padding: _size_2_t = 0):
    stride = stride if stride is not None else kernel_size
    padding = _pair(padding)
    pad_size = padding[0] + padding[1]

    out_dim = (math.floor((in_dim + pad_size - kernel_size) / stride + 1))
    return out_dim

from functools import partial
import torch
import time
import crypten
from crypten.encoder import FixedPointEncoder
import crypten.communicator as comm
from crypten.mpc.mpc import MPCTensor
from crypten.mpc.primitives import ArithmeticSharedTensor
from crypten.common.rng import generate_random_ring_element
from typing import Optional, Tuple, Union


encoder = FixedPointEncoder()
s = encoder.scale


def get_bias_shape(output_shape):
    return (1,) * (len(output_shape) - 1) + (-1,)


class Linear:
    def __init__(self, weight, bias=None, src=0):
        self.weight = encoder.encode(weight)
        self.bias = encoder.encode(bias).reshape(1, 1, -1) if bias is not None else None
        self.src = src

    def forward(self, x: MPCTensor):
        x.share = torch.matmul(x.share, self.weight).div_(s, rounding_mode="trunc")
        if self.bias is not None and comm.get().get_rank() == self.src:
            x.share += self.bias
        return x

    def __call__(self, x):
        return self.forward(x)


class Conv1d:
    def __init__(self, weight, bias=None, stride=1, padding=0, src=0):
        self.src = src
        self.weight = encoder.encode(weight)
        self.bias = encoder.encode(bias).reshape(1, -1, 1) if bias is not None else None
        self.stride = (stride,) if isinstance(stride, int) else stride
        self.padding = padding

    def forward(self, x: MPCTensor):
        x.share = torch.nn.functional.conv1d(x.share, self.weight, stride=self.stride, padding="same").div_(s, rounding_mode="trunc")
        if self.bias is not None and comm.get().get_rank() == self.src:
            x.share += self.bias
        return x

    def __call__(self, x):
        return self.forward(x)


class Conv2d:
    def __init__(self, weight, bias=None, stride=1, padding=0, groups=1, src=0):
        self.src = src
        self.weight = encoder.encode(weight)
        self.bias = encoder.encode(bias).reshape(1, -1, 1, 1) if bias is not None else None
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = ((padding, padding), (padding, padding)) if isinstance(padding, int) else padding
        self.groups = groups

    def forward(self, x: MPCTensor):
        x.share = torch.nn.functional.conv2d(x.share.float(), self.weight.float(), stride=self.stride, padding=self.padding).div_(s, rounding_mode="trunc").to(torch.int64)
        if self.bias is not None and comm.get().get_rank() == self.src:
            x.share += self.bias
        return x

    def __call__(self, x):
        return self.forward(x)


class SquareActivation:
    def __init__(self, shape, src=0, device='cpu'):
        provider = crypten.mpc.get_default_provider()
        self.r_pair = provider.square(shape, device=device)
        self.src = src
        if comm.get().get_rank() == self.src:
            self.r = generate_random_ring_element(shape)
        else:
            self.recv_buff = torch.zeros(shape, dtype=torch.int64)

    def forward(self, x: MPCTensor):
        x = x.square(sharing=self.r_pair)
        x.share = self._fn(x.share)
        return x

    @partial(torch.jit.script, static_argnums=(0,))
    def _fn(self, share):
        if comm.get().get_rank() == self.src:
            comm.get().send(share - self.r, 1 - self.src)
            share = self.r.repeat(share.shape[0], *([1] * (len(share.shape) - 1)))
        else:
            share += comm.get().recv(share, self.src)
        return share

    def __call__(self, x):
        return self.forward(x)


class BatchNorm1d:
    def __init__(self, running_mean, running_var, src=0, eps=1e-5):
        self.running_mean = encoder.encode(running_mean.reshape(1, -1, 1))
        self.running_std_inv = encoder.encode(1.0 / torch.sqrt(running_var.reshape(1, -1, 1) + eps))
        self.src = src

    def forward(self, x: MPCTensor):
        if comm.get().get_rank() != self.src:
            x.share -= self.running_mean
        x.share = (x.share * self.running_std_inv).div_(s, rounding_mode="trunc").to(torch.int64)
        return x

    def __call__(self, x):
        return self.forward(x)


class BatchNorm2d:
    def __init__(self, running_mean, running_var, src=0, eps=1e-5):
        self.running_mean = encoder.encode(running_mean.reshape(1, -1, 1, 1))
        self.running_std_inv = encoder.encode(1.0 / torch.sqrt(running_var.reshape(1, -1, 1, 1) + eps))
        self.src = src

    def forward(self, x: MPCTensor):
        if comm.get().get_rank() != self.src:
            x.share -= self.running_mean
        x.share = (x.share * self.running_std_inv).div_(s, rounding_mode="trunc").to(torch.int64)
        return x

    def __call__(self, x):
        return self.forward(x)


class AffBatchNorm1d:
    def __init__(self, affine, running_mean, running_var, src=0, eps=1e-5):
        self.bias = encoder.encode(running_mean.reshape(1, -1, 1))
        self.scale = encoder.encode((1.0 / torch.sqrt(running_var.reshape(1, -1, 1) + eps)) * affine.reshape(1, 1, -1))
        self.src = src

    def forward(self, x: MPCTensor):
        if comm.get().get_rank() != self.src:
            x.share -= self.bias
        x.share = (x.share * self.scale).div_(s, rounding_mode="trunc").to(torch.int64)
        return x

    def __call__(self, x):
        return self.forward(x)

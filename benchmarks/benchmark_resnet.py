import warnings
warnings.filterwarnings("ignore")

import os
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import torch
import time
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
# crypten.init()
from crypten.encoder import FixedPointEncoder
import numpy as np
encoder = FixedPointEncoder()
# m = encoder.inv_scale
s = encoder.scale

class Linear:
    def __init__(self, weight, bias=None, src=0, device='cpu'):
        super().__init__()
        self.weight = encoder.encode(weight.to(device))
        self.bias = encoder.encode(bias.to(device)).reshape(1, 1, -1) if bias is not None else None
        self.src = src

    def forward(self, x):
        x.share = torch.matmul(x.share, self.weight).div_(s, rounding_mode="trunc")
        if self.bias is not None and comm.get().get_rank() == self.src:
            x.share += self.bias
        return x

    def __call__(self, x):
        return self.forward(x)


class Conv2d:
    def __init__(self, weight, stride=1, padding=0, src=0, device='cpu'):
        super().__init__()
        self.src = src
        self.weight = encoder.encode(weight.to(device))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x.share = torch.nn.functional.conv2d(x.share, self.weight, stride=self.stride, padding=self.padding).div_(
            s, rounding_mode="trunc")
        return x

    def __call__(self, x):
        return self.forward(x)


class BatchNorm2d:
    def __init__(self, running_mean, running_var, src=0, eps=1e-5, device='cpu'):
        super().__init__()
        self.running_mean = encoder.encode(running_mean.to(device).reshape(1, -1, 1, 1))
        self.running_std_inv = encoder.encode(1.0 / torch.sqrt(running_var.to(device).reshape(1, -1, 1, 1) + eps))
        self.src = src

    def forward(self, x):
        if comm.get().get_rank() != self.src:
            x.share -= self.running_mean
        x.share = (x.share * self.running_std_inv).div_(s, rounding_mode="trunc")
        return x

    def __call__(self, x):
        return self.forward(x)


class BasicBlock:
    def __init__(self, in_planes, planes, stride=1, src=0, device='cpu'):
        super().__init__()
        self.conv1 = Conv2d(torch.ones((planes, in_planes, 3, 3)), stride=stride, padding=1, src=src, device=device)
        self.bn1 = BatchNorm2d(torch.ones((planes,)), torch.ones((planes,)), src=src, device=device)
        self.conv2 = Conv2d(torch.ones((planes, planes, 3, 3)), stride=1, padding=1, src=src, device=device)
        self.bn2 = BatchNorm2d(torch.ones((planes,)), torch.ones((planes,)), src=src, device=device)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = Conv2d(torch.ones((planes, in_planes, 1, 1)), stride=stride, padding=0, src=src, device=device)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = out.relu()
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out.share += identity.share
        return out

    def __call__(self, x):
        return self.forward(x)


class ResNet18:
    def __init__(self, hidden_dim=64, num_classes=10, src=0, device='cpu'):
        super().__init__()
        self.conv1 = Conv2d(torch.ones((hidden_dim, 3, 3, 3)), stride=1, padding=1, src=src, device=device)
        self.bn1 = BatchNorm2d(torch.ones((hidden_dim,)), torch.ones((hidden_dim,)), src=src, device=device)

        self.layer1 = self._make_layer(hidden_dim, hidden_dim, stride=1, src=src, device=device)
        self.layer2 = self._make_layer(hidden_dim, hidden_dim * 2, stride=2, src=src, device=device)
        self.layer3 = self._make_layer(hidden_dim * 2, hidden_dim * 4, stride=2, src=src, device=device)
        self.layer4 = self._make_layer(hidden_dim * 4, hidden_dim * 8, stride=2, src=src, device=device)

        self.fc = Linear(torch.ones((hidden_dim * 8, num_classes)), src=src, device=device)

    def _make_layer(self, in_planes, planes, stride, src, device):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride=stride, src=src, device=device))
        for _ in range(1, 2):  # Add 2 blocks per layer as in ResNet18
            layers.append(BasicBlock(planes, planes, stride=1, src=src, device=device))
        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.relu()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x)

        x = x.share.mean(dim=(-2, -1), keepdim=True)  # Global average pooling
        x = self.fc(x.view(x.share.size(0), -1))
        return x

    def __call__(self, x):
        return self.forward(x)


@mpc.run_multiprocess(world_size=2)
def test_resnet18():
    rank = comm.get().get_rank()
    device = f'cuda:{rank}'
    
    model = ResNet18(hidden_dim=64, num_classes=10, src=0, device=device)
    x = crypten.cryptensor(torch.ones((1, 3, 32, 32)) * 2, src=0).to(device)  # Batch size = 4
    y = model(x)
    
    if rank == 0:
        print("Output shape:", y.share.size())


if __name__ == "__main__":
    test_resnet18()

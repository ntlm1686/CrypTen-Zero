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
encoder = FixedPointEncoder()
# m = encoder.inv_scale
s = encoder.scale


class Linear:
    def __init__(
            self,
            weight,
            bias=None,
            src=0,  # default to data source
    ):
        self.weight = encoder.encode(weight)
        # self.bias = encoder.encode(bias).reshape(1, 1, -1) if bias is not None else None
        self.src = src

    # inplace
    def forward(self, x):
        # x = x.clone()
        tmp = torch.matmul(x.share, self.weight).div_(s, rounding_mode="trunc")
        # if self.bias is not None and comm.get().get_rank() == self.src:
        #     x.share += self.bias
        return x

    def __call__(self, x):
        return self.forward(x)

@mpc.run_multiprocess(world_size=2)
def examine_arithmetic_shares():
    import time
    rank = comm.get().get_rank()


    device = f'cpu'

    x_enc = crypten.cryptensor(torch.ones(512,768), src=0).to(device)
    weight = encoder.encode(torch.ones(768, 3072)).to(device)
    # begin eval

    torch.matmul(x_enc.share, weight).div_(s, rounding_mode="trunc") 
    torch.matmul(x_enc.share, weight).div_(s, rounding_mode="trunc") 

    begin = time.time()

    for _ in range(50):
        torch.matmul(x_enc.share, weight).div_(s, rounding_mode="trunc") 
        # y = act(x_enc)
        # y = x_enc.square()
        # y = linear_slow(x_enc)
        # y = conv_slow(x_enc)
        # y = conv(x_enc)
    # y = compute(x_enc)

    elapsed = time.time() - begin
    if rank == 0:
        print(elapsed/50)
        

# examine_arithmetic_shares()

device = f'cpu'
# x_enc = crypten.cryptensor(torch.ones(512,768), src=0).to(device)
x_enc = encoder.encode(torch.ones(512,768)).to(device)
weight = encoder.encode(torch.ones(768, 3072)).to(device)
# begin eval

torch.matmul(x_enc, weight).div_(s, rounding_mode="trunc") 
torch.matmul(x_enc, weight).div_(s, rounding_mode="trunc") 

begin = time.time()

for _ in range(50):
    torch.matmul(x_enc, weight).div_(s, rounding_mode="trunc") 
    # y = act(x_enc)
    # y = x_enc.square()
    # y = linear_slow(x_enc)
    # y = conv_slow(x_enc)
    # y = conv(x_enc)
# y = compute(x_enc)

elapsed = time.time() - begin
print(elapsed/50)
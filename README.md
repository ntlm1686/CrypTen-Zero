# CrypTen-Zero

**CrypTen-Zero** is a high-performance framework for secure machine learning inference, built upon an optimized version of [CrypTen](https://github.com/facebookresearch/CrypTen). This project implements optimization strategies from the [DELPHI](https://www.usenix.org/conference/usenixsecurity19/presentation/mishra) paper, focusing on minimizing the latency of the online phase in private inference scenarios.

## Key Features

- **DELPHI-inspired Optimizations**: Divides cryptographic protocols into offline (input-independent) and online (input-dependent) phases. Most of the expensive computation is handled in the offline phase, resulting in ultra-low latency during online inference.
- **Custom MPC Layers**: A from-scratch implementation of common neural network layers (`Linear`, `Conv2d`, `SquareActivation`, etc.) designed to maximize performance in the 2-party computation (2PC) setting.
- **Extensive Benchmarking**: Includes a suite of benchmarks for various neural network architectures, including ResNet.
- **Submodule-based**: Uses a modified version of CrypTen as a git submodule to clearly separate the base framework from the performance-enhancing layers.

## Project Structure

```
├── CrypTen/              # Git submodule pointing to the modified CrypTen fork
├── crypten_zero/         # Core library code for optimized MPC layers
├── benchmarks/           # Performance benchmark scripts
├── analysis/             # Jupyter notebooks for analysis and plotting
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Clone the Repository

Clone this repository recursively to also fetch the `CrypTen` submodule:

```bash
git clone --recursive https://github.com/your-username/crypten-zero.git
cd crypten-zero
```

*If you have already cloned the repository without the `--recursive` flag, you can initialize the submodule by running:*
```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

It is recommended to use a virtual environment (e.g., conda or venv).

```bash
pip install -r requirements.txt
```

## Usage Example: Building a Network

Building a secure network with CrypTen-Zero involves composing layers from the `crypten_zero.layers` module. The process is similar to defining a model in PyTorch.

Here is how you would define a simple CNN. Note that weights and biases are loaded as plain tensors during initialization, typically from a pre-trained model.

```python
from crypten_zero.layers import Conv2d, SquareActivation, Linear
import torch

class SimpleSecureCNN:
    def __init__(self, device='cpu'):
        # 1. Load pre-trained weights (here we use random tensors as placeholders)
        conv_weight = torch.randn(16, 1, 3, 3)
        fc_weight = torch.randn(16 * 28 * 28, 10)

        # 2. Initialize the secure layers with the plaintext weights
        self.conv1 = Conv2d(weight=conv_weight, stride=1, padding=1, device=device)
        
        # The activation function is input-independent and pre-computes its randomness
        self.act1 = SquareActivation(shape=(1, 16, 28, 28), device=device)
        
        self.fc1 = Linear(weight=fc_weight, device=device)

    def forward(self, x_encrypted):
        """Defines the forward pass for an encrypted input tensor."""
        x = self.conv1(x_encrypted)
        x = self.act1(x)
        
        # Reshape/flatten the tensor for the linear layer.
        # This operation is performed on the underlying tensor share.
        x.share = x.share.view(x.share.size(0), -1)
        
        x = self.fc1(x)
        return x
```

### Running a Full Example

For complete, runnable scripts that show how to encrypt data and launch the multi-party computation, see the files in the `examples/` directory.

To run the simple CNN example:
```bash
python -m crypten.launcher examples/simple_cnn.py
```

## How It Works

CrypTen-Zero accelerates private inference by leveraging pre-computed cryptographic materials (e.g., Beaver Triples for multiplications and squares) during the online phase. The custom layers in `crypten_zero/layers.py` are designed to work with these pre-computed values, effectively turning expensive online cryptographic operations into simple arithmetic on secret shares.

This significantly reduces both the computational and communication overhead during the live inference, making it suitable for real-time applications.

## Citation

The optimization techniques used in this project are based on the following work. If you use CrypTen-Zero in your research, please consider citing:

```
@article{li2024xmlp,
  title={xMLP: Revolutionizing Private Inference with Exclusive Square Activation},
  author={Li, Jiajie and Xiong, Jinjun},
  journal={arXiv preprint arXiv:2403.08024},
  year={2024}
}
```

[**xMLP: Revolutionizing Private Inference with Exclusive Square Activation**](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=oMCzOmoAAAAJ&citation_for_view=oMCzOmoAAAAJ:9yKSN-GCB0IC)  
*Jiajie Li, Jinjun Xiong. arXiv preprint arXiv:2403.08024.*


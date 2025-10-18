# CrypTen-Zero

**CrypTen-Zero** is a high-performance framework for secure machine learning inference, built upon an optimized version of [CrypTen](https://github.com/facebookresearch/CrypTen). This project implements optimization strategies from the [DELPHI](https://www.usenix.org/conference/usenixsecurity19/presentation/mishra) paper, focusing on minimizing the latency of the online phase in private inference scenarios.

## Key Features

- **DELPHI-inspired Optimizations**: Divides cryptographic protocols into offline (input-independent) and online (input-dependent) phases. Most of the expensive computation is handled in the offline phase, resulting in ultra-low latency during online inference.
- **Custom MPC Layers**: A from-scratch implementation of common neural network layers (`Linear`, `Conv2d`, `SquareActivation`, etc.) designed to maximize performance in the 2-party computation (2PC) setting.
- **Extensive Benchmarking**: Includes a suite of benchmarks for various neural network architectures, including ResNet, Transformers, and the custom `Qute` model.
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

### 3. Running Benchmarks

The benchmark scripts are located in the `benchmarks/` directory. You can run them using the `crypten` launcher. For example, to run the Qute model benchmark:

```bash
crypten_launcher benchmarks/benchmark_qute.py
```

This will launch the 2-party computation protocol.

## How It Works

CrypTen-Zero accelerates private inference by leveraging pre-computed cryptographic materials (e.g., Beaver Triples for multiplications and squares) during the online phase. The custom layers in `crypten_zero/layers.py` are designed to work with these pre-computed values, effectively turning expensive online cryptographic operations into simple arithmetic on secret shares.

This significantly reduces both the computational and communication overhead during the live inference, making it suitable for real-time applications.

import crypten
import crypten.mpc as mpc
import torch
from crypten_zero.layers import Linear, SquareActivation

# 1. Define the Model
# This simple MLP has an input layer, one hidden layer with a square activation,
# and an output layer.
class SimpleMLP:
    def __init__(self, device='cpu'):
        # Define the layers. The weights are initialized with random tensors.
        # In a real scenario, you would load your pre-trained model weights here.
        self.fc1 = Linear(weight=torch.randn(784, 128), device=device)
        self.act1 = SquareActivation(shape=(1, 128), device=device) # Shape matches hidden layer output
        self.fc2 = Linear(weight=torch.randn(128, 10), device=device)

    def forward(self, x):
        """Defines the forward pass of the model."""
        print("Running forward pass...")
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        print("Forward pass complete.")
        return x

    def __call__(self, x):
        return self.forward(x)

# 2. Use the multiprocess decorator to run the MPC protocol
@mpc.run_multiprocess(world_size=2)
def run_mlp_inference():
    """
    This function initializes the model and data, then runs inference
    on encrypted data.
    """
    # Set the device for computation (e.g., 'cpu' or 'cuda:0')
    # Note: Both parties must have the specified device available.
    rank = crypten.communicator.get().get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    
    # Instantiate the model
    model = SimpleMLP(device=device)

    # Create a dummy input tensor (e.g., a flattened 28x28 image)
    # The input data is provided by Party 0 (src=0)
    dummy_input = torch.randn(1, 784)
    x_enc = crypten.cryptensor(dummy_input, src=0).to(device)

    # 3. Run Inference
    # The model is called like a regular PyTorch model
    output_enc = model(x_enc)

    # 4. Get the result
    # The output is still encrypted. We can decrypt it to see the result.
    output_plain = output_enc.get_plain_text()

    # In a real application, only one party might receive the output,
    # or it might be used for further encrypted computation.
    if rank == 0:
        print("Inference Output (plaintext):\n", output_plain)


if __name__ == "__main__":
    # To run this example:
    # crypten_launcher examples/simple_mlp.py
    run_mlp_inference()

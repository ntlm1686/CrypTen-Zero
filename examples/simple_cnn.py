import crypten
import crypten.mpc as mpc
import torch
from crypten_zero.layers import Conv2d, SquareActivation, Linear

# 1. Define the Model
# A simple CNN for image data, with one convolutional layer, an activation,
# and a final linear layer.
class SimpleCNN:
    def __init__(self, device='cpu'):
        # Define the layers with random weights.
        # In a real scenario, you would load your pre-trained model weights.
        self.conv1 = Conv2d(weight=torch.randn(16, 1, 3, 3), stride=1, padding=1, device=device)
        self.act1 = SquareActivation(shape=(1, 16, 28, 28), device=device) # Shape matches conv output
        
        # The output of the conv layer will be flattened before the linear layer.
        # Shape: (Batch, Channels, Height, Width) -> (1, 16, 28, 28) -> flattened to 12544
        self.fc1 = Linear(weight=torch.randn(16 * 28 * 28, 10), device=device)

    def forward(self, x):
        """Defines the forward pass of the model."""
        print("Running CNN forward pass...")
        
        # Convolution and Activation
        x = self.conv1(x)
        x = self.act1(x)
        
        # Flatten the output for the linear layer
        # We operate on the underlying share tensor for reshaping
        x.share = x.share.view(x.share.size(0), -1)
        
        # Linear layer
        x = self.fc1(x)
        
        print("Forward pass complete.")
        return x

    def __call__(self, x):
        return self.forward(x)

# 2. Use the multiprocess decorator to run the MPC protocol
@mpc.run_multiprocess(world_size=2)
def run_cnn_inference():
    """
    Initializes the CNN model and data, then runs inference.
    """
    rank = crypten.communicator.get().get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # Instantiate the model
    model = SimpleCNN(device=device)

    # Create a dummy input tensor (e.g., a 1x28x28 image like MNIST)
    # Data is provided by Party 0 (src=0)
    dummy_input = torch.randn(1, 1, 28, 28) # Batch, Channels, Height, Width
    x_enc = crypten.cryptensor(dummy_input, src=0).to(device)

    # 3. Run Inference
    output_enc = model(x_enc)

    # 4. Get the result
    output_plain = output_enc.get_plain_text()

    if rank == 0:
        print("CNN Inference Output (plaintext):\n", output_plain)


if __name__ == "__main__":
    # To run this example:
    # crypten_launcher examples/simple_cnn.py
    run_cnn_inference()

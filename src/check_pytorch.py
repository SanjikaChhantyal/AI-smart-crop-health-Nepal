"""Check that PyTorch and torchvision work correctly."""

import platform

import torch
import torchvision


def main() -> None:
    """Print PyTorch configuration and test a basic calculation."""

    print("PyTorch environment check")
    print("-------------------------")
    print(f"Operating system: {platform.system()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Selected device: {device}")

    first_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    second_tensor = torch.tensor([4.0, 5.0, 6.0], device=device)

    result = first_tensor + second_tensor

    print(f"Test calculation: {result.cpu().tolist()}")
    print("\nPyTorch installed successfully!")


if __name__ == "__main__":
    main()
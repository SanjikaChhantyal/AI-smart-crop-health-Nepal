"""Check that MobileNetV3 transfer learning works on this computer."""

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]


def select_device() -> torch.device:
    """Select Apple GPU when available, otherwise use CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_model() -> tuple[nn.Module, MobileNet_V3_Small_Weights]:
    """Load pretrained MobileNetV3 and adapt it to three potato classes."""

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights,
    )

    # Freeze the original pretrained feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Unfreeze the last three feature blocks so they can adapt
    # to potato leaf patterns.
    for feature_block in model.features[-3:]:
        for parameter in feature_block.parameters():
            parameter.requires_grad = True

    # Replace the original 1,000-class ImageNet output layer
    # with our three potato disease classes.
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        len(CLASS_NAMES),
    )

    return model, weights


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def main() -> None:
    """Create the advanced model and run one test prediction."""

    print("MobileNetV3 transfer-learning check")
    print("-----------------------------------")

    device = select_device()

    print(f"Selected device: {device}")

    model, weights = create_model()
    model = model.to(device)
    model.eval()

    total_parameters, trainable_parameters = count_parameters(
        model
    )

    print(f"Classes: {CLASS_NAMES}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # Create one artificial image with the required shape:
    # batch size 1, RGB channels 3, height 224, width 224.
    test_image = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    with torch.inference_mode():
        output = model(test_image)
        probabilities = torch.softmax(
            output,
            dim=1,
        )

    print(f"Model output shape: {list(output.shape)}")
    print(
        "Test probabilities:",
        probabilities.cpu().numpy().round(4).tolist(),
    )

    print("\nRecommended pretrained preprocessing:")
    print(weights.transforms())

    print("\nMobileNetV3 test completed successfully!")


if __name__ == "__main__":
    main()
"""Load the trained MobileNetV3 model and predict one potato leaf image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


MODEL_PATH = Path("models/advanced_mobilenet_v3.pth")

DISPLAY_NAMES = {
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Healthy Potato Leaf",
}

IMAGE_SIZE = 224


def select_device() -> torch.device:
    """Select Apple GPU when available, otherwise use CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_prediction_transform(
    normalisation_mean: list[float],
    normalisation_std: list[float],
) -> transforms.Compose:
    """Create the same preprocessing used during evaluation."""

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalisation_mean,
                std=normalisation_std,
            ),
        ]
    )


def load_trained_model(
    model_path: Path = MODEL_PATH,
) -> tuple[nn.Module, list[str], transforms.Compose, torch.device]:
    """Load the saved model checkpoint and its metadata."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    device = select_device()

    checkpoint: dict[str, Any] = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]

    model = mobilenet_v3_small(
        weights=None,
    )

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        len(class_names),
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)
    model.eval()

    prediction_transform = create_prediction_transform(
        checkpoint["normalisation_mean"],
        checkpoint["normalisation_std"],
    )

    return (
        model,
        class_names,
        prediction_transform,
        device,
    )


def validate_image(image: Image.Image) -> Image.Image:
    """Validate and convert an uploaded image to RGB."""

    if image.width < 50 or image.height < 50:
        raise ValueError(
            "The image is too small. Please use an image "
            "that is at least 50 × 50 pixels."
        )

    return image.convert("RGB")


def predict_potato_disease(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    prediction_transform: transforms.Compose,
    device: torch.device,
) -> dict[str, Any]:
    """Predict the disease class and confidence for one image."""

    validated_image = validate_image(image)

    image_tensor = prediction_transform(
        validated_image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

    predicted_index = int(
        probabilities.argmax().item()
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index].item()
    )

    class_probabilities = {
        DISPLAY_NAMES.get(class_name, class_name): float(
            probabilities[index].item()
        )
        for index, class_name in enumerate(class_names)
    }

    return {
        "internal_class": predicted_class,
        "display_name": DISPLAY_NAMES.get(
            predicted_class,
            predicted_class,
        ),
        "confidence": confidence,
        "probabilities": class_probabilities,
    }


def predict_from_file(image_path: Path) -> dict[str, Any]:
    """Load and predict an image from a file path."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            model, classes, image_transform, device = (
                load_trained_model()
            )

            return predict_potato_disease(
                image=image,
                model=model,
                class_names=classes,
                prediction_transform=image_transform,
                device=device,
            )

    except UnidentifiedImageError as error:
        raise ValueError(
            "The selected file is not a valid image."
        ) from error
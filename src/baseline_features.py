"""Feature extraction functions for the baseline potato classifier."""

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(image_path: Path) -> np.ndarray:
    """Load an image, convert it to RGB and resize it."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    return image


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """Extract Histogram of Oriented Gradients features."""

    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    features = hog(
        grayscale_image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.astype(np.float32)


def extract_colour_features(image: np.ndarray) -> np.ndarray:
    """Extract normalised HSV colour histograms."""

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV,
    )

    histogram_features = []

    channel_settings = [
        (0, [0, 180]),
        (1, [0, 256]),
        (2, [0, 256]),
    ]

    for channel_number, value_range in channel_settings:
        histogram = cv2.calcHist(
            [hsv_image],
            [channel_number],
            None,
            [24],
            value_range,
        ).flatten()

        histogram = histogram / (
            histogram.sum() + 1e-8
        )

        histogram_features.extend(histogram)

    return np.asarray(
        histogram_features,
        dtype=np.float32,
    )


def extract_image_features(
    image_path: Path,
) -> np.ndarray:
    """Extract the complete baseline feature vector."""

    image = load_image(image_path)

    hog_features = extract_hog_features(image)
    colour_features = extract_colour_features(image)

    return np.concatenate(
        [hog_features, colour_features]
    )


def get_image_paths(
    split_directory: Path,
    class_name: str,
) -> list[Path]:
    """Return supported image paths for one class."""

    class_directory = split_directory / class_name

    if not class_directory.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_directory}"
        )

    return sorted(
        path
        for path in class_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    )
"""Inspect the potato leaf image dataset before model training."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


DATASET_DIR = Path(
    "data/raw/PlantVillage-Dataset/raw/color"
)

RESULTS_DIR = Path("results")

POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_image(image_path: Path) -> dict:
    """Check whether an image can be opened and return its properties."""

    try:
        with Image.open(image_path) as image:
            image.load()

            return {
                "file_path": str(image_path),
                "class_name": image_path.parent.name,
                "width": image.width,
                "height": image.height,
                "colour_mode": image.mode,
                "status": "valid",
                "error": "",
            }

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return {
            "file_path": str(image_path),
            "class_name": image_path.parent.name,
            "width": None,
            "height": None,
            "colour_mode": None,
            "status": "invalid",
            "error": str(error),
        }


def collect_image_information() -> pd.DataFrame:
    """Inspect every image in the three potato disease classes."""

    records = []

    for class_name in POTATO_CLASSES:
        class_directory = DATASET_DIR / class_name

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Dataset class folder not found: {class_directory}"
            )

        image_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        print(f"Checking {class_name}: {len(image_paths)} images")

        for image_path in image_paths:
            records.append(inspect_image(image_path))

    return pd.DataFrame(records)


def save_class_distribution(dataframe: pd.DataFrame) -> None:
    """Save a bar chart showing the number of valid images per class."""

    valid_images = dataframe[dataframe["status"] == "valid"]

    class_counts = (
        valid_images["class_name"]
        .value_counts()
        .reindex(POTATO_CLASSES)
    )

    plt.figure(figsize=(10, 6))
    class_counts.plot(kind="bar")

    plt.title("Potato Leaf Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    chart_path = RESULTS_DIR / "class_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Class-distribution chart saved to: {chart_path}")


def main() -> None:
    """Inspect the dataset and save the inspection results."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Potato dataset inspection")
    print("-------------------------")

    dataframe = collect_image_information()

    output_path = RESULTS_DIR / "dataset_inventory.csv"
    dataframe.to_csv(output_path, index=False)

    valid_images = dataframe[dataframe["status"] == "valid"]
    invalid_images = dataframe[dataframe["status"] == "invalid"]

    print("\nImages per class:")
    print(valid_images["class_name"].value_counts())

    print(f"\nTotal images checked: {len(dataframe)}")
    print(f"Valid images: {len(valid_images)}")
    print(f"Invalid images: {len(invalid_images)}")

    if not valid_images.empty:
        print("\nImage dimensions:")
        print(
            valid_images[["width", "height"]]
            .describe()
            .round(2)
        )

        print("\nColour modes:")
        print(valid_images["colour_mode"].value_counts())

    if not invalid_images.empty:
        print("\nInvalid image files:")
        print(invalid_images[["file_path", "error"]].to_string(index=False))

    save_class_distribution(dataframe)

    print(f"Dataset inventory saved to: {output_path}")
    print("\nDataset inspection completed successfully!")


if __name__ == "__main__":
    main()
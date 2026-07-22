"""Split the potato leaf dataset into training, validation and test sets."""

from pathlib import Path
import random
import shutil

import matplotlib.pyplot as plt
import pandas as pd


SOURCE_DIR = Path("data/raw/PlantVillage-Dataset/raw/color")
OUTPUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42


def prepare_output_directories() -> None:
    """Create clean train, validation and test directories."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete only previously generated split folders.
    # The raw dataset is never modified.
    for split_name in ["train", "validation", "test"]:
        split_directory = OUTPUT_DIR / split_name

        if split_directory.exists():
            shutil.rmtree(split_directory)

        for class_name in POTATO_CLASSES:
            (split_directory / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )


def get_image_paths(class_name: str) -> list[Path]:
    """Return all supported image files for one class."""

    class_directory = SOURCE_DIR / class_name

    if not class_directory.exists():
        raise FileNotFoundError(
            f"Class directory does not exist: {class_directory}"
        )

    image_paths = [
        path
        for path in class_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]

    return sorted(image_paths)


def calculate_split_sizes(total_images: int) -> tuple[int, int, int]:
    """Calculate the train, validation and test sizes."""

    train_size = round(total_images * TRAIN_RATIO)
    validation_size = round(total_images * VALIDATION_RATIO)
    test_size = total_images - train_size - validation_size

    return train_size, validation_size, test_size


def copy_images(
    image_paths: list[Path],
    class_name: str,
    split_name: str,
    records: list[dict],
) -> None:
    """Copy images into one split and record the operation."""

    destination_directory = OUTPUT_DIR / split_name / class_name

    for source_path in image_paths:
        destination_path = destination_directory / source_path.name

        shutil.copy2(source_path, destination_path)

        records.append(
            {
                "file_name": source_path.name,
                "class_name": class_name,
                "split": split_name,
                "source_path": str(source_path),
                "destination_path": str(destination_path),
            }
        )


def split_one_class(
    class_name: str,
    class_number: int,
    records: list[dict],
) -> None:
    """Randomly split the images belonging to one class."""

    image_paths = get_image_paths(class_name)

    # A fixed seed makes the split reproducible.
    random_generator = random.Random(RANDOM_SEED + class_number)
    random_generator.shuffle(image_paths)

    train_size, validation_size, test_size = calculate_split_sizes(
        len(image_paths)
    )

    train_end = train_size
    validation_end = train_size + validation_size

    training_images = image_paths[:train_end]
    validation_images = image_paths[train_end:validation_end]
    test_images = image_paths[validation_end:]

    copy_images(
        training_images,
        class_name,
        "train",
        records,
    )

    copy_images(
        validation_images,
        class_name,
        "validation",
        records,
    )

    copy_images(
        test_images,
        class_name,
        "test",
        records,
    )

    print(f"\n{class_name}")
    print(f"  Total:      {len(image_paths)}")
    print(f"  Train:      {len(training_images)}")
    print(f"  Validation: {len(validation_images)}")
    print(f"  Test:       {len(test_images)}")

    if len(test_images) != test_size:
        raise RuntimeError(
            f"Incorrect test split size for {class_name}"
        )


def save_split_results(dataframe: pd.DataFrame) -> None:
    """Save the split manifest, distribution table and chart."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = RESULTS_DIR / "split_manifest.csv"
    dataframe.to_csv(manifest_path, index=False)

    distribution = pd.crosstab(
        dataframe["class_name"],
        dataframe["split"],
    )

    distribution = distribution.reindex(
        index=POTATO_CLASSES,
        columns=["train", "validation", "test"],
        fill_value=0,
    )

    distribution_path = RESULTS_DIR / "split_distribution.csv"
    distribution.to_csv(distribution_path)

    distribution.plot(
        kind="bar",
        figsize=(11, 6),
    )

    plt.title("Potato Dataset Train, Validation and Test Split")
    plt.xlabel("Potato Leaf Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    chart_path = RESULTS_DIR / "split_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print("\nSplit distribution:")
    print(distribution)

    print(f"\nManifest saved to: {manifest_path}")
    print(f"Distribution saved to: {distribution_path}")
    print(f"Chart saved to: {chart_path}")


def verify_split(dataframe: pd.DataFrame) -> None:
    """Verify that every source image appears exactly once."""

    duplicated_sources = dataframe["source_path"].duplicated().sum()

    if duplicated_sources > 0:
        raise RuntimeError(
            f"{duplicated_sources} images appear in multiple splits."
        )

    expected_total = sum(
        len(get_image_paths(class_name))
        for class_name in POTATO_CLASSES
    )

    if len(dataframe) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} images but split "
            f"{len(dataframe)} images."
        )

    print("\nVerification:")
    print(f"Total source images: {expected_total}")
    print(f"Total split images:  {len(dataframe)}")
    print("Duplicate images across splits: 0")


def main() -> None:
    """Create and verify the dataset split."""

    print("Potato dataset splitting")
    print("------------------------")
    print(f"Random seed: {RANDOM_SEED}")
    print("Split ratio: 70% train, 15% validation, 15% test")

    prepare_output_directories()

    records: list[dict] = []

    for class_number, class_name in enumerate(POTATO_CLASSES):
        split_one_class(
            class_name,
            class_number,
            records,
        )

    dataframe = pd.DataFrame(records)

    verify_split(dataframe)
    save_split_results(dataframe)

    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    main()
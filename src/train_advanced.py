"""Train and evaluate an advanced MobileNetV3 potato disease classifier."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


DATA_DIRECTORY = Path("data/processed")
MODEL_DIRECTORY = Path("models")
RESULTS_DIRECTORY = Path("results")

MODEL_PATH = MODEL_DIRECTORY / "advanced_mobilenet_v3.pth"

EXPECTED_CLASSES = {
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
}

IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
RANDOM_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_DEVIATION = [0.229, 0.224, 0.225]


def set_random_seeds() -> None:
    """Set random seeds to make training more reproducible."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def select_device() -> torch.device:
    """Use Apple MPS when available, otherwise use the CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_transforms() -> dict[str, transforms.Compose]:
    """Create training augmentation and evaluation preprocessing."""

    training_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.75, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.20),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.20,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STANDARD_DEVIATION,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STANDARD_DEVIATION,
            ),
        ]
    )

    return {
        "train": training_transform,
        "validation": evaluation_transform,
        "test": evaluation_transform,
    }


def create_datasets() -> dict[str, datasets.ImageFolder]:
    """Load the train, validation and test image folders."""

    image_transforms = create_transforms()

    dataset_collection = {
        split_name: datasets.ImageFolder(
            root=DATA_DIRECTORY / split_name,
            transform=image_transforms[split_name],
        )
        for split_name in ["train", "validation", "test"]
    }

    training_classes = set(
        dataset_collection["train"].classes
    )

    if training_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Unexpected class folders. "
            f"Found: {dataset_collection['train'].classes}"
        )

    training_mapping = dataset_collection["train"].class_to_idx

    for split_name in ["validation", "test"]:
        if (
            dataset_collection[split_name].class_to_idx
            != training_mapping
        ):
            raise ValueError(
                f"Class mapping differs in {split_name} split."
            )

    return dataset_collection


def create_data_loaders(
    dataset_collection: dict[str, datasets.ImageFolder],
) -> dict[str, DataLoader]:
    """Create data loaders for every dataset split."""

    random_generator = torch.Generator()
    random_generator.manual_seed(RANDOM_SEED)

    return {
        "train": DataLoader(
            dataset_collection["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=random_generator,
        ),
        "validation": DataLoader(
            dataset_collection["validation"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            dataset_collection["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
    }


def calculate_class_weights(
    training_dataset: datasets.ImageFolder,
    device: torch.device,
) -> torch.Tensor:
    """Calculate weights to reduce the effect of class imbalance."""

    class_counts = np.bincount(
        training_dataset.targets,
        minlength=len(training_dataset.classes),
    )

    class_weights = len(training_dataset.targets) / (
        len(class_counts) * class_counts
    )

    print("\nTraining images and class weights:")

    for class_name, class_index in (
        training_dataset.class_to_idx.items()
    ):
        print(
            f"{class_name}: "
            f"{class_counts[class_index]} images, "
            f"weight {class_weights[class_index]:.4f}"
        )

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )


def create_model(number_of_classes: int) -> nn.Module:
    """Create the pretrained MobileNetV3 transfer-learning model."""

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights,
    )

    # Freeze most of the pretrained feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Fine-tune the final three feature blocks.
    for feature_block in model.features[-3:]:
        for parameter in feature_block.parameters():
            parameter.requires_grad = True

    output_input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        output_input_features,
        number_of_classes,
    )

    return model


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float, float]:
    """Run one training or evaluation epoch."""

    training_mode = optimizer is not None

    if training_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if training_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training_mode):
            outputs = model(images)
            loss = loss_function(outputs, labels)

            if training_mode:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        true_labels.extend(
            labels.detach().cpu().tolist()
        )

        predicted_labels.extend(
            predictions.detach().cpu().tolist()
        )

    average_loss = total_loss / len(data_loader.dataset)

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    _, _, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(data_loader.dataset.classes))
        ),
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def train_model(
    model: nn.Module,
    data_loaders: dict[str, DataLoader],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, int, float]:
    """Train with validation monitoring and early stopping."""

    optimizer = AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    best_validation_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_records: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        print("-" * 45)

        train_loss, train_accuracy, train_f1 = run_epoch(
            model=model,
            data_loader=data_loaders["train"],
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy, validation_f1 = (
            run_epoch(
                model=model,
                data_loader=data_loaders["validation"],
                loss_function=loss_function,
                device=device,
            )
        )

        scheduler.step(validation_loss)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history_records.append(
            {
                "epoch": epoch,
                "training_loss": train_loss,
                "training_accuracy": train_accuracy,
                "training_macro_f1": train_f1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_f1,
                "learning_rate": current_learning_rate,
            }
        )

        print(
            f"Training   - loss: {train_loss:.4f}, "
            f"accuracy: {train_accuracy:.4f}, "
            f"macro F1: {train_f1:.4f}"
        )

        print(
            f"Validation - loss: {validation_loss:.4f}, "
            f"accuracy: {validation_accuracy:.4f}, "
            f"macro F1: {validation_f1:.4f}"
        )

        print(
            f"Learning rate: "
            f"{current_learning_rate:.8f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0

            print("New best model saved in memory.")

        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print("\nEarly stopping activated.")
            break

    model.load_state_dict(best_model_state)

    history_dataframe = pd.DataFrame(
        history_records
    )

    return (
        model,
        history_dataframe,
        best_epoch,
        best_validation_f1,
    )


def save_training_history(
    history_dataframe: pd.DataFrame,
) -> None:
    """Save training history as CSV and charts."""

    history_path = (
        RESULTS_DIRECTORY
        / "advanced_training_history.csv"
    )

    history_dataframe.to_csv(
        history_path,
        index=False,
    )

    loss_figure = plt.figure(figsize=(9, 6))

    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["training_loss"],
        label="Training loss",
    )

    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["validation_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MobileNetV3 Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    loss_chart_path = (
        RESULTS_DIRECTORY
        / "advanced_training_loss.png"
    )

    loss_figure.savefig(
        loss_chart_path,
        dpi=300,
    )

    plt.close(loss_figure)

    accuracy_figure = plt.figure(figsize=(9, 6))

    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["training_accuracy"],
        label="Training accuracy",
    )

    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["validation_accuracy"],
        label="Validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MobileNetV3 Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    accuracy_chart_path = (
        RESULTS_DIRECTORY
        / "advanced_training_accuracy.png"
    )

    accuracy_figure.savefig(
        accuracy_chart_path,
        dpi=300,
    )

    plt.close(accuracy_figure)

    print(f"Training history saved to: {history_path}")
    print(f"Loss chart saved to: {loss_chart_path}")
    print(
        f"Accuracy chart saved to: "
        f"{accuracy_chart_path}"
    )


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    dataset: datasets.ImageFolder,
    loss_function: nn.Module,
    device: torch.device,
    split_name: str,
) -> dict:
    """Evaluate the trained model and save detailed results."""

    model.eval()

    total_loss = 0.0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidence_scores: list[float] = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, predictions = probabilities.max(
                dim=1
            )

            total_loss += loss.item() * images.size(0)

            true_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

            confidence_scores.extend(
                confidence.cpu().tolist()
            )

    average_loss = total_loss / len(dataset)

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(len(dataset.classes))),
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(len(dataset.classes))),
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        RESULTS_DIRECTORY
        / f"advanced_{split_name}_report.csv"
    )

    report_dataframe.to_csv(report_path)

    file_paths = [
        file_path
        for file_path, _ in dataset.samples
    ]

    predictions_dataframe = pd.DataFrame(
        {
            "file_path": file_paths,
            "actual_class": [
                dataset.classes[label]
                for label in true_labels
            ],
            "predicted_class": [
                dataset.classes[label]
                for label in predicted_labels
            ],
            "confidence": confidence_scores,
            "correct": np.asarray(true_labels)
            == np.asarray(predicted_labels),
        }
    )

    predictions_path = (
        RESULTS_DIRECTORY
        / f"advanced_{split_name}_predictions.csv"
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(dataset.classes))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=dataset.classes,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    display.plot(
        ax=axis,
        values_format="d",
        xticks_rotation=20,
    )

    axis.set_title(
        f"Advanced MobileNetV3 Confusion Matrix – "
        f"{split_name.title()}"
    )

    figure.tight_layout()

    confusion_matrix_path = (
        RESULTS_DIRECTORY
        / f"advanced_{split_name}_confusion_matrix.png"
    )

    figure.savefig(
        confusion_matrix_path,
        dpi=300,
    )

    plt.close(figure)

    metrics = {
        "split": split_name,
        "number_of_images": len(dataset),
        "loss": float(average_loss),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    print(f"\n{split_name.title()} results")
    print("-" * 45)
    print(f"Loss:            {average_loss:.4f}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    return metrics


def main() -> None:
    """Train, evaluate and save the advanced model."""

    print("Advanced potato disease classifier")
    print("==================================")
    print("Model: MobileNetV3 Small transfer learning")

    set_random_seeds()

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = select_device()

    print(f"Selected device: {device}")

    dataset_collection = create_datasets()
    data_loaders = create_data_loaders(
        dataset_collection
    )

    class_names = dataset_collection["train"].classes

    print(f"Class order: {class_names}")
    print(
        f"Training images: "
        f"{len(dataset_collection['train'])}"
    )
    print(
        f"Validation images: "
        f"{len(dataset_collection['validation'])}"
    )
    print(
        f"Test images: "
        f"{len(dataset_collection['test'])}"
    )

    class_weights = calculate_class_weights(
        dataset_collection["train"],
        device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    model = create_model(
        number_of_classes=len(class_names)
    )

    model = model.to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_parameters:,}"
    )

    start_time = time.time()

    (
        model,
        training_history,
        best_epoch,
        best_validation_f1,
    ) = train_model(
        model=model,
        data_loaders=data_loaders,
        loss_function=loss_function,
        device=device,
    )

    save_training_history(training_history)

    validation_metrics = evaluate_model(
        model=model,
        data_loader=data_loaders["validation"],
        dataset=dataset_collection["validation"],
        loss_function=loss_function,
        device=device,
        split_name="validation",
    )

    test_metrics = evaluate_model(
        model=model,
        data_loader=data_loaders["test"],
        dataset=dataset_collection["test"],
        loss_function=loss_function,
        device=device,
        split_name="test",
    )

    checkpoint = {
        "model_name": "MobileNetV3 Small",
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": (
            dataset_collection["train"].class_to_idx
        ),
        "image_size": IMAGE_SIZE,
        "normalisation_mean": IMAGENET_MEAN,
        "normalisation_std": (
            IMAGENET_STANDARD_DEVIATION
        ),
        "best_epoch": best_epoch,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    elapsed_time = time.time() - start_time

    summary = {
        "model_name": (
            "MobileNetV3 Small transfer learning"
        ),
        "device": str(device),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "epochs_completed": int(
            len(training_history)
        ),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(
            best_validation_f1
        ),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "training_images": len(
            dataset_collection["train"]
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": float(elapsed_time),
    }

    metrics_path = (
        RESULTS_DIRECTORY
        / "advanced_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=4,
        )

    print(f"\nBest epoch: {best_epoch}")
    print(
        f"Best validation macro F1: "
        f"{best_validation_f1:.4f}"
    )
    print(f"Model saved locally to: {MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(
        f"Total runtime: "
        f"{elapsed_time / 60:.2f} minutes"
    )

    print(
        "\nAdvanced model training and "
        "evaluation completed successfully!"
    )


if __name__ == "__main__":
    main()
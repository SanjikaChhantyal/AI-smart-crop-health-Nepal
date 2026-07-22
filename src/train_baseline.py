"""Train and evaluate the baseline HOG and SVM potato classifier."""

from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baseline_features import (
    CLASS_NAMES,
    IMAGE_SIZE,
    extract_image_features,
    get_image_paths,
)


DATA_DIRECTORY = Path("data/processed")
MODEL_DIRECTORY = Path("models")
RESULTS_DIRECTORY = Path("results")

MODEL_PATH = MODEL_DIRECTORY / "baseline_hog_svm.joblib"

RANDOM_SEED = 42


def load_split(
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features and labels from one dataset split."""

    split_directory = DATA_DIRECTORY / split_name

    features = []
    labels = []
    file_paths = []

    print(f"\nLoading {split_name} split")
    print("-" * 40)

    total_processed = 0

    for class_name in CLASS_NAMES:
        image_paths = get_image_paths(
            split_directory,
            class_name,
        )

        print(
            f"{class_name}: {len(image_paths)} images"
        )

        for image_path in image_paths:
            try:
                image_features = extract_image_features(
                    image_path
                )

                features.append(image_features)
                labels.append(class_name)
                file_paths.append(str(image_path))

                total_processed += 1

                if total_processed % 100 == 0:
                    print(
                        f"  Processed {total_processed} images..."
                    )

            except (ValueError, OSError) as error:
                print(
                    f"Skipping {image_path}: {error}"
                )

    if not features:
        raise RuntimeError(
            f"No features extracted from {split_name}"
        )

    feature_array = np.asarray(
        features,
        dtype=np.float32,
    )

    label_array = np.asarray(labels)

    print(
        f"Completed {split_name}: "
        f"{len(feature_array)} images"
    )

    print(
        f"Feature vector size: "
        f"{feature_array.shape[1]}"
    )

    return feature_array, label_array, file_paths


def create_baseline_model() -> Pipeline:
    """Create the baseline scaling and SVM pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    features: np.ndarray,
    true_labels: np.ndarray,
    file_paths: list[str],
    split_name: str,
) -> dict:
    """Evaluate the model and save reports and predictions."""

    predicted_labels = model.predict(features)
    predicted_probabilities = model.predict_proba(
        features
    )

    confidence_scores = (
        predicted_probabilities.max(axis=1)
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=CLASS_NAMES,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        RESULTS_DIRECTORY
        / f"baseline_{split_name}_report.csv"
    )

    report_dataframe.to_csv(report_path)

    predictions_dataframe = pd.DataFrame(
        {
            "file_path": file_paths,
            "actual_class": true_labels,
            "predicted_class": predicted_labels,
            "confidence": confidence_scores,
            "correct": (
                true_labels == predicted_labels
            ),
        }
    )

    predictions_path = (
        RESULTS_DIRECTORY
        / f"baseline_{split_name}_predictions.csv"
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
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
        f"Baseline SVM Confusion Matrix – "
        f"{split_name.title()}"
    )

    figure.tight_layout()

    confusion_matrix_path = (
        RESULTS_DIRECTORY
        / f"baseline_{split_name}_confusion_matrix.png"
    )

    figure.savefig(
        confusion_matrix_path,
        dpi=300,
    )

    plt.close(figure)

    metrics = {
        "split": split_name,
        "number_of_images": int(len(true_labels)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(weighted_f1),
    }

    print(f"\n{split_name.title()} results")
    print("-" * 40)
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    print(
        f"Classification report saved to: "
        f"{report_path}"
    )

    print(
        f"Predictions saved to: "
        f"{predictions_path}"
    )

    print(
        f"Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    return metrics


def main() -> None:
    """Train, evaluate and save the baseline model."""

    print("Baseline potato disease classifier")
    print("==================================")
    print("Method: HOG + HSV colour features + SVM")

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.time()

    training_features, training_labels, _ = (
        load_split("train")
    )

    validation_features, validation_labels, validation_paths = (
        load_split("validation")
    )

    test_features, test_labels, test_paths = (
        load_split("test")
    )

    print("\nTraining baseline SVM model...")
    model = create_baseline_model()

    model.fit(
        training_features,
        training_labels,
    )

    print("Training completed.")

    validation_metrics = evaluate_model(
        model,
        validation_features,
        validation_labels,
        validation_paths,
        "validation",
    )

    test_metrics = evaluate_model(
        model,
        test_features,
        test_labels,
        test_paths,
        "test",
    )

    model_bundle = {
        "model": model,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "feature_method": (
            "HOG plus normalised HSV histograms"
        ),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    elapsed_time = time.time() - start_time

    summary = {
        "model_name": "Baseline HOG and SVM",
        "feature_method": (
            "HOG plus normalised HSV histograms"
        ),
        "classifier": "SVC with RBF kernel",
        "class_weight": "balanced",
        "image_size": list(IMAGE_SIZE),
        "training_images": int(
            len(training_labels)
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "total_runtime_seconds": float(
            elapsed_time
        ),
    }

    summary_path = (
        RESULTS_DIRECTORY
        / "baseline_metrics.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=4,
        )

    print(f"\nModel saved locally to: {MODEL_PATH}")
    print(f"Metrics saved to: {summary_path}")
    print(
        f"Total runtime: "
        f"{elapsed_time / 60:.2f} minutes"
    )

    print(
        "\nBaseline model training and "
        "evaluation completed successfully!"
    )


if __name__ == "__main__":
    main()
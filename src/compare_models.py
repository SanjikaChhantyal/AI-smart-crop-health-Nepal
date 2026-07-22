"""Compare baseline and advanced potato disease classifiers."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIRECTORY = Path("results")
MODELS_DIRECTORY = Path("models")

BASELINE_METRICS_PATH = (
    RESULTS_DIRECTORY / "baseline_metrics.json"
)

ADVANCED_METRICS_PATH = (
    RESULTS_DIRECTORY / "advanced_metrics.json"
)


def load_json(file_path: Path) -> dict:
    """Load and return a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required metrics file not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        return json.load(input_file)


def get_file_size_megabytes(file_path: Path) -> float | None:
    """Return a model file's size in megabytes."""

    if not file_path.exists():
        return None

    return file_path.stat().st_size / (1024 * 1024)


def build_comparison_table(
    baseline_data: dict,
    advanced_data: dict,
) -> pd.DataFrame:
    """Create the baseline-versus-advanced comparison table."""

    baseline_metrics = baseline_data["test_metrics"]
    advanced_metrics = advanced_data["test_metrics"]

    if (
        baseline_metrics["number_of_images"]
        != advanced_metrics["number_of_images"]
    ):
        raise ValueError(
            "The two models were not evaluated on "
            "the same number of test images."
        )

    records = [
        {
            "model": "Baseline HOG + SVM",
            "test_images": baseline_metrics[
                "number_of_images"
            ],
            "accuracy_percent": (
                baseline_metrics["accuracy"] * 100
            ),
            "macro_precision_percent": (
                baseline_metrics["macro_precision"] * 100
            ),
            "macro_recall_percent": (
                baseline_metrics["macro_recall"] * 100
            ),
            "macro_f1_percent": (
                baseline_metrics["macro_f1"] * 100
            ),
            "weighted_f1_percent": (
                baseline_metrics["weighted_f1"] * 100
            ),
            "runtime_seconds": baseline_data[
                "total_runtime_seconds"
            ],
            "model_size_mb": get_file_size_megabytes(
                MODELS_DIRECTORY
                / "baseline_hog_svm.joblib"
            ),
        },
        {
            "model": "MobileNetV3 Transfer Learning",
            "test_images": advanced_metrics[
                "number_of_images"
            ],
            "accuracy_percent": (
                advanced_metrics["accuracy"] * 100
            ),
            "macro_precision_percent": (
                advanced_metrics["macro_precision"] * 100
            ),
            "macro_recall_percent": (
                advanced_metrics["macro_recall"] * 100
            ),
            "macro_f1_percent": (
                advanced_metrics["macro_f1"] * 100
            ),
            "weighted_f1_percent": (
                advanced_metrics["weighted_f1"] * 100
            ),
            "runtime_seconds": advanced_data[
                "runtime_seconds"
            ],
            "model_size_mb": get_file_size_megabytes(
                MODELS_DIRECTORY
                / "advanced_mobilenet_v3.pth"
            ),
        },
    ]

    return pd.DataFrame(records)


def save_metric_chart(
    comparison_dataframe: pd.DataFrame,
) -> None:
    """Save a grouped chart comparing model metrics."""

    metric_columns = [
        "accuracy_percent",
        "macro_precision_percent",
        "macro_recall_percent",
        "macro_f1_percent",
        "weighted_f1_percent",
    ]

    chart_dataframe = comparison_dataframe.set_index(
        "model"
    )[metric_columns].transpose()

    chart_dataframe.index = [
        "Accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
        "Weighted F1",
    ]

    axis = chart_dataframe.plot(
        kind="bar",
        figsize=(12, 7),
    )

    axis.set_title(
        "Baseline versus Advanced Model Performance"
    )

    axis.set_xlabel("Evaluation Metric")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(
        axis="x",
        rotation=20,
    )

    for container in axis.containers:
        axis.bar_label(
            container,
            fmt="%.1f",
            fontsize=8,
        )

    plt.legend(
        title="Model",
        loc="lower right",
    )

    plt.tight_layout()

    chart_path = (
        RESULTS_DIRECTORY
        / "model_comparison.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
    )

    plt.close()

    print(f"Comparison chart saved to: {chart_path}")


def save_improvements(
    comparison_dataframe: pd.DataFrame,
) -> None:
    """Calculate the advanced model's metric differences."""

    baseline_row = comparison_dataframe.iloc[0]
    advanced_row = comparison_dataframe.iloc[1]

    metric_columns = [
        "accuracy_percent",
        "macro_precision_percent",
        "macro_recall_percent",
        "macro_f1_percent",
        "weighted_f1_percent",
    ]

    improvement_records = []

    for metric_name in metric_columns:
        difference = (
            advanced_row[metric_name]
            - baseline_row[metric_name]
        )

        improvement_records.append(
            {
                "metric": metric_name,
                "baseline_percent": baseline_row[
                    metric_name
                ],
                "advanced_percent": advanced_row[
                    metric_name
                ],
                "difference_percentage_points": difference,
            }
        )

    improvement_dataframe = pd.DataFrame(
        improvement_records
    )

    output_path = (
        RESULTS_DIRECTORY
        / "model_improvements.csv"
    )

    improvement_dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Improvement table saved to: {output_path}")


def main() -> None:
    """Compare both models and save the evidence."""

    print("Potato disease model comparison")
    print("===============================")

    baseline_data = load_json(
        BASELINE_METRICS_PATH
    )

    advanced_data = load_json(
        ADVANCED_METRICS_PATH
    )

    comparison_dataframe = build_comparison_table(
        baseline_data,
        advanced_data,
    )

    comparison_path = (
        RESULTS_DIRECTORY
        / "model_comparison.csv"
    )

    comparison_dataframe.to_csv(
        comparison_path,
        index=False,
    )

    print("\nModel comparison:")
    print(
        comparison_dataframe.round(4).to_string(
            index=False
        )
    )

    save_metric_chart(comparison_dataframe)
    save_improvements(comparison_dataframe)

    print(
        f"Comparison table saved to: "
        f"{comparison_path}"
    )

    print(
        "\nModel comparison completed successfully!"
    )


if __name__ == "__main__":
    main()
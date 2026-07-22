"""Train and evaluate the advanced bilingual farmer-question classifier."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline


TRAINING_PATH = Path(
    "data/nlp/advanced_training_questions.csv"
)

EVALUATION_PATH = Path(
    "data/nlp/advanced_evaluation_questions.csv"
)

MODEL_PATH = Path(
    "models/advanced_nlp_intent_classifier.joblib"
)

RESULTS_DIRECTORY = Path("results")

RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.55


def normalise_question(question: str) -> str:
    """Normalise English or Nepali text without removing Devanagari."""

    question = unicodedata.normalize(
        "NFKC",
        str(question),
    )

    question = question.lower().strip()

    # Replace repeated whitespace with one space.
    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question


def load_dataset(
    dataset_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Load and validate one NLP dataset."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: "
            f"{dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path)

    required_columns = {
        "question",
        "intent",
        "language",
        "source",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "question",
            "intent",
            "language",
        ]
    ).copy()

    dataframe["question"] = (
        dataframe["question"]
        .astype(str)
        .str.strip()
    )

    dataframe["clean_question"] = (
        dataframe["question"].map(
            normalise_question
        )
    )

    dataframe["intent"] = (
        dataframe["intent"]
        .astype(str)
        .str.strip()
    )

    dataframe["language"] = (
        dataframe["language"]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["clean_question"].str.len() >= 3
    ]

    duplicated_questions = dataframe[
        "clean_question"
    ].duplicated().sum()

    if duplicated_questions:
        raise ValueError(
            f"{dataset_name} dataset contains "
            f"{duplicated_questions} duplicate questions."
        )

    return dataframe.reset_index(drop=True)


def verify_dataset_separation(
    training_dataframe: pd.DataFrame,
    evaluation_dataframe: pd.DataFrame,
) -> None:
    """Ensure the evaluation questions were not used for training."""

    training_questions = set(
        training_dataframe["clean_question"]
    )

    evaluation_questions = set(
        evaluation_dataframe["clean_question"]
    )

    overlap = training_questions.intersection(
        evaluation_questions
    )

    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} "
            "questions appear in both datasets."
        )

    training_intents = set(
        training_dataframe["intent"]
    )

    evaluation_intents = set(
        evaluation_dataframe["intent"]
    )

    if training_intents != evaluation_intents:
        raise ValueError(
            "Training and evaluation intent sets differ."
        )


def create_pipeline() -> Pipeline:
    """Create combined word and character TF-IDF features."""

    combined_features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=15000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={
            "word_tfidf": 1.2,
            "character_tfidf": 1.0,
        },
    )

    classifier = LogisticRegression(
        C=4.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            (
                "features",
                combined_features,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def run_cross_validation(
    training_dataframe: pd.DataFrame,
) -> tuple[float, float, list[float]]:
    """Run cross-validation using training questions only."""

    pipeline = create_pipeline()

    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    scores = cross_val_score(
        pipeline,
        training_dataframe["clean_question"],
        training_dataframe["intent"],
        cv=cross_validator,
        scoring="f1_macro",
    )

    return (
        float(scores.mean()),
        float(scores.std()),
        scores.tolist(),
    )


def calculate_metrics(
    true_labels,
    predicted_labels,
    intent_names: list[str],
) -> dict:
    """Calculate the main classification metrics."""

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=intent_names,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=intent_names,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }


def save_confusion_matrix(
    true_labels,
    predicted_labels,
    intent_names: list[str],
) -> Path:
    """Save the advanced NLP confusion matrix."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=intent_names,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=intent_names,
    )

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    display.plot(
        ax=axis,
        values_format="d",
        xticks_rotation=30,
    )

    axis.set_title(
        "Advanced Bilingual NLP Intent "
        "Confusion Matrix"
    )

    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY
        / "advanced_nlp_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)

    return output_path


def evaluate_by_language(
    evaluation_dataframe: pd.DataFrame,
    predicted_labels: np.ndarray,
    intent_names: list[str],
) -> pd.DataFrame:
    """Calculate separate English and Nepali results."""

    records = []

    for language in sorted(
        evaluation_dataframe["language"].unique()
    ):
        language_mask = (
            evaluation_dataframe["language"]
            == language
        )

        language_true_labels = (
            evaluation_dataframe.loc[
                language_mask,
                "intent",
            ]
        )

        language_predictions = (
            predicted_labels[language_mask.to_numpy()]
        )

        language_metrics = calculate_metrics(
            true_labels=language_true_labels,
            predicted_labels=language_predictions,
            intent_names=intent_names,
        )

        records.append(
            {
                "language": language,
                "questions": int(
                    language_mask.sum()
                ),
                **language_metrics,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Train and evaluate the advanced bilingual NLP model."""

    print("Advanced bilingual NLP classifier")
    print("=================================")

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_dataframe = load_dataset(
        TRAINING_PATH,
        "Training",
    )

    evaluation_dataframe = load_dataset(
        EVALUATION_PATH,
        "Evaluation",
    )

    verify_dataset_separation(
        training_dataframe,
        evaluation_dataframe,
    )

    print(
        f"Training questions: "
        f"{len(training_dataframe)}"
    )

    print(
        f"Evaluation questions: "
        f"{len(evaluation_dataframe)}"
    )

    print("\nTraining distribution:")
    print(
        pd.crosstab(
            training_dataframe["intent"],
            training_dataframe["language"],
        )
    )

    (
        cross_validation_mean,
        cross_validation_std,
        cross_validation_scores,
    ) = run_cross_validation(
        training_dataframe
    )

    print("\nFive-fold training cross-validation:")
    print(
        ", ".join(
            f"{score:.4f}"
            for score in cross_validation_scores
        )
    )

    print(
        f"Mean macro F1: "
        f"{cross_validation_mean:.4f} "
        f"(± {cross_validation_std:.4f})"
    )

    pipeline = create_pipeline()

    print("\nTraining advanced NLP classifier...")

    pipeline.fit(
        training_dataframe["clean_question"],
        training_dataframe["intent"],
    )

    print("Training completed.")

    predicted_labels = pipeline.predict(
        evaluation_dataframe["clean_question"]
    )

    predicted_probabilities = pipeline.predict_proba(
        evaluation_dataframe["clean_question"]
    )

    confidence_scores = (
        predicted_probabilities.max(axis=1)
    )

    intent_names = sorted(
        training_dataframe["intent"]
        .unique()
        .tolist()
    )

    evaluation_metrics = calculate_metrics(
        true_labels=evaluation_dataframe[
            "intent"
        ],
        predicted_labels=predicted_labels,
        intent_names=intent_names,
    )

    predictions_dataframe = pd.DataFrame(
        {
            "question": evaluation_dataframe[
                "question"
            ],
            "language": evaluation_dataframe[
                "language"
            ],
            "actual_intent": evaluation_dataframe[
                "intent"
            ],
            "predicted_intent": predicted_labels,
            "confidence": confidence_scores,
        }
    )

    predictions_dataframe["correct"] = (
        predictions_dataframe[
            "actual_intent"
        ]
        == predictions_dataframe[
            "predicted_intent"
        ]
    )

    predictions_dataframe["accepted"] = (
        predictions_dataframe["confidence"]
        >= CONFIDENCE_THRESHOLD
    )

    accepted_predictions = (
        predictions_dataframe[
            predictions_dataframe["accepted"]
        ]
    )

    confidence_coverage = (
        len(accepted_predictions)
        / len(predictions_dataframe)
    )

    if accepted_predictions.empty:
        accepted_accuracy = 0.0
    else:
        accepted_accuracy = float(
            accepted_predictions["correct"].mean()
        )

    predictions_path = (
        RESULTS_DIRECTORY
        / "advanced_nlp_predictions.csv"
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    report = classification_report(
        evaluation_dataframe["intent"],
        predicted_labels,
        labels=intent_names,
        target_names=intent_names,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        RESULTS_DIRECTORY
        / "advanced_nlp_report.csv"
    )

    report_dataframe.to_csv(report_path)

    language_metrics_dataframe = (
        evaluate_by_language(
            evaluation_dataframe,
            predicted_labels,
            intent_names,
        )
    )

    language_metrics_path = (
        RESULTS_DIRECTORY
        / "advanced_nlp_language_metrics.csv"
    )

    language_metrics_dataframe.to_csv(
        language_metrics_path,
        index=False,
    )

    confusion_matrix_path = (
        save_confusion_matrix(
            true_labels=evaluation_dataframe[
                "intent"
            ],
            predicted_labels=predicted_labels,
            intent_names=intent_names,
        )
    )

    metrics = {
        "model_name": (
            "Combined word and character TF-IDF "
            "with Logistic Regression"
        ),
        "training_questions": int(
            len(training_dataframe)
        ),
        "evaluation_questions": int(
            len(evaluation_dataframe)
        ),
        "number_of_intents": len(
            intent_names
        ),
        "confidence_threshold": (
            CONFIDENCE_THRESHOLD
        ),
        "evaluation_metrics": (
            evaluation_metrics
        ),
        "cross_validation_macro_f1_mean": (
            cross_validation_mean
        ),
        "cross_validation_macro_f1_std": (
            cross_validation_std
        ),
        "cross_validation_scores": (
            cross_validation_scores
        ),
        "accepted_prediction_coverage": float(
            confidence_coverage
        ),
        "accepted_prediction_accuracy": float(
            accepted_accuracy
        ),
    }

    metrics_path = (
        RESULTS_DIRECTORY
        / "advanced_nlp_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metrics,
            output_file,
            indent=4,
            ensure_ascii=False,
        )

    model_bundle = {
        "pipeline": pipeline,
        "intent_names": intent_names,
        "confidence_threshold": (
            CONFIDENCE_THRESHOLD
        ),
        "normalisation_method": (
            "Unicode NFKC, lowercase and "
            "whitespace normalisation"
        ),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("\nSeparate evaluation results")
    print("--------------------------------------")

    print(
        f"Accuracy:        "
        f"{evaluation_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro precision: "
        f"{evaluation_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro recall:    "
        f"{evaluation_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1-score:  "
        f"{evaluation_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{evaluation_metrics['weighted_f1']:.4f}"
    )

    print(
        f"\nAccepted prediction coverage: "
        f"{confidence_coverage:.2%}"
    )

    print(
        f"Accepted prediction accuracy: "
        f"{accepted_accuracy:.2%}"
    )

    print("\nResults by language:")
    print(
        language_metrics_dataframe.round(
            4
        ).to_string(index=False)
    )

    print(f"\nModel saved locally to: {MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    print(
        f"Predictions saved to: "
        f"{predictions_path}"
    )
    print(
        f"Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    print(
        "\nAdvanced NLP classifier training and "
        "evaluation completed successfully!"
    )


if __name__ == "__main__":
    main()
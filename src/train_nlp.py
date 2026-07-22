"""Train and evaluate the bilingual farmer-question intent classifier."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
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
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("data/nlp/farmer_questions.csv")
MODEL_PATH = Path("models/nlp_intent_classifier.joblib")
RESULTS_DIRECTORY = Path("results")

RANDOM_SEED = 42
TEST_SIZE = 0.25


def normalise_question(question: str) -> str:
    """Clean English or Nepali text while preserving Devanagari characters."""

    question = unicodedata.normalize("NFKC", str(question))
    question = question.lower().strip()
    question = re.sub(r"\s+", " ", question)

    return question


def load_dataset() -> pd.DataFrame:
    """Load, validate and clean the bilingual question dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"NLP dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    required_columns = {
        "question",
        "intent",
        "language",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=["question", "intent", "language"]
    ).copy()

    dataframe["question"] = dataframe["question"].map(
        normalise_question
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
        dataframe["question"].str.len() >= 3
    ]

    duplicated_rows = dataframe.duplicated(
        subset=["question"]
    ).sum()

    if duplicated_rows:
        print(
            f"Removing {duplicated_rows} duplicate questions."
        )

        dataframe = dataframe.drop_duplicates(
            subset=["question"]
        )

    intent_counts = dataframe["intent"].value_counts()

    if intent_counts.min() < 4:
        raise ValueError(
            "Every intent needs at least four questions."
        )

    return dataframe.reset_index(drop=True)


def create_pipeline() -> Pipeline:
    """Create a combined word and character TF-IDF classifier."""

    combined_features = FeatureUnion(
        [
            (
                "word_features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=10000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character_features",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "features",
                combined_features,
            ),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    C=2.0,
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

def run_cross_validation(
    dataframe: pd.DataFrame,
) -> tuple[float, float]:
    """Measure performance using four-fold stratified cross-validation."""

    pipeline = create_pipeline()

    cross_validator = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    scores = cross_val_score(
        pipeline,
        dataframe["question"],
        dataframe["intent"],
        cv=cross_validator,
        scoring="f1_macro",
    )

    mean_score = float(scores.mean())
    standard_deviation = float(scores.std())

    print("\nFour-fold cross-validation macro F1:")
    print(
        ", ".join(
            f"{score:.4f}"
            for score in scores
        )
    )

    print(
        f"Mean macro F1: {mean_score:.4f} "
        f"(± {standard_deviation:.4f})"
    )

    return mean_score, standard_deviation


def save_confusion_matrix(
    true_labels: pd.Series,
    predicted_labels,
    intent_names: list[str],
) -> None:
    """Save the NLP test confusion matrix."""

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
        "Bilingual Farmer Question Intent Confusion Matrix"
    )

    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY
        / "nlp_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)

    print(f"Confusion matrix saved to: {output_path}")


def main() -> None:
    """Train, evaluate and save the bilingual NLP classifier."""

    print("Bilingual farmer-question NLP classifier")
    print("========================================")

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset()

    print(f"Total questions: {len(dataframe)}")

    print("\nQuestions per intent:")
    print(dataframe["intent"].value_counts())

    print("\nQuestions per language:")
    print(dataframe["language"].value_counts())

    cross_validation_mean, cross_validation_std = (
        run_cross_validation(dataframe)
    )

    stratification_labels = (
        dataframe["intent"]
        + "__"
        + dataframe["language"]
    )
    training_questions, test_questions, training_labels, test_labels = (
        train_test_split(
            dataframe["question"],
            dataframe["intent"],
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=stratification_labels,
        )
    )

    print(
        f"\nTraining questions: {len(training_questions)}"
    )

    print(f"Test questions: {len(test_questions)}")

    pipeline = create_pipeline()

    print("\nTraining NLP classifier...")

    pipeline.fit(
        training_questions,
        training_labels,
    )

    print("Training completed.")

    predicted_labels = pipeline.predict(
        test_questions
    )

    predicted_probabilities = pipeline.predict_proba(
        test_questions
    )

    confidence_scores = predicted_probabilities.max(
        axis=1
    )

    accuracy = accuracy_score(
        test_labels,
        predicted_labels,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    intent_names = sorted(
        dataframe["intent"].unique().tolist()
    )

    report = classification_report(
        test_labels,
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
        / "nlp_classification_report.csv"
    )

    report_dataframe.to_csv(report_path)

    predictions_dataframe = pd.DataFrame(
        {
            "question": test_questions.reset_index(
                drop=True
            ),
            "actual_intent": test_labels.reset_index(
                drop=True
            ),
            "predicted_intent": predicted_labels,
            "confidence": confidence_scores,
        }
    )

    predictions_dataframe["correct"] = (
        predictions_dataframe["actual_intent"]
        == predictions_dataframe["predicted_intent"]
    )

    predictions_path = (
        RESULTS_DIRECTORY
        / "nlp_test_predictions.csv"
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    save_confusion_matrix(
        true_labels=test_labels,
        predicted_labels=predicted_labels,
        intent_names=intent_names,
    )

    metrics = {
        "model_name": (
            "Character TF-IDF and Logistic Regression"
        ),
        "total_questions": int(len(dataframe)),
        "training_questions": int(
            len(training_questions)
        ),
        "test_questions": int(len(test_questions)),
        "number_of_intents": int(
            dataframe["intent"].nunique()
        ),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "cross_validation_macro_f1_mean": (
            cross_validation_mean
        ),
        "cross_validation_macro_f1_std": (
            cross_validation_std
        ),
    }

    metrics_path = (
        RESULTS_DIRECTORY
        / "nlp_metrics.json"
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
        "normalisation_method": (
            "Unicode NFKC, lowercase and whitespace cleaning"
        ),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print("\nTest results")
    print("----------------------------------------")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    print(f"\nModel saved locally to: {MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")

    print(
        "\nNLP classifier training and "
        "evaluation completed successfully!"
    )


if __name__ == "__main__":
    main()
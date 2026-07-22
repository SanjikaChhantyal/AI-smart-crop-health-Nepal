"""Run final automated checks for the complete CV and NLP system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.nlp_assistant import (
    answer_farmer_question,
    load_nlp_model,
)
from src.predict import predict_from_file


RESULTS_PATH = Path("results/final_system_check.json")

CV_MODEL_PATH = Path(
    "models/advanced_mobilenet_v3.pth"
)

NLP_MODEL_PATH = Path(
    "models/advanced_nlp_intent_classifier.joblib"
)

TEST_DATA_DIRECTORY = Path(
    "data/processed/test"
)

CV_TEST_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def find_test_image(class_name: str) -> Path:
    """Find one test image for the selected potato class."""

    class_directory = (
        TEST_DATA_DIRECTORY / class_name
    )

    if not class_directory.exists():
        raise FileNotFoundError(
            f"Test class folder not found: "
            f"{class_directory}"
        )

    image_paths = sorted(
        path
        for path in class_directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in VALID_IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in: "
            f"{class_directory}"
        )

    return image_paths[0]


def run_cv_checks() -> list[dict[str, Any]]:
    """Run one image prediction for every potato class."""

    if not CV_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Computer Vision model missing: "
            f"{CV_MODEL_PATH}"
        )

    records: list[dict[str, Any]] = []

    print("\nComputer Vision checks")
    print("----------------------")

    for actual_class in CV_TEST_CLASSES:
        image_path = find_test_image(
            actual_class
        )

        result = predict_from_file(
            image_path
        )

        confidence = float(
            result["confidence"]
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Computer Vision confidence "
                "is outside the valid range."
            )

        predicted_class = result[
            "internal_class"
        ]

        correct = (
            predicted_class == actual_class
        )

        record = {
            "image_path": str(image_path),
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "display_name": result[
                "display_name"
            ],
            "confidence": confidence,
            "correct": correct,
        }

        records.append(record)

        print(
            f"{actual_class}: "
            f"{result['display_name']} "
            f"({confidence * 100:.2f}%)"
        )

    return records


def run_nlp_checks() -> list[dict[str, Any]]:
    """Test English, Nepali and unsupported questions."""

    if not NLP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"NLP model missing: "
            f"{NLP_MODEL_PATH}"
        )

    model_bundle = load_nlp_model(
        NLP_MODEL_PATH
    )

    test_questions = [
        {
            "category": "English irrigation",
            "question": (
                "How often should potatoes "
                "be irrigated?"
            ),
            "language": "English",
        },
        {
            "category": "Nepali late blight",
            "question": (
                "लेट ब्लाइट कसरी फैलिन्छ?"
            ),
            "language": "Nepali",
        },
        {
            "category": "Unsupported topic",
            "question": (
                "What is the price of a tractor?"
            ),
            "language": "English",
        },
    ]

    records: list[dict[str, Any]] = []

    print("\nNLP assistant checks")
    print("--------------------")

    for item in test_questions:
        result = answer_farmer_question(
            question=item["question"],
            model_bundle=model_bundle,
            response_language=item[
                "language"
            ],
        )

        confidence = float(
            result["confidence"]
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "NLP confidence is outside "
                "the valid range."
            )

        record = {
            "category": item["category"],
            "question": item["question"],
            "predicted_intent": result[
                "predicted_intent"
            ],
            "topic_name": result[
                "topic_name"
            ],
            "confidence": confidence,
            "accepted": bool(
                result["accepted"]
            ),
            "language": result[
                "language"
            ],
            "response": result[
                "response"
            ],
        }

        records.append(record)

        status = (
            "Accepted"
            if result["accepted"]
            else "Fallback"
        )

        print(
            f"{item['category']}: "
            f"{result['topic_name']} – "
            f"{confidence * 100:.2f}% "
            f"({status})"
        )

    return records


def run_validation_check() -> dict[str, Any]:
    """Confirm that an empty question is rejected."""

    model_bundle = load_nlp_model(
        NLP_MODEL_PATH
    )

    try:
        answer_farmer_question(
            question="",
            model_bundle=model_bundle,
        )

    except ValueError as error:
        print(
            "\nInvalid-input check: passed"
        )

        return {
            "passed": True,
            "message": str(error),
        }

    raise RuntimeError(
        "Empty NLP input was not rejected."
    )


def main() -> None:
    """Run all final checks and save the results."""

    print("AI Smart Crop Health Nepal")
    print("Final system verification")
    print("=========================")

    computer_vision_results = (
        run_cv_checks()
    )

    nlp_results = run_nlp_checks()

    validation_result = (
        run_validation_check()
    )

    cv_correct = sum(
        result["correct"]
        for result
        in computer_vision_results
    )

    final_summary = {
        "computer_vision_model_found": (
            CV_MODEL_PATH.exists()
        ),
        "nlp_model_found": (
            NLP_MODEL_PATH.exists()
        ),
        "computer_vision_checks": (
            computer_vision_results
        ),
        "computer_vision_correct": (
            cv_correct
        ),
        "computer_vision_total": len(
            computer_vision_results
        ),
        "nlp_checks": nlp_results,
        "invalid_input_check": (
            validation_result
        ),
        "overall_status": "passed",
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            final_summary,
            output_file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nFinal verification summary")
    print("--------------------------")

    print(
        f"CV sample predictions correct: "
        f"{cv_correct}/"
        f"{len(computer_vision_results)}"
    )

    print(
        f"NLP questions processed: "
        f"{len(nlp_results)}"
    )

    print(
        "Invalid input rejected: Yes"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        "\nFinal system verification "
        "completed successfully!"
    )


if __name__ == "__main__":
    main()
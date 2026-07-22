"""Prediction and bilingual response functions for the farmer assistant."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib


MODEL_PATH = Path(
    "models/advanced_nlp_intent_classifier.joblib"
)


INTENT_DISPLAY_NAMES = {
    "early_blight": {
        "English": "Potato Early Blight",
        "Nepali": "आलुको अर्ली ब्लाइट",
    },
    "late_blight": {
        "English": "Potato Late Blight",
        "Nepali": "आलुको लेट ब्लाइट",
    },
    "disease_management": {
        "English": "Disease Management",
        "Nepali": "रोग व्यवस्थापन",
    },
    "irrigation": {
        "English": "Potato Irrigation",
        "Nepali": "आलुको सिँचाइ",
    },
    "fertilizer": {
        "English": "Fertilizer Management",
        "Nepali": "मल व्यवस्थापन",
    },
    "prevention": {
        "English": "Disease Prevention",
        "Nepali": "रोग रोकथाम",
    },
}


INTENT_RESPONSES = {
    "early_blight": {
        "English": (
            "Early blight is commonly associated with dry brown spots "
            "that may develop ring-like or target-shaped patterns. "
            "Symptoms often appear first on older or lower potato leaves. "
            "Remove heavily affected leaves, improve airflow and avoid "
            "leaving infected plant material in the field. Ask a local "
            "agricultural technician to confirm the disease before using "
            "a fungicide."
        ),
        "Nepali": (
            "अर्ली ब्लाइटमा प्रायः पुराना वा तल्लो पातमा सुक्खा खैरो "
            "र घेराजस्तो दाग देखिन्छ। धेरै संक्रमित पात सुरक्षित रूपमा "
            "हटाउनुहोस्, बिरुवाबीच हावाको आवतजावत सुधार गर्नुहोस् र "
            "संक्रमित अवशेष खेतमा नछोड्नुहोस्। ढुसीनाशक प्रयोग गर्नुअघि "
            "स्थानीय कृषि प्राविधिकबाट रोग पुष्टि गराउनुहोस्।"
        ),
    },
    "late_blight": {
        "English": (
            "Late blight may produce dark, water-soaked-looking patches "
            "and can spread rapidly during cool and humid weather. Pale "
            "or white growth may appear underneath infected leaves. "
            "Separate badly affected plants, inspect nearby plants and "
            "contact an agricultural technician promptly because this "
            "disease can progress quickly."
        ),
        "Nepali": (
            "लेट ब्लाइटमा पातमा कालो वा भिजेको जस्तो दाग देखिन सक्छ। "
            "चिसो र आर्द्र मौसममा यो रोग छिटो फैलिन सक्छ र संक्रमित "
            "पातको तल सेतो तह देखिन सक्छ। धेरै संक्रमित बिरुवालाई अलग "
            "गर्नुहोस्, नजिकका बिरुवा जाँच गर्नुहोस् र छिटो कृषि "
            "प्राविधिकसँग सम्पर्क गर्नुहोस्।"
        ),
    },
    "disease_management": {
        "English": (
            "Remove severely infected leaves or plants carefully and do "
            "not leave them beside healthy crops. Clean tools after "
            "handling infected plants and check surrounding plants for "
            "similar symptoms. Avoid choosing or mixing disease-control "
            "chemicals without advice from a qualified agricultural "
            "technician."
        ),
        "Nepali": (
            "धेरै संक्रमित पात वा बिरुवालाई सावधानीपूर्वक हटाउनुहोस् र "
            "स्वस्थ बालीको नजिक नछोड्नुहोस्। संक्रमित बिरुवा छोएपछि "
            "औजार सफा गर्नुहोस् र वरपरका बिरुवामा उस्तै लक्षण जाँच "
            "गर्नुहोस्। योग्य कृषि प्राविधिकको सल्लाहबिना रोग नियन्त्रणको "
            "औषधि छनोट वा मिश्रण नगर्नुहोस्।"
        ),
    },
    "irrigation": {
        "English": (
            "Water potatoes according to soil moisture rather than using "
            "a fixed schedule. Irrigate near the soil and roots instead "
            "of directly wetting the leaves. Morning irrigation is often "
            "preferable because foliage can dry during the day. Avoid "
            "waterlogging, as continuously wet soil can damage roots and "
            "encourage disease."
        ),
        "Nepali": (
            "निश्चित तालिकामा मात्र नभई माटोको चिस्यान जाँच गरेर आलुमा "
            "सिँचाइ गर्नुहोस्। पातमा सिधै पानी हाल्नुको सट्टा माटो र "
            "जरातिर पानी दिनुहोस्। बिहान सिँचाइ गर्दा पात दिनभर सुक्न "
            "सक्छ। लामो समय पानी जम्न नदिनुहोस्, किनकि यसले जरा बिगार्न "
            "र रोग बढाउन सक्छ।"
        ),
    },
    "fertilizer": {
        "English": (
            "Fertilizer should be selected according to soil condition "
            "and crop requirements. A soil test can help determine which "
            "nutrients are needed. Excess nitrogen may produce excessive "
            "leaf growth and increase disease vulnerability. Follow local "
            "agricultural recommendations rather than guessing the amount."
        ),
        "Nepali": (
            "माटोको अवस्था र बालीको आवश्यकताअनुसार मल छनोट गर्नुपर्छ। "
            "माटो परीक्षणले कुन पोषक तत्व आवश्यक छ भन्ने निर्धारण गर्न "
            "मद्दत गर्छ। धेरै नाइट्रोजनले अत्यधिक पात वृद्धि गराउन र "
            "रोगको जोखिम बढाउन सक्छ। मलको मात्रा अनुमान गरेर नभई स्थानीय "
            "कृषि सिफारिसअनुसार प्रयोग गर्नुहोस्।"
        ),
    },
    "prevention": {
        "English": (
            "Reduce potato-disease risk through crop rotation, clean "
            "tools, suitable plant spacing and regular field inspection. "
            "Remove infected plant material, use healthy planting material "
            "and avoid keeping leaves wet for long periods. Early detection "
            "makes disease management more effective."
        ),
        "Nepali": (
            "बाली चक्र, सफा औजार, बिरुवाबीच उचित दूरी र नियमित खेत "
            "निरीक्षणले आलुको रोगको जोखिम घटाउँछ। संक्रमित अवशेष हटाउनुहोस्, "
            "स्वस्थ रोपण सामग्री प्रयोग गर्नुहोस् र पात लामो समयसम्म "
            "भिजिरहन नदिनुहोस्। रोग चाँडै पत्ता लाग्दा व्यवस्थापन प्रभावकारी "
            "हुन्छ।"
        ),
    },
}


FALLBACK_RESPONSES = {
    "English": (
        "I am not confident that I understood the question. Please ask "
        "about Early Blight, Late Blight, disease management, irrigation, "
        "fertilizer or potato-disease prevention."
    ),
    "Nepali": (
        "मैले प्रश्न सही रूपमा बुझेकोमा पर्याप्त विश्वास छैन। कृपया "
        "अर्ली ब्लाइट, लेट ब्लाइट, रोग व्यवस्थापन, सिँचाइ, मल वा आलुको "
        "रोग रोकथामबारे प्रश्न सोध्नुहोस्।"
    ),
}


def normalise_question(question: str) -> str:
    """Apply the same text normalisation used during model training."""

    question = unicodedata.normalize(
        "NFKC",
        str(question),
    )

    question = question.lower().strip()

    return re.sub(
        r"\s+",
        " ",
        question,
    )


def detect_language(question: str) -> str:
    """Detect whether the question contains Devanagari text."""

    devanagari_characters = re.findall(
        r"[\u0900-\u097F]",
        question,
    )

    if devanagari_characters:
        return "Nepali"

    return "English"


def load_nlp_model(
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Load the trained NLP model bundle."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Advanced NLP model not found: {model_path}"
        )

    model_bundle = joblib.load(model_path)

    required_items = {
        "pipeline",
        "intent_names",
        "confidence_threshold",
    }

    missing_items = required_items.difference(
        model_bundle
    )

    if missing_items:
        raise ValueError(
            "The NLP model bundle is missing: "
            f"{sorted(missing_items)}"
        )

    return model_bundle


def validate_question(question: str) -> str:
    """Validate the farmer's question."""

    cleaned_question = normalise_question(
        question
    )

    if not cleaned_question:
        raise ValueError(
            "Please enter a question before selecting Ask."
        )

    if len(cleaned_question) < 4:
        raise ValueError(
            "The question is too short. Please provide more detail."
        )

    if len(cleaned_question) > 500:
        raise ValueError(
            "The question is too long. Please keep it under 500 characters."
        )

    return cleaned_question


def answer_farmer_question(
    question: str,
    model_bundle: dict[str, Any],
    response_language: str | None = None,
) -> dict[str, Any]:
    """Classify a question and return a safe bilingual response."""

    cleaned_question = validate_question(
        question
    )

    pipeline = model_bundle["pipeline"]

    predicted_probabilities = pipeline.predict_proba(
        [cleaned_question]
    )[0]

    class_names = pipeline.named_steps[
        "classifier"
    ].classes_

    best_index = int(
        predicted_probabilities.argmax()
    )

    predicted_intent = str(
        class_names[best_index]
    )

    confidence = float(
        predicted_probabilities[best_index]
    )

    confidence_threshold = float(
        model_bundle["confidence_threshold"]
    )

    accepted = (
        confidence >= confidence_threshold
    )

    if response_language is None:
        response_language = detect_language(
            question
        )

    if response_language not in {
        "English",
        "Nepali",
    }:
        response_language = "English"

    probabilities = {
        INTENT_DISPLAY_NAMES[intent][
            response_language
        ]: float(probability)
        for intent, probability in zip(
            class_names,
            predicted_probabilities,
        )
    }

    if accepted:
        response = INTENT_RESPONSES[
            predicted_intent
        ][response_language]

    else:
        response = FALLBACK_RESPONSES[
            response_language
        ]

    return {
        "question": question,
        "clean_question": cleaned_question,
        "predicted_intent": predicted_intent,
        "topic_name": INTENT_DISPLAY_NAMES[
            predicted_intent
        ][response_language],
        "confidence": confidence,
        "confidence_threshold": (
            confidence_threshold
        ),
        "accepted": accepted,
        "language": response_language,
        "response": response,
        "probabilities": probabilities,
    }
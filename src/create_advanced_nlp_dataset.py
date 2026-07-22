"""Create separate training and evaluation datasets for the advanced NLP assistant."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd


TRAINING_OUTPUT = Path(
    "data/nlp/advanced_training_questions.csv"
)

EVALUATION_OUTPUT = Path(
    "data/nlp/advanced_evaluation_questions.csv"
)


ENGLISH_TEMPLATES = [
    "Please explain {topic}.",
    "What should a farmer know about {topic}?",
    "I need guidance on {topic}.",
    "Could you give advice about {topic}?",
]

NEPALI_TEMPLATES = [
    "कृपया {topic} बारे बताइदिनुहोस्।",
    "किसानले {topic} सम्बन्धमा के जान्नुपर्छ?",
    "मलाई {topic} बारे सल्लाह चाहिन्छ।",
    "{topic} सम्बन्धी मार्गदर्शन दिनुहोस्।",
]


TRAINING_TOPICS = {
    "early_blight": {
        "English": [
            "potato early blight",
            "brown target-shaped rings on older potato leaves",
            "circular dry spots caused by early blight",
            "signs of early blight in potato plants",
            "how early blight starts on lower leaves",
        ],
        "Nepali": [
            "आलुको अर्ली ब्लाइट",
            "पुराना पातमा देखिने गोलाकार खैरो दाग",
            "अर्ली ब्लाइटका सुरुका लक्षण",
            "आलुको पातमा घेराजस्तो दाग",
            "अर्ली ब्लाइट कसरी चिन्ने",
        ],
    },
    "late_blight": {
        "English": [
            "potato late blight",
            "dark water-soaked patches on potato leaves",
            "late blight during cool and humid weather",
            "the rapid spread of late blight",
            "white fungal growth under infected leaves",
        ],
        "Nepali": [
            "आलुको लेट ब्लाइट",
            "पातमा देखिने कालो र भिजेको जस्तो दाग",
            "चिसो र आर्द्र मौसममा फैलिने लेट ब्लाइट",
            "लेट ब्लाइट छिटो फैलिने कारण",
            "संक्रमित पातको तल देखिने सेतो ढुसी",
        ],
    },
    "disease_management": {
        "English": [
            "removing infected potato leaves safely",
            "controlling an existing potato disease",
            "what to do after potato disease is detected",
            "using fungicide only with expert advice",
            "disposing of infected potato plants",
        ],
        "Nepali": [
            "संक्रमित आलुको पात सुरक्षित रूपमा हटाउने तरिका",
            "लागिसकेको आलुको रोग नियन्त्रण गर्ने उपाय",
            "आलुमा रोग भेटिएपछि गर्नुपर्ने काम",
            "विशेषज्ञको सल्लाहमा ढुसीनाशक प्रयोग",
            "संक्रमित आलुको बिरुवा व्यवस्थापन",
        ],
    },
    "irrigation": {
        "English": [
            "how often to irrigate potatoes",
            "watering potatoes without wetting the leaves",
            "the best time of day to water potato plants",
            "the effect of too much water around potato roots",
            "maintaining suitable soil moisture for potatoes",
        ],
        "Nepali": [
            "आलुमा कति पटक सिँचाइ गर्ने",
            "पात नभिजाई आलुमा पानी हाल्ने तरिका",
            "आलुमा पानी हाल्ने उपयुक्त समय",
            "आलुको जरामा धेरै पानी हुँदा पर्ने असर",
            "आलुको लागि उचित माटोको चिस्यान",
        ],
    },
    "fertilizer": {
        "English": [
            "choosing fertilizer for potato crops",
            "when to apply fertilizer to potatoes",
            "using nitrogen fertilizer carefully",
            "testing soil before applying fertilizer",
            "signs of excessive fertilizer use",
        ],
        "Nepali": [
            "आलुको लागि उपयुक्त मल छनोट",
            "आलुमा मल हाल्ने सही समय",
            "नाइट्रोजन मल सावधानीपूर्वक प्रयोग",
            "मल हाल्नुअघि माटो परीक्षण",
            "धेरै मल प्रयोग भएको लक्षण",
        ],
    },
    "prevention": {
        "English": [
            "preventing potato diseases before they appear",
            "crop rotation for healthy potato plants",
            "field hygiene and clean farming tools",
            "proper plant spacing and airflow",
            "regular inspection for early disease symptoms",
        ],
        "Nepali": [
            "रोग लाग्नुअघि आलुको रोग रोकथाम",
            "स्वस्थ आलुको लागि बाली चक्र",
            "खेतको सरसफाइ र सफा कृषि उपकरण",
            "बिरुवाबीच उचित दूरी र हावाको आवतजावत",
            "रोगका सुरुका लक्षणका लागि नियमित निरीक्षण",
        ],
    },
}


EVALUATION_QUESTIONS = {
    "early_blight": {
        "English": [
            "My older potato leaves have dry brown rings. What could this be?",
            "Which disease creates target-like circles on the lower leaves?",
            "Does early blight normally begin on mature potato foliage?",
            "How do the first visible signs of early blight look?",
        ],
        "Nepali": [
            "पुरानो आलुको पातमा सुक्खा खैरो घेरा आएको छ, यो कुन रोग हो?",
            "तल्लो पातमा गोलो दाग बनाउने रोग कुन हो?",
            "अर्ली ब्लाइट प्रायः पुरानो पातबाट सुरु हुन्छ?",
            "अर्ली ब्लाइट सुरु हुँदा पात कस्तो देखिन्छ?",
        ],
    },
    "late_blight": {
        "English": [
            "The leaf has wet-looking dark patches after several humid days.",
            "Which potato disease can destroy plants quickly in cool weather?",
            "Why is there pale growth underneath a dark infected leaf?",
            "Can late blight move rapidly from one potato plant to another?",
        ],
        "Nepali": [
            "आर्द्र मौसमपछि पातमा कालो भिजेको दाग आएको छ।",
            "चिसो मौसममा आलु छिटो नष्ट गर्ने रोग कुन हो?",
            "कालो दाग लागेको पातको तल सेतो तह किन देखिन्छ?",
            "लेट ब्लाइट एउटा बिरुवाबाट अर्कोमा छिटो सर्छ?",
        ],
    },
    "disease_management": {
        "English": [
            "I found diseased leaves today. What immediate action should I take?",
            "How should badly infected potato plants be removed?",
            "Is it safe to leave infected foliage beside the field?",
            "Who should I consult before applying a disease-control chemical?",
        ],
        "Nepali": [
            "आज रोग लागेको पात भेटियो, तुरुन्त के गर्नुपर्छ?",
            "धेरै संक्रमित आलुको बिरुवा कसरी हटाउने?",
            "संक्रमित पात खेतको छेउमा छोड्न मिल्छ?",
            "रोग नियन्त्रणको औषधि प्रयोग गर्नुअघि कसलाई सोध्ने?",
        ],
    },
    "irrigation": {
        "English": [
            "The soil is already wet. Should I water the potatoes again?",
            "Is morning irrigation better than watering late in the evening?",
            "How can I water the roots while keeping the foliage dry?",
            "What problems can continuous waterlogging cause in potatoes?",
        ],
        "Nepali": [
            "माटो पहिले नै भिजेको छ, फेरि पानी हाल्ने?",
            "बेलुकाभन्दा बिहान सिँचाइ गर्नु राम्रो हो?",
            "पात नभिजाई जरामा कसरी पानी पुर्‍याउने?",
            "लामो समय पानी जम्दा आलुमा के समस्या हुन्छ?",
        ],
    },
    "fertilizer": {
        "English": [
            "Which nutrients should be selected after checking the soil?",
            "Can excessive nitrogen make potato plants unhealthy?",
            "At which growth stage should potato fertilizer be provided?",
            "Why should a farmer avoid guessing the fertilizer amount?",
        ],
        "Nepali": [
            "माटो जाँच गरेपछि कुन पोषक तत्व छनोट गर्ने?",
            "धेरै नाइट्रोजनले आलुको बिरुवा बिगार्छ?",
            "आलुको कुन अवस्थामा मल दिनु उपयुक्त हुन्छ?",
            "मलको मात्रा अनुमान गरेर मात्र किन हाल्नु हुँदैन?",
        ],
    },
    "prevention": {
        "English": [
            "What routine practices reduce the chance of potato disease?",
            "Should the same field be planted with potatoes every season?",
            "How does wider spacing help prevent leaf infections?",
            "Why should farming tools be cleaned between fields?",
        ],
        "Nepali": [
            "कुन नियमित कामले आलुमा रोग लाग्ने सम्भावना घटाउँछ?",
            "हरेक सिजन एउटै खेतमा आलु लगाउनु ठीक हो?",
            "बिरुवाबीच बढी दूरीले रोग रोक्न कसरी मद्दत गर्छ?",
            "एउटा खेतबाट अर्कोमा जाँदा औजार किन सफा गर्नुपर्छ?",
        ],
    },
}


def normalise_text(text: str) -> str:
    """Normalise text for reliable duplicate and overlap checking."""

    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.lower().strip().split())


def generate_training_records() -> list[dict[str, str]]:
    """Create 240 training records from controlled bilingual templates."""

    records: list[dict[str, str]] = []

    for intent, language_topics in TRAINING_TOPICS.items():
        for language, topics in language_topics.items():
            templates = (
                ENGLISH_TEMPLATES
                if language == "English"
                else NEPALI_TEMPLATES
            )

            for topic in topics:
                for template in templates:
                    records.append(
                        {
                            "question": template.format(topic=topic),
                            "intent": intent,
                            "language": language,
                            "source": "template_generated_training",
                        }
                    )

    return records


def generate_evaluation_records() -> list[dict[str, str]]:
    """Create 48 manually written evaluation records."""

    records: list[dict[str, str]] = []

    for intent, language_questions in EVALUATION_QUESTIONS.items():
        for language, questions in language_questions.items():
            for question in questions:
                records.append(
                    {
                        "question": question,
                        "intent": intent,
                        "language": language,
                        "source": "held_out_manual_evaluation",
                    }
                )

    return records


def validate_datasets(
    training_dataframe: pd.DataFrame,
    evaluation_dataframe: pd.DataFrame,
) -> None:
    """Check counts, duplicates and train/evaluation separation."""

    expected_intents = set(TRAINING_TOPICS)

    if set(training_dataframe["intent"]) != expected_intents:
        raise ValueError("Training intents are incomplete.")

    if set(evaluation_dataframe["intent"]) != expected_intents:
        raise ValueError("Evaluation intents are incomplete.")

    training_duplicates = training_dataframe[
        "question"
    ].map(normalise_text).duplicated().sum()

    evaluation_duplicates = evaluation_dataframe[
        "question"
    ].map(normalise_text).duplicated().sum()

    if training_duplicates:
        raise ValueError(
            f"Training data contains {training_duplicates} duplicates."
        )

    if evaluation_duplicates:
        raise ValueError(
            f"Evaluation data contains {evaluation_duplicates} duplicates."
        )

    training_questions = set(
        training_dataframe["question"].map(normalise_text)
    )

    evaluation_questions = set(
        evaluation_dataframe["question"].map(normalise_text)
    )

    overlap = training_questions.intersection(
        evaluation_questions
    )

    if overlap:
        raise ValueError(
            f"{len(overlap)} evaluation questions also appear in training."
        )

    training_counts = training_dataframe.groupby(
        ["intent", "language"]
    ).size()

    evaluation_counts = evaluation_dataframe.groupby(
        ["intent", "language"]
    ).size()

    if not (training_counts == 20).all():
        raise ValueError(
            "Every intent must have 20 training questions per language."
        )

    if not (evaluation_counts == 4).all():
        raise ValueError(
            "Every intent must have 4 evaluation questions per language."
        )


def main() -> None:
    """Generate, validate and save both advanced NLP datasets."""

    training_records = generate_training_records()
    evaluation_records = generate_evaluation_records()

    training_dataframe = pd.DataFrame(training_records)
    evaluation_dataframe = pd.DataFrame(evaluation_records)

    validate_datasets(
        training_dataframe,
        evaluation_dataframe,
    )

    TRAINING_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_dataframe.to_csv(
        TRAINING_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    evaluation_dataframe.to_csv(
        EVALUATION_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    print("Advanced bilingual NLP datasets created")
    print("=======================================")

    print(
        f"\nTraining questions: "
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

    print("\nEvaluation distribution:")
    print(
        pd.crosstab(
            evaluation_dataframe["intent"],
            evaluation_dataframe["language"],
        )
    )

    print("\nDuplicate training questions: 0")
    print("Duplicate evaluation questions: 0")
    print("Training/evaluation overlap: 0")

    print(f"\nTraining data saved to: {TRAINING_OUTPUT}")
    print(
        f"Evaluation data saved to: "
        f"{EVALUATION_OUTPUT}"
    )

    print(
        "\nDataset generation completed successfully!"
    )


if __name__ == "__main__":
    main()
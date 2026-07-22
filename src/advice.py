"""Bilingual information for potato leaf predictions."""

from typing import Any


DISEASE_ADVICE: dict[str, dict[str, Any]] = {
    "Potato___Early_blight": {
        "english_name": "Potato Early Blight",
        "nepali_name": "आलुको अर्ली ब्लाइट",
        "english_description": (
            "Early blight is a fungal potato disease that commonly "
            "produces brown circular marks on older leaves."
        ),
        "nepali_description": (
            "अर्ली ब्लाइट आलुमा लाग्ने ढुसीजन्य रोग हो। यसले प्रायः "
            "पुराना पातमा खैरो गोलाकार दाग देखाउँछ।"
        ),
        "english_actions": [
            "Remove heavily affected leaves and dispose of them safely.",
            "Avoid leaving infected plant material in the field.",
            "Improve spacing and airflow between plants.",
            "Avoid unnecessarily wetting the leaves during irrigation.",
            "Consult a local agricultural technician before using fungicide.",
        ],
        "nepali_actions": [
            "धेरै संक्रमित पात हटाएर सुरक्षित रूपमा व्यवस्थापन गर्नुहोस्।",
            "संक्रमित बिरुवाको अवशेष खेतमा नछोड्नुहोस्।",
            "बिरुवाबीच उचित दूरी र हावाको आवतजावत कायम गर्नुहोस्।",
            "सिँचाइ गर्दा पात अनावश्यक रूपमा भिजाउनबाट बच्नुहोस्।",
            "ढुसीनाशक प्रयोग गर्नुअघि कृषि प्राविधिकसँग परामर्श गर्नुहोस्।",
        ],
    },
    "Potato___Late_blight": {
        "english_name": "Potato Late Blight",
        "nepali_name": "आलुको लेट ब्लाइट",
        "english_description": (
            "Late blight is a serious potato disease that may spread "
            "rapidly during cool and humid conditions."
        ),
        "nepali_description": (
            "लेट ब्लाइट आलुको गम्भीर रोग हो। चिसो र धेरै आर्द्र "
            "मौसममा यो छिटो फैलिन सक्छ।"
        ),
        "english_actions": [
            "Separate severely affected plants from healthy plants.",
            "Remove infected leaves without spreading plant material.",
            "Check nearby potato plants for similar symptoms.",
            "Reduce prolonged leaf wetness where possible.",
            "Contact an agricultural technician promptly for confirmation.",
        ],
        "nepali_actions": [
            "धेरै संक्रमित बिरुवालाई स्वस्थ बिरुवाबाट अलग गर्नुहोस्।",
            "संक्रमित पात अन्यत्र नफैलिने गरी हटाउनुहोस्।",
            "नजिकका आलुका बिरुवामा पनि उस्तै लक्षण छन् कि जाँच गर्नुहोस्।",
            "सम्भव भएसम्म पात लामो समयसम्म भिजिरहन नदिनुहोस्।",
            "पुष्टिका लागि छिटो कृषि प्राविधिकसँग सम्पर्क गर्नुहोस्।",
        ],
    },
    "Potato___healthy": {
        "english_name": "Healthy Potato Leaf",
        "nepali_name": "स्वस्थ आलुको पात",
        "english_description": (
            "The model did not identify the Early Blight or Late Blight "
            "patterns represented in its training dataset."
        ),
        "nepali_description": (
            "मोडेलले तालिममा समावेश गरिएको अर्ली ब्लाइट वा लेट "
            "ब्लाइटको ढाँचा पहिचान गरेन।"
        ),
        "english_actions": [
            "Continue observing the plant regularly.",
            "Maintain suitable irrigation and field hygiene.",
            "Check again if new spots, wilting or colour changes appear.",
            "Seek expert confirmation if the plant continues to weaken.",
        ],
        "nepali_actions": [
            "बिरुवालाई नियमित रूपमा निरीक्षण गरिरहनुहोस्।",
            "उचित सिँचाइ र खेतको सरसफाइ कायम गर्नुहोस्।",
            "नयाँ दाग, ओइलाउने वा रङ परिवर्तन भए पुनः जाँच गर्नुहोस्।",
            "बिरुवा कमजोर हुँदै गएमा विशेषज्ञको सल्लाह लिनुहोस्।",
        ],
    },
}


def get_disease_advice(
    internal_class: str,
    language: str = "English",
) -> dict[str, Any]:
    """Return advice for the predicted class and selected language."""

    if internal_class not in DISEASE_ADVICE:
        raise KeyError(
            f"No advice is available for class: {internal_class}"
        )

    information = DISEASE_ADVICE[internal_class]

    if language == "नेपाली":
        return {
            "name": information["nepali_name"],
            "description": information["nepali_description"],
            "actions": information["nepali_actions"],
        }

    return {
        "name": information["english_name"],
        "description": information["english_description"],
        "actions": information["english_actions"],
    }
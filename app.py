"""Streamlit interface for the potato leaf disease classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.advice import get_disease_advice
from src.predict import (
    load_trained_model,
    predict_potato_disease,
)


MODEL_PATH = Path("models/advanced_mobilenet_v3.pth")
LOW_CONFIDENCE_THRESHOLD = 0.65


st.set_page_config(
    page_title="Potato Leaf Health Assistant",
    page_icon="🥔",
    layout="wide",
)


@st.cache_resource
def get_prediction_resources():
    """Load and cache the trained model."""

    return load_trained_model(MODEL_PATH)


def display_probability_table(
    probabilities: dict[str, float],
) -> None:
    """Display probabilities as percentages and a chart."""

    probability_dataframe = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": [
                probability * 100
                for probability in probabilities.values()
            ],
        }
    )

    probability_dataframe = probability_dataframe.sort_values(
        "Probability",
        ascending=False,
    )

    st.subheader("Class probabilities")

    st.dataframe(
        probability_dataframe.style.format(
            {"Probability": "{:.2f}%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_dataframe = probability_dataframe.set_index(
        "Class"
    )

    st.bar_chart(chart_dataframe)


def main() -> None:
    """Run the Streamlit application."""

    st.title("🥔 Potato Leaf Health Assistant")

    st.write(
        "Upload a clear potato-leaf image. The system will classify "
        "it as Early Blight, Late Blight or Healthy."
    )

    st.warning(
        "This prototype is an educational decision-support tool. "
        "Its prediction should not replace diagnosis by an "
        "agricultural technician."
    )

    language = st.radio(
        "Advice language",
        options=["English", "नेपाली"],
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a potato-leaf image",
        type=["jpg", "jpeg", "png"],
        help=(
            "Use a clear image containing one visible potato leaf. "
            "Avoid unrelated objects and extremely dark images."
        ),
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be read as an image. "
            "Please upload a valid JPG, JPEG or PNG file."
        )
        return

    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.subheader("Uploaded image")

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

        st.write(
            f"Image size: {image.width} × {image.height} pixels"
        )

    with right_column:
        st.subheader("AI analysis")

        analyse_button = st.button(
            "Analyse potato leaf",
            type="primary",
            use_container_width=True,
        )

        if not analyse_button:
            st.write(
                "Select **Analyse potato leaf** to generate a prediction."
            )
            return

        try:
            with st.spinner("Analysing the image..."):
                (
                    model,
                    class_names,
                    prediction_transform,
                    device,
                ) = get_prediction_resources()

                prediction = predict_potato_disease(
                    image=image,
                    model=model,
                    class_names=class_names,
                    prediction_transform=prediction_transform,
                    device=device,
                )

        except FileNotFoundError:
            st.error(
                "The trained model file was not found. "
                "Expected location: models/advanced_mobilenet_v3.pth"
            )
            return

        except ValueError as error:
            st.error(str(error))
            return

        confidence_percent = (
            prediction["confidence"] * 100
        )

        advice = get_disease_advice(
            prediction["internal_class"],
            language,
        )

        st.metric(
            label="Predicted condition",
            value=advice["name"],
        )

        st.metric(
            label="Model confidence",
            value=f"{confidence_percent:.2f}%",
        )

        if prediction["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                "The model has low confidence in this result. "
                "Try another clearer image and seek expert confirmation."
            )
        else:
            st.success("Prediction completed successfully.")

        st.write(advice["description"])

        st.subheader(
            "Recommended next steps"
            if language == "English"
            else "सिफारिस गरिएका अर्को कदम"
        )

        for action in advice["actions"]:
            st.write(f"- {action}")

    display_probability_table(
        prediction["probabilities"]
    )

    with st.expander("Prototype limitations"):
        st.write(
            "- The model recognises only Early Blight, Late Blight "
            "and Healthy potato leaves."
        )
        st.write(
            "- It was trained mainly using controlled-background "
            "PlantVillage images."
        )
        st.write(
            "- Unrelated plants, multiple leaves, poor lighting or "
            "unseen diseases may produce unreliable predictions."
        )
        st.write(
            "- Field validation with Nepal-specific images is still required."
        )


if __name__ == "__main__":
    main()
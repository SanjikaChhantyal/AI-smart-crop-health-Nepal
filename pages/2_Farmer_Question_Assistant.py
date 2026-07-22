"""Streamlit page for the bilingual farmer-question assistant."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.nlp_assistant import (
    answer_farmer_question,
    load_nlp_model,
)


st.set_page_config(
    page_title="Farmer Question Assistant",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Sage-green page background */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #9CAF88;
    }

    /* Transparent top header */
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Main headings and normal text */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li {
        color: #111111 !important;
    }

    /* Radio-button labels */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label span {
        color: #111111 !important;
    }

    /* Text-area background and entered text */
    [data-testid="stTextArea"] textarea {
        background-color: #E8EDE2 !important;
        color: #111111 !important;
        border: 1px solid #596A52 !important;
    }

    /* Text-area placeholder */
    [data-testid="stTextArea"] textarea::placeholder {
        color: #555555 !important;
        opacity: 1;
    }

    /* Warning message */
    [data-testid="stAlert"] {
        background-color: #DCE5D4 !important;
        color: #111111 !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div {
        color: #111111 !important;
    }

    /* Information message */
    [data-testid="stNotification"] {
        background-color: #DCE5D4 !important;
    }

    [data-testid="stNotification"] p,
    [data-testid="stNotification"] div {
        color: #111111 !important;
    }

    /* Expander text */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] p {
        color: #111111 !important;
    }

    /* Keep Ask the assistant button text white */
    .stButton > button[kind="primary"] {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #E53E3E !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_nlp_model():
    """Load and cache the advanced NLP classifier."""

    return load_nlp_model()


def main() -> None:
    """Display the bilingual farmer-question interface."""

    st.title("🌱 Farmer Question Assistant")

    st.write(
        "Ask a potato-farming question in English or Nepali. "
        "The assistant supports Early Blight, Late Blight, "
        "disease management, irrigation, fertilizer and prevention."
    )

    st.warning(
        "This is an educational prototype. Important farming and "
        "chemical-treatment decisions should be confirmed by a "
        "qualified agricultural technician."
    )

    response_language_option = st.radio(
        "Answer language",
        options=[
            "Automatic",
            "English",
            "नेपाली",
        ],
        horizontal=True,
    )

    selected_language = {
        "Automatic": None,
        "English": "English",
        "नेपाली": "Nepali",
    }[response_language_option]

    question = st.text_area(
        "Enter your farming question",
        height=130,
        max_chars=500,
        placeholder=(
            "Example: How can I stop late blight from spreading?\n"
            "उदाहरण: आलुको पातमा कालो दाग आएमा के गर्ने?"
        ),
    )

    ask_button = st.button(
        "Ask the assistant",
        type="primary",
        use_container_width=True,
    )

    if not ask_button:
        st.info(
            "Enter a question and select **Ask the assistant**."
        )

        with st.expander("Example questions"):
            st.write(
                "- What are the symptoms of Early Blight?"
            )
            st.write(
                "- How often should potatoes be irrigated?"
            )
            st.write(
                "- Should I test the soil before applying fertilizer?"
            )
            st.write(
                "- लेट ब्लाइट कसरी फैलिन्छ?"
            )
            st.write(
                "- आलुमा पानी हाल्ने राम्रो समय कुन हो?"
            )

        return

    try:
        with st.spinner(
            "Understanding the question..."
        ):
            model_bundle = get_nlp_model()

            result = answer_farmer_question(
                question=question,
                model_bundle=model_bundle,
                response_language=selected_language,
            )

    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    first_column, second_column = st.columns(
        [1, 1]
    )

    with first_column:
        st.metric(
            "Detected topic",
            result["topic_name"],
        )

    with second_column:
        st.metric(
            "Model confidence",
            f"{result['confidence'] * 100:.2f}%",
        )

    if result["accepted"]:
        st.success(
            "The question passed the confidence threshold."
        )
    else:
        st.warning(
            "The model is uncertain, so a safe fallback "
            "response is being shown."
        )

    st.subheader(
        "Assistant response"
        if result["language"] == "English"
        else "सहायकको जवाफ"
    )

    st.write(result["response"])

    probability_dataframe = pd.DataFrame(
        {
            "Topic": list(
                result["probabilities"].keys()
            ),
            "Probability": [
                probability * 100
                for probability in
                result["probabilities"].values()
            ],
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    with st.expander(
        "View topic probabilities"
    ):
        st.dataframe(
            probability_dataframe.style.format(
                {
                    "Probability": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            probability_dataframe.set_index(
                "Topic"
            )
        )

    with st.expander(
        "Model limitations"
    ):
        st.write(
            "- The assistant recognises only six supported topics."
        )
        st.write(
            "- Its separate evaluation accuracy was 70.83%."
        )
        st.write(
            "- English evaluation accuracy was 79.17%."
        )
        st.write(
            "- Nepali evaluation accuracy was 62.50%."
        )
        st.write(
            "- Low-confidence questions receive a safe fallback."
        )
        st.write(
            "- The assistant does not prescribe chemical dosages."
        )


if __name__ == "__main__":
    main()
# AI Smart Crop Health Nepal

AI Smart Crop Health Nepal is a bilingual Computer Vision and Natural Language Processing system designed to support potato farmers. The system can analyse potato-leaf images, classify leaf health conditions and answer common potato-farming questions in English or Nepali.

The project was developed as an educational AI prototype and combines image classification, bilingual text classification, confidence-based decision support and an interactive Streamlit interface.

---

## Main Features

### Potato Leaf Disease Classification

A user can upload a potato-leaf image and receive one of three predictions:

- Potato Early Blight
- Potato Late Blight
- Healthy Potato Leaf

The system displays:

- Predicted condition
- Confidence score
- Probabilities for all classes
- English or Nepali disease information
- Suggested next steps
- Low-confidence warnings

### Bilingual Farmer Question Assistant

A user can ask potato-farming questions in English or Nepali.

The assistant supports six topics:

- Early Blight
- Late Blight
- Disease management
- Irrigation
- Fertilizer management
- Disease prevention

Questions below the model’s confidence threshold receive a safe fallback instead of an unreliable answer.

---

## Project Objectives

The main objectives of this project are to:

1. Develop a working Computer Vision system for potato-leaf classification.
2. Compare a traditional baseline model with an advanced deep-learning model.
3. Develop a bilingual NLP assistant for English and Nepali farming questions.
4. Compare baseline and advanced NLP approaches.
5. Display predictions and confidence scores through an interactive interface.
6. Validate incorrect and unsupported inputs.
7. Save evaluation results for reproducibility and comparison.

---

## Dataset

### Computer Vision Dataset

The Computer Vision system uses potato-leaf images from the PlantVillage dataset.

The selected classes are:

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

Total selected images: **2,152**

| Class | Total | Training | Validation | Test |
|---|---:|---:|---:|---:|
| Potato Early Blight | 1,000 | 700 | 150 | 150 |
| Potato Late Blight | 1,000 | 700 | 150 | 150 |
| Healthy Potato Leaf | 152 | 106 | 23 | 23 |
| **Total** | **2,152** | **1,506** | **323** | **323** |

A stratified split was used to preserve the original class distribution. Duplicate images across the training, validation and test sets were checked, and no overlap was found.

### NLP Datasets

The baseline NLP dataset contains 72 bilingual questions across six intents.

The advanced NLP system uses:

- 240 bilingual training questions
- 48 separately written evaluation questions
- 20 English and 20 Nepali training questions per intent
- 4 English and 4 Nepali evaluation questions per intent
- No duplicate questions
- No training/evaluation overlap

---

## Data Preprocessing

### Computer Vision Preprocessing

The image pipeline includes:

- Image validation
- RGB conversion
- Image resizing
- Centre cropping
- Pixel normalisation
- Random resized cropping during training
- Horizontal flipping
- Limited vertical flipping
- Random rotation
- Colour augmentation
- ImageNet mean and standard-deviation normalisation

Class weights were used during advanced-model training to reduce the effect of class imbalance.

### NLP Preprocessing

The text-processing pipeline includes:

- Unicode NFKC normalisation
- Lowercase conversion for English text
- Leading and trailing whitespace removal
- Repeated-whitespace cleaning
- English and Nepali text preservation
- Word TF-IDF features
- Character TF-IDF features
- Question-length validation
- Confidence-threshold validation

---

## Computer Vision Models

### Baseline Model

The baseline Computer Vision model uses:

- Histogram of Oriented Gradients features
- Support Vector Machine classifier

The baseline represents a traditional machine-learning approach where image features are manually extracted before classification.

### Advanced Model

The advanced model uses:

- MobileNetV3 Small
- Pretrained ImageNet weights
- Transfer learning
- Data augmentation
- Class-weighted cross-entropy loss
- AdamW optimiser
- Learning-rate scheduling
- Early stopping
- Apple MPS acceleration when available

Most pretrained feature layers are initially frozen, while the final feature blocks and classifier are fine-tuned for the three potato classes.

---

## Computer Vision Results

Both models were evaluated on the same 323 test images.

| Metric | HOG + SVM Baseline | MobileNetV3 Advanced |
|---|---:|---:|
| Accuracy | 93.81% | **96.28%** |
| Macro Precision | **95.86%** | 91.15% |
| Macro Recall | 82.06% | **97.33%** |
| Macro F1-score | 86.39% | **93.66%** |
| Weighted F1-score | 93.40% | **96.36%** |

MobileNetV3 was selected for the final application because it produced higher accuracy, recall, macro F1 and weighted F1.

---

## NLP Models

### Baseline NLP Model

The baseline NLP model uses:

- Word and character TF-IDF
- Linear Support Vector Machine
- Six intent classes
- English and Nepali questions

Baseline results:

| Metric | Result |
|---|---:|
| Accuracy | 44.44% |
| Macro F1-score | 41.63% |

The small original dataset caused confusion between similar intents such as prevention, irrigation and disease management.

### Advanced NLP Model

The advanced NLP system uses:

- Expanded bilingual training data
- Separate held-out evaluation data
- Combined word and character TF-IDF
- Logistic Regression
- Balanced intent classes
- Confidence threshold of 0.55
- Safe fallback for uncertain questions

Advanced results:

| Metric | Result |
|---|---:|
| Overall Accuracy | **70.83%** |
| English Accuracy | **79.17%** |
| Nepali Accuracy | **62.50%** |
| High-confidence Coverage | 45.83% |
| Accuracy of Accepted Predictions | **95.45%** |

The confidence threshold improves safety by preventing low-confidence questions from receiving specific farming advice.

---

## Project Structure

```text
AI-smart-crop-health-Nepal/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── nlp/
│       ├── farmer_questions.csv
│       ├── advanced_training_questions.csv
│       └── advanced_evaluation_questions.csv
│
├── models/
│   ├── baseline_hog_svm.joblib
│   ├── advanced_mobilenet_v3.pth
│   ├── nlp_baseline_intent_classifier.joblib
│   └── advanced_nlp_intent_classifier.joblib
│
├── notebooks/
│
├── pages/
│   └── 2_Farmer_Question_Assistant.py
│
├── results/
│   ├── model evaluation metrics
│   ├── classification reports
│   ├── predictions
│   ├── confusion matrices
│   ├── training graphs
│   ├── model comparisons
│   └── final_system_check.json
│
├── docs/
│
└── src/
    ├── advice.py
    ├── baseline_features.py
    ├── check_setup.py
    ├── compare_models.py
    ├── create_nlp_dataset.py
    ├── create_advanced_nlp_dataset.py
    ├── final_system_check.py
    ├── inspect_dataset.py
    ├── nlp_assistant.py
    ├── predict.py
    ├── split_dataset.py
    ├── train_baseline.py
    ├── train_advanced.py
    ├── train_nlp.py
    └── train_advanced_nlp.py
```

---

## System Requirements

Recommended environment:

- Python 3.12
- macOS, Windows or Linux
- At least 4 GB of available memory
- Apple Silicon MPS, CUDA GPU or CPU
- Git
- Visual Studio Code or another Python editor

The project was developed and tested using Python 3.12 on macOS with Apple MPS acceleration.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SanjikaChhantyal/AI-smart-crop-health-Nepal.git
```

Enter the project folder:

```bash
cd AI-smart-crop-health-Nepal
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Check the environment

```bash
python -u src/check_setup.py
```

For PyTorch and device verification:

```bash
python -u src/check_pytorch.py
```

---

## Dataset Preparation

The raw PlantVillage potato classes must be placed inside:

```text
data/raw/PlantVillage-Dataset/raw/color/
```

Inspect the dataset:

```bash
python -u src/inspect_dataset.py
```

Create the training, validation and test splits:

```bash
python -u src/split_dataset.py
```

Generate the baseline NLP dataset:

```bash
python -u src/create_nlp_dataset.py
```

Generate the advanced NLP datasets:

```bash
python -u src/create_advanced_nlp_dataset.py
```

---

## Model Training

### Train the baseline Computer Vision model

```bash
python -u src/train_baseline.py
```

### Train the advanced Computer Vision model

```bash
python -u src/train_advanced.py
```

### Compare the Computer Vision models

```bash
python -u src/compare_models.py
```

### Train the baseline NLP model

```bash
python -u src/train_nlp.py
```

### Train the advanced NLP model

```bash
python -u src/train_advanced_nlp.py
```

---

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

The sidebar contains:

1. Potato Leaf Health Assistant
2. Farmer Question Assistant

---

## Run Final Verification

Execute the final automated system check as a Python module:

```bash
python -u -m src.final_system_check
```

This test verifies:

- Early Blight image prediction
- Late Blight image prediction
- Healthy-leaf image prediction
- English farmer-question processing
- Nepali farmer-question processing
- Unsupported-question fallback
- Empty-input rejection
- Model availability
- Confidence-score validation

The results are saved to:

```text
results/final_system_check.json
```

---

## Model Files

Trained model files are excluded from GitHub because they may be large or machine-generated.

Expected local files:

```text
models/baseline_hog_svm.joblib
models/advanced_mobilenet_v3.pth
models/nlp_baseline_intent_classifier.joblib
models/advanced_nlp_intent_classifier.joblib
```

They can be recreated using the training commands described above.

---

## Input Validation

The system checks for:

- Missing images
- Unsupported file formats
- Images that are too small
- Missing trained models
- Empty farming questions
- Very short questions
- Questions longer than 500 characters
- Low-confidence NLP predictions
- Unsupported farming topics

Invalid or uncertain inputs produce clear error or fallback messages.

---

## Limitations

### Computer Vision Limitations

- The classifier recognises only Early Blight, Late Blight and Healthy potato leaves.
- It was trained mainly using controlled-background PlantVillage images.
- Whole plants, unrelated crops and complex field backgrounds may reduce reliability.
- The healthy class contains fewer images than the disease classes.
- The model has not yet been extensively validated using Nepal-specific field images.
- Confidence is not the same as a confirmed agricultural diagnosis.

### NLP Limitations

- The assistant recognises only six supported topics.
- Nepali performance is lower than English performance.
- The training dataset is controlled and relatively small.
- The assistant does not understand every possible farming question.
- It does not recommend pesticide or fungicide dosages.
- Low-confidence questions receive a general fallback response.

---

## Ethical and Safety Considerations

This system is an educational decision-support prototype. It is not intended to replace a qualified agricultural technician.

The application:

- Displays confidence scores
- Warns users about uncertain predictions
- Avoids chemical dosage recommendations
- Provides a fallback for unsupported questions
- States the limitations of its training data
- Encourages expert confirmation for important decisions

---

## Future Improvements

Future development could include:

- Nepal-specific potato-field images
- More balanced healthy-leaf data
- Additional potato diseases and pests
- Object detection for multiple leaves
- Image-background segmentation
- Nepali speech input
- Larger naturally collected Nepali question datasets
- Weather and location information
- Model deployment through a cloud platform
- Expert-reviewed treatment recommendations
- Continuous field validation

---

## Disclaimer

This project is intended for education, experimentation and decision support. Predictions must not be treated as a confirmed diagnosis. Farmers should consult a qualified agricultural technician before applying agricultural chemicals or making major crop-management decisions.
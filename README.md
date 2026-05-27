# Multimodal Emotion Recognition Using Speech and Text

## Overview
This project implements a multimodal emotion recognition system using speech and text modalities. Three separate models were developed and compared:
- Speech-only model
- Text-only model
- Multimodal fusion model

The project uses the TESS dataset and performs emotion classification using deep learning techniques.

---

# Features
- Speech Emotion Recognition using MFCC + BiLSTM
- Text Emotion Recognition using Embedding + BiLSTM
- Multimodal Fusion using Concatenation
- t-SNE Visualization of Emotion Clusters
- Accuracy Comparison

---

# Technologies Used
- Python
- TensorFlow
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Dataset
Dataset used:
- Toronto Emotional Speech Set (TESS)

Place the dataset inside:

```text
dataset/TESS/
Final structure:

dataset/
└── TESS/
    ├── OAF_angry/
    ├── OAF_happy/
    ├── OAF_sad/
    ├── OAF_fear/
    ├── OAF_disgust/
    ├── OAF_neutral/
    └── ...
Project Structure
Multimodal-Emotion-Recognition/
├── dataset/
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
├── plots/
├── Results/
├── utils/
├── README.md
└── requirements.txt

How to Execute
1. Clone Repository
git clone https://github.com/satyakalpana12/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment
Windows
venv\Scripts\activate

4. Install Dependencies
pip install -r requirements.txt

5. Prepare Dataset Dataframe
python utils/data_prep.py

This script:

scans dataset folders
extracts emotion labels
creates dataframe.csv

Output:
dataset/dataframe.csv

6. Extract Speech Features
python utils/speech_features.py

This script:

loads audio files
extracts MFCC features
saves processed arrays

Outputs:

dataset/speech_X.npy
dataset/speech_y.npy

7. Train Speech Model
python models/speech_pipeline/train.py

Architecture:

MFCC → BiLSTM → Dense → Softmax

Output:

speech_emotion_model.h5

8. Train Text Model
python models/text_pipeline/train.py

Architecture:

Embedding → BiLSTM → Dense → Softmax

Outputs:

text_emotion_model.keras
tokenizer.pkl
label_encoder.pkl

9. Train Fusion Model
python models/fusion_pipeline/train.py

Architecture:

Speech Features + Text Features
↓
Concatenation Fusion
↓
Dense Layer
↓
Softmax

Output:

fusion_emotion_model.keras

10. Generate Visualizations
python plots/visualize_embeddings.py

This generates:

Speech t-SNE plot
Text t-SNE plot
Fusion t-SNE plot

Outputs:

Results/speech_tsne.png
Results/text_tsne.png
Results/fusion_tsne.png

Results
Model	Accuracy
Speech-only	99.55%
Text-only	4.64%
Fusion	99.73%

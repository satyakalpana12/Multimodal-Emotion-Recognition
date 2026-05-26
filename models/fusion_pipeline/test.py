import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load data
df = pd.read_csv("dataset/dataframe.csv")
speech_X = np.load("dataset/speech_X.npy")
speech_X = speech_X.transpose(0, 2, 1)

texts = df["text"].astype(str).values
labels = df["emotion"].values

# Load tokenizer and label encoder
with open("models/text_pipeline/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/text_pipeline/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

y_encoded = label_encoder.transform(labels)

# Same split as training
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

speech_test = speech_X[test_idx]
text_test = texts[test_idx]
y_test = y_encoded[test_idx]

# Tokenize text
MAX_LEN = 10
text_test_seq = tokenizer.texts_to_sequences(text_test)
text_test_pad = pad_sequences(text_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# Load model
model = load_model("models/fusion_pipeline/fusion_emotion_model.keras")

# Evaluate
loss, accuracy = model.evaluate([speech_test, text_test_pad], y_test)
print(f"Fusion Model Accuracy: {accuracy * 100:.2f}%")
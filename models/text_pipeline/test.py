import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load data
df = pd.read_csv("dataset/dataframe.csv")
texts = df["text"].astype(str).values
labels = df["emotion"].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)

# Same split as training
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Load tokenizer
with open("models/text_pipeline/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 10

# Tokenize
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# Load model
model = load_model("models/text_pipeline/text_emotion_model.keras")

# Evaluate
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Text Model Accuracy: {accuracy * 100:.2f}%")
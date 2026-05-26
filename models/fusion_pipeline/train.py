import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_csv("dataset/dataframe.csv")
speech_X = np.load("dataset/speech_X.npy")

# Make speech shape suitable for LSTM: (samples, time_steps, features)
speech_X = speech_X.transpose(0, 2, 1)

texts = df["text"].astype(str).values
labels = df["emotion"].values

# Load tokenizer and label encoder from text pipeline
with open("models/text_pipeline/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/text_pipeline/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

y_encoded = label_encoder.transform(labels)

# Split using same indices for both modalities
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

speech_train, speech_test = speech_X[train_idx], speech_X[test_idx]
text_train, text_test = texts[train_idx], texts[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# Text to sequences
MAX_LEN = 10
text_train_seq = tokenizer.texts_to_sequences(text_train)
text_test_seq = tokenizer.texts_to_sequences(text_test)

text_train_pad = pad_sequences(text_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
text_test_pad = pad_sequences(text_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# Model inputs
speech_input = Input(shape=(speech_X.shape[1], speech_X.shape[2]), name="speech_input")
text_input = Input(shape=(MAX_LEN,), name="text_input")

# Speech branch
s = Bidirectional(LSTM(64))(speech_input)
s = Dropout(0.3)(s)

# Text branch
vocab_size = min(5000, len(tokenizer.word_index) + 1)
t = Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LEN)(text_input)
t = Bidirectional(LSTM(32))(t)
t = Dropout(0.3)(t)

# Fusion
x = Concatenate()([s, t])
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(len(np.unique(y_encoded)), activation="softmax")(x)

model = Model(inputs=[speech_input, text_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    [speech_train, text_train_pad],
    y_train,
    validation_data=([speech_test, text_test_pad], y_test),
    epochs=20,
    batch_size=32
)

loss, accuracy = model.evaluate([speech_test, text_test_pad], y_test)
print(f"Fusion Model Accuracy: {accuracy * 100:.2f}%")

os.makedirs("models/fusion_pipeline", exist_ok=True)
model.save("models/fusion_pipeline/fusion_emotion_model.keras")

print("Fusion model saved successfully.")
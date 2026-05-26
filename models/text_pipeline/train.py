import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_csv("dataset/dataframe.csv")

texts = df["text"].astype(str).values
labels = df["emotion"].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

# Split data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Tokenizer
MAX_WORDS = 5000
MAX_LEN = 10

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# Model
vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
num_classes = len(np.unique(y_encoded))

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=20,
    batch_size=32
)

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Text Model Accuracy: {accuracy * 100:.2f}%")

# Save model and tokenizer
os.makedirs("models/text_pipeline", exist_ok=True)

model.save("models/text_pipeline/text_emotion_model.keras")

with open("models/text_pipeline/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/text_pipeline/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Text model saved successfully.")
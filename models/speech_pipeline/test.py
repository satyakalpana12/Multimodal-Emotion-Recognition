import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load data
X = np.load("dataset/speech_X.npy")
y = np.load("dataset/speech_y.npy")

# Reshape for LSTM
X = X.transpose(0, 2, 1)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Load model
model = load_model("models/speech_pipeline/speech_emotion_model.h5")

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Speech Model Accuracy: {accuracy * 100:.2f}%")
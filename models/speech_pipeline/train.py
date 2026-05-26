import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load saved features
X = np.load("dataset/speech_X.npy")
y = np.load("dataset/speech_y.npy")

# Reshape for LSTM
X = X.transpose(0, 2, 1)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-hot encoding
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Build model
model = Sequential()

model.add(
    Bidirectional(
        LSTM(128, return_sequences=False),
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("models/speech_pipeline/speech_emotion_model.h5")

print("Model saved successfully.")
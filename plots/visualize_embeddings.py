import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("dataset/dataframe.csv")
speech_X = np.load("dataset/speech_X.npy").transpose(0, 2, 1)
texts = df["text"].astype(str).values
labels = df["emotion"].values

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

with open("models/text_pipeline/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 10
text_pad = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post", truncating="post")

speech_model = load_model("models/speech_pipeline/speech_emotion_model.h5")
text_model = load_model("models/text_pipeline/text_emotion_model.keras")
fusion_model = load_model("models/fusion_pipeline/fusion_emotion_model.keras")

speech_model.predict(np.zeros((1, speech_X.shape[1], speech_X.shape[2]), dtype=np.float32), verbose=0)
text_model.predict(np.zeros((1, MAX_LEN), dtype=np.int32), verbose=0)
fusion_model.predict([np.zeros((1, speech_X.shape[1], speech_X.shape[2]), dtype=np.float32), np.zeros((1, MAX_LEN), dtype=np.int32)], verbose=0)

speech_extractor = Model(inputs=speech_model.inputs, outputs=speech_model.layers[0].output)
text_extractor = Model(inputs=text_model.inputs, outputs=text_model.layers[1].output)
fusion_extractor = Model(inputs=fusion_model.inputs, outputs=fusion_model.layers[-2].output)

speech_features = speech_extractor.predict(speech_X, verbose=0)
text_features = text_extractor.predict(text_pad, verbose=0)
fusion_features = fusion_extractor.predict([speech_X, text_pad], verbose=0)

def plot_tsne(features, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=18)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, encoder.classes_, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    os.makedirs("Results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_tsne(speech_features, y, "Speech Embeddings", "Results/speech_tsne.png")
plot_tsne(text_features, y, "Text Embeddings", "Results/text_tsne.png")
plot_tsne(fusion_features, y, "Fusion Embeddings", "Results/fusion_tsne.png")

print("All plots saved")
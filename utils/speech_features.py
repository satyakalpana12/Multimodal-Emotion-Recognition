import os
import numpy as np
import librosa
from tqdm import tqdm

from data_prep import build_dataframe

DATASET_PATH = r"dataset/TESS"
MAX_LEN = 130 # length of audio
N_MFCC = 40 # diff features of audio like pitch , tone

def extract_mfcc(file_path, max_len=MAX_LEN, n_mfcc=N_MFCC):
    audio, sr = librosa.load(file_path, sr=22050)

    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

def create_speech_dataset(dataset_path):
    df = build_dataframe(dataset_path)

    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row["file_path"]
        emotion = row["emotion"]

        mfcc = extract_mfcc(file_path)
        X.append(mfcc)
        y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    return X, y, df

if __name__ == "__main__":
    X, y, df = create_speech_dataset(DATASET_PATH)

    ## for checking bext max length values
    lengths = []

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".wav"):

                path = os.path.join(root, file)

                audio, sr = librosa.load(path, sr=22050)

                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

                lengths.append(mfcc.shape[1])

    print("Min:", min(lengths))
    print("Max:", max(lengths))
    print("Average:", sum(lengths)/len(lengths))


    print("MFCC shape:", X.shape)
    print("Labels shape:", y.shape)

    np.save("dataset/speech_X.npy", X)
    np.save("dataset/speech_y.npy", y)

    df.to_csv("dataset/dataframe.csv", index=False)
    print("Saved speech features and dataframe.")
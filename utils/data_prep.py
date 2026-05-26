import os
import pandas as pd

DATASET_PATH = r"dataset/TESS"

def extract_info_from_filename(filename):

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 3:
        return None, None

    word = parts[1]
    emotion = parts[2]

    return word, emotion


def build_dataframe(dataset_path):

    data = []

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            if file.endswith(".wav"):

                file_path = os.path.join(root, file)

                word, emotion = extract_info_from_filename(file)

                if word is not None and emotion is not None:

                    data.append({
                        "file_path": file_path,
                        "text": word,
                        "emotion": emotion
                    })

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":

    df = build_dataframe(DATASET_PATH)

    print(df.head())

    print("\nTotal samples:", len(df))

    print("\nEmotion counts:")
    print(df["emotion"].value_counts())
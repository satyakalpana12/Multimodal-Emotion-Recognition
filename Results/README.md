# Multimodal Emotion Recognition

## Overview
This project builds an emotion recognition system using three approaches:
- Speech-only
- Text-only
- Multimodal fusion

The dataset used is TESS.

## Models
- Speech: MFCC + BiLSTM
- Text: Tokenization + Embedding + BiLSTM
- Fusion: Speech + Text concatenation

## Results
| Model | Accuracy |
|---|---|
| Speech | 99.55% |
| Text | 4.64% |
| Fusion | 99.73% |

## Visualization
The project includes t-SNE plots for:
- Speech embeddings
- Text embeddings
- Fusion embeddings

## Folder Structure
- `dataset/`
- `models/`
- `plots/`
- `Results/`
- `utils/`

## Run
```bash
python utils/data_prep.py
python utils/speech_features.py
python models/speech_pipeline/train.py
python models/text_pipeline/train.py
python models/fusion_pipeline/train.py
python plots/visualize_embeddings.py
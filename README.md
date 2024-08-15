# Hate Speech Detection in Videos and Memes 

This repository contains code for detecting hate speech in videos and memes using multimodal architectures. The project includes Baseline and MO-Hate architectures.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Running Baseline Experiments](#running-baseline-architecture-experiments)
- [Running MO-Hate Architecture Experiments](#running-mo-hate-architecture-experiments)
- [Testing on Specific Videos/Memes](#testing-on-specific-videos-or-memes)

## Dataset

### Download
The datasets used in this project are HateMM and Hateful Memes. You can download it from the following link:

- [HateMM Dataset](https://doi.org/10.5281/zenodo.7799469)
- Hateful Memes Dataset on [Kaggle](https://www.kaggle.com/datasets/chauri/facebook-hateful-memes) or [Hugging Face](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)

## Installation

To install the necessary packages, run the following command:

```sh
pip install -r requirements.txt
```

## Preprocessing

Before running the experiments, you need to preprocess the dataset to extract features. The preprocessing scripts are located in the [`Preprocessing/`]("Preprocessing/") directory. Make sure to specify the correct path to the downloaded datasets in the below code files.

1. **Video Frames and Audio Transcript**:
    ```sh
    python Preprocessing/frameExtract.py
    ```
    and then
    ```sh
    python Preprocessing/WhisperTranscript.py
    ```

2. **Audio Features**:
    ```sh
    python Preprocessing/AudioMFCC_Features.py
    ```
    or
    ```sh
    python Preprocessing/CLAP_and_Wav2Vec2_features.py
    ```

3. **Text Features**:
    ```sh
    python Preprocessing/BERT_HXP_Embeddings.py
    ```

4. **Image Features**:
    ```sh
    python Preprocessing/CLIP_image_features.py
    ```
    or
    ```sh
    python Preprocessing/DINOv2_image_features.py
    ```
    or
    ```sh
    python Preprocessing/ViT_Memes_Features.py
    ```

5. **Video Features**:
    ```sh
    python Preprocessing/ViT_VideoFrame_Features.py
    ```

## Running Baseline Experiments

The Baseline architecture experiments are implemented in the [`Baseline/`]("Baseline/") directory. To run the baseline experiments, follow these steps:

1. **Train the Baseline Model on HateMM**:
    ```sh
    python Baseline/HateMM_Fusion.py
    ```

2. **Train the Baseline Model on Hateful Memes**:
    ```sh
    python Baseline/HateMemesFusion.py
    ```

## Running MO-Hate Architecture Experiments

The MO-Hate architecture experiments are implemented in the [`MO-Hate/`]("MO-Hate/") directory. To run these experiments, follow these steps:

1. **Train the MO-Hate Model on HateMM**:
    ```sh
    python MO-Hate/main.py
    ```

2. **Evaluate the MO-Hate Model on Hateful Memes**:
    ```sh
    python MO-Hate/memes.py
    ```

## Testing on Specific Videos/Memes

Load the trained model for testing either on random or specific images from the test set:
    ```sh
    python test_videos.py
    ```
    or
    ```sh
    python test_memes.py
    ```

Some parts of the preprocessing and training codes for Baseline and MO-Hate have been taken from their respective GitHub repositories. Please refer to the following links for more details:
- [HateMM](https://github.com/hate-alert/HateMM)
- [MO-Sarcation](https://github.com/mohit2b/MO-Sarcation)

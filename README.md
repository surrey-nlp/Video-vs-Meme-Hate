# Hate Speech Detection in Memes and Videos

This repository contains code for detecting hate speech in memes and videos using various multimodal architectures. The project includes Baseline and MO-Hate architectures.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Running Baseline Experiments](#running-baseline-architecture-experiments)
- [Running MO-Hate Architecture Experiments](#running-mo-hate-architecture-experiments)

## Dataset

### Download
The dataset used in this project is the Hateful Memes dataset. You can download it from the following link:

- [HateMM Dataset](https://doi.org/10.5281/zenodo.7799469)
- [Hateful Memes Dataset](https://www.kaggle.com/datasets/chauri/facebook-hateful-memes) or (https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)

## Installation

To install the necessary packages, run the following command:

```sh
pip install -r requirements.txt
```

## Preprocessing

Before running the experiments, you need to preprocess the dataset to extract features. The preprocessing scripts are located in the [`Preprocessing/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fdiptesh%2Fworkspace%2FVideo-vs-Meme-Hate%2FPreprocessing%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/diptesh/workspace/Video-vs-Meme-Hate/Preprocessing/") directory.

1. **Audio Features**:
    ```sh
    python Preprocessing/AudioMFCC_Features.py
    ```

2. **Text Features**:
    ```sh
    python Preprocessing/BERT_HXP_Embeddings.py
    ```

3. **Image Features**:
    ```sh
    python Preprocessing/CLIP_image_features.py
    ```

4. **Video Features**:
    ```sh
    python Preprocessing/ViT_VideoFrame_Features.py
    ```

## Running Baseline Experiments

The baseline experiments are implemented in the [`Baseline/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fdiptesh%2Fworkspace%2FVideo-vs-Meme-Hate%2FBaseline%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/diptesh/workspace/Video-vs-Meme-Hate/Baseline/") directory. To run the baseline experiments, follow these steps:

1. **Train the Baseline Model**:
    ```sh
    python Baseline/HateMemesFusion.py
    ```

2. **Evaluate the Baseline Model**:
    ```sh
    python test_memes.py
    ```

## Running MO-Hate Architecture Experiments

The MO-Hate architecture experiments are implemented in the [`MO-Hate/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fdiptesh%2Fworkspace%2FVideo-vs-Meme-Hate%2FMO-Hate%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/diptesh/workspace/Video-vs-Meme-Hate/MO-Hate/") directory. To run these experiments, follow these steps:

1. **Train the MO-Hate Model**:
    ```sh
    python MO-Hate/main.py
    ```

2. **Evaluate the MO-Hate Model**:
    ```sh
    python test_videos.py
    ```

Some parts of the preprocessing and training codes for Baseline and MO-Hate have been taken from their respective GitHub repositories. Please refer to the following links for more details:
- [HateMM](https://github.com/hate-alert/HateMM)
- [MO-Sarcation](https://github.com/mohit2b/MO-Sarcation)

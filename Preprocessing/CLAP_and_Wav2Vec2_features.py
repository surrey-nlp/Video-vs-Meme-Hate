import os
import traceback
import pickle
import librosa
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, ClapAudioModel
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

FOLDER_NAME ='/backup/girish_datasets/HateMM/'

def extract_features(audio_path, feature_type):
    if feature_type == "CLAP":
        # Load the CLAP model and processor
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
        model = model.to("cuda")

        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        # Load the audio file
        audio, _ = librosa.load(audio_path, sr=48000)

        # Process the audio waveform
        outputs = processor(audios=audio, return_tensors="pt", sampling_rate=48000)
        features = model(**outputs).last_hidden_state

        # Convert tensors to numpy arrays for serialization
        features = features.cpu().detach().numpy()

    elif feature_type == "Wav2Vec2":
        # Load the Wav2Vec2 model and processor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model = model.to("cuda")

        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        # Load the audio file
        audio, _ = librosa.load(audio_path, sr=16000)

        # Chunk the audio into 30-second segments
        chunk_size = 30 * 16000  # 30 seconds * sample rate
        audio_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
        all_features = []
        for chunk in audio_chunks:
            # Process each audio chunk
            inputs = feature_extractor(chunk, return_tensors="pt", sampling_rate=16000).input_values.to("cuda")
            with torch.no_grad():
                features = model(input_values=inputs).last_hidden_state
                all_features.append(features.cpu().detach().numpy().squeeze(0))  # Remove the first dimension

        # Concatenate all chunk features along the first dimension
        features = np.concatenate(all_features, axis=0)

    else:
        print("Invalid feature type")
        return None

    return features


def extract_all_features(feature_type):
    FOLDER_NAME = '/backup/girish_datasets/HateMM/'
    with open(FOLDER_NAME + 'final_allNewData.pkl', 'rb') as fp:
        allDataAnnotation = pickle.load(fp)
        allVidList = list(allDataAnnotation.values())

    allAudioFeatures = {}
    failedList = []
    for audio_path in tqdm(allVidList):
        try:
            features = extract_features(audio_path, feature_type)
            video_name = os.path.basename(audio_path)
            allAudioFeatures[video_name.replace(".wav", ".mp4")] = features
        except Exception as e:
            failedList.append(audio_path)
            print(f"Failed to extract features for {audio_path}")
            print(f"Error: {e}")
            traceback.print_exc()

    return allAudioFeatures, failedList


# Example usage
allAudioFeatures, failedList = extract_all_features("CLAP")
print(failedList)

# Save the features
with open(FOLDER_NAME + 'CLAP_features.pkl', 'wb') as fp:
    pickle.dump(allAudioFeatures, fp)

# Uncomment the following code to extract Wav2Vec2 features
# allAudioFeatures, failedList = extract_all_features("Wav2Vec2")
# print(failedList)

# Save the features
# with open(FOLDER_NAME + 'Wav2Vec2_features_chunked.pkl', 'wb') as fp:
#     pickle.dump(allAudioFeatures, fp)
import os
import librosa
import numpy as np
import librosa.display
from tqdm import tqdm
import pickle
import traceback

FOLDER_NAME ='/backup/girish_datasets/HateMM/'

def extract_mfcc(path):
    try:
        audio, sr = librosa.load(path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros(40)

audio_plots_path = os.path.join(FOLDER_NAME, "Audio_plots") 

# Ensure the Audio_plots directory exists
os.makedirs(audio_plots_path, exist_ok=True)

with open(FOLDER_NAME+'final_allNewData.pkl', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)
    allVidList = list(allDataAnnotation.values())

allAudioFeatures = {}
failedList = []

for i in tqdm(allVidList):
    try:
        aud = extract_mfcc(i)
        # Extract the base name without extension as the key
        video_name = os.path.basename(i)
        allAudioFeatures[video_name.replace(".wav", ".mp4")] = aud
    except Exception as e:
        print(f"Error processing {i}: {e}")
        traceback.print_exc()  # This will print the stack trace
        failedList.append(i)

for i in failedList:
    allAudioFeatures[i] = np.zeros(40)

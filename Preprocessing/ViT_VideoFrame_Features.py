
import os
import numpy as np
import torch
from PIL import Image
import pickle
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

FOLDER_NAME = '/backup/girish_datasets/HateMM/'
VITF_FOLDER = os.path.join(FOLDER_NAME, 'VITF')

# Create the VITF directory if it does not exist
if not os.path.exists(VITF_FOLDER):
    os.makedirs(VITF_FOLDER)
    print(f"Created directory: {VITF_FOLDER}")
else:
    print(f"Directory already exists: {VITF_FOLDER}")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

minFrames = 100
begin_frame, end_frame, skip_frame = 0, minFrames, 0

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

import pickle
with open(FOLDER_NAME+'final_allImageFrames.pkl', 'rb') as fp:
    allVidList = pickle.load(fp)

data_image_path = "/backup/girish_datasets/HateMM_Images/"

def extract_vit_features(video_frames):
    # Extract a unique identifier for the video, assuming the first frame's path can be used for this
    # This line extracts the video's folder name from the first frame's path
    video_id = os.path.basename(os.path.dirname(video_frames[0]))

    # Construct the filename for the pickle file using the video_id
    pickle_filename = os.path.join(VITF_FOLDER, f"{video_id}_vit.pkl")

    if os.path.exists(pickle_filename):
        return

    try:
        video = read_images(video_frames)
        inputs = feature_extractor(images=video, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        video_features = [(last_hidden_states[i][0].detach().numpy()) for i in range(0, 100)]

        with open(pickle_filename, 'wb') as fp:
            pickle.dump(video_features, fp)
    except Exception as e:
        print(e)
        return

def read_images(frame_paths):
    X = []
    currFrameCount = 0
    videoFrameCount = len(frame_paths)
    if videoFrameCount <= minFrames:
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            X.append(image)
            currFrameCount += 1
            if currFrameCount == minFrames:
                break
        paddingImage = Image.fromarray(np.zeros((100, 100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount += 1
    else:
        step = int(videoFrameCount / minFrames)
        for i in range(0, videoFrameCount, step):
            image = Image.open(frame_paths[i])
            X.append(image)
            currFrameCount += 1
            if currFrameCount == minFrames:
                break
        paddingImage = Image.fromarray(np.zeros((100, 100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount += 1
    return X

# Iterate over all video frames and extract ViT features
for video_frames in tqdm(allVidList):
    extract_vit_features(video_frames)